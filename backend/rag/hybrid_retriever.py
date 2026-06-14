"""
Full RAG retrieval pipeline — المكتبة الناطقة v3

Query flow:
  1. prepare_query()     — strip Arabic question openers, normalise alef
  2. expand_query()      — multi-query template variants (0 ms, offline)
  3. _bm25_search()      — keyword exact/partial match   (bm25s ≫ rank-bm25)
  4. _vector_search()    — semantic MMR search           (ONNX, ~15 ms)
     ↑ repeated for every query variant
  5. retrieve_with_hyde()— hypothetical passage search   (Ollama, optional)
  6. _rrf_fuse_multi()   — Reciprocal Rank Fusion across all result lists
  7. rerank()            — cross-encoder joint scoring   (~50 ms, multilingual)

Environment overrides:
  ENABLE_HYDE=true          enable HyDE (requires Ollama, adds ~1-2 s)
  ENABLE_RERANKER=false     disable cross-encoder reranking
  RERANKER_MODEL=<hf-id>    override reranker model
"""
import os
import re
import json
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from langchain_core.documents import Document
from qdrant_client.models import Filter, FieldCondition, MatchValue

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", os.path.join(PROJECT_ROOT, "bm25_cache"))


def _get_native_pool():
    """Return the C++ thread pool singleton, or None if native_engine is unavailable."""
    try:
        from backend.rag.native_embeddings import _native_pool
        return _native_pool
    except Exception:
        return None

logger = logging.getLogger("hybrid_retriever")

_ENABLE_HYDE        = os.getenv("ENABLE_HYDE",        "true").lower()  == "true"
_ENABLE_RERANKER    = os.getenv("ENABLE_RERANKER",    "true").lower()  == "true"
_ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "true").lower()  == "true"   # P1: enabled
_HYDE_TIMEOUT       = float(os.getenv("HYDE_TIMEOUT", "1.5"))   # P4: 0.4 → 1.5 s
# Minimum cosine similarity to accept retrieved docs as "relevant".
# Below this threshold get_context() returns ("", False) and the LLM falls back
# to web search or "not found" rather than hallucinating from off-topic chunks.
_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.30"))  # P5: 0.20 → 0.30

# Cross-lingual retrieval: translate the query to the other language and add it
# as an extra search variant. Without this, an Arabic question can't reach an
# English-content book via BM25 (keyword search doesn't cross scripts) and only
# the weak bi-encoder carries it. With a translated variant, BM25 + the
# bi-encoder both match the book in its own language, then the multilingual
# reranker picks the best — so answers come from the most relevant book
# regardless of its language.
_ENABLE_QUERY_TRANSLATION = os.getenv("ENABLE_QUERY_TRANSLATION", "true").lower() == "true"
_XLATE_TIMEOUT = float(os.getenv("QUERY_TRANSLATION_TIMEOUT", "8.0"))
_XLATE_CACHE: "dict[str, str]" = {}
_XLATE_CACHE_MAX = 2000
_AR_RE = __import__("re").compile(r'[؀-ۿ]')


def _translate_query(query: str) -> str:
    """Return the query translated to the other language (AR↔EN), or "" on
    failure/timeout. Cached. Short and best-effort — retrieval still works with
    the original query alone if this returns ""."""
    if not _ENABLE_QUERY_TRANSLATION:
        return ""
    q = query.strip()
    if len(q) < 4 or len(q) > 300:
        return ""
    cached = _XLATE_CACHE.get(q)
    if cached is not None:
        return cached

    is_arabic = len(_AR_RE.findall(q)) >= 2
    target    = "English" if is_arabic else "Arabic"
    out = ""
    try:
        import backend.llm.offline_llm as _llm
        raw = _llm.chat(
            f"Translate the user's text to {target}. Output ONLY the translation, "
            f"no quotes, no explanation.",
            q, num_predict=60, num_ctx=512, timeout=int(_XLATE_TIMEOUT),
        )
        raw = (raw or "").strip().strip('"').strip()
        # Guard against the model echoing the question or refusing
        if raw and raw.lower() != q.lower() and len(raw) < 400:
            out = raw
    except Exception as e:
        logger.debug(f"[XLATE] translation skipped: {e}")

    if len(_XLATE_CACHE) > _XLATE_CACHE_MAX:
        _XLATE_CACHE.clear()
    _XLATE_CACHE[q] = out
    return out


# ── Text normalisation ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Arabic normalisation for BM25 tokenisation."""
    text = re.sub(r'[ً-ٰٟ]', '', text)   # strip tashkeel
    text = re.sub(r'[أإآٱ]', 'ا', text)   # unify alef
    text = re.sub(r'ى',      'ي', text)   # unify ya
    text = re.sub(r'ة',      'ه', text)   # unify ta-marbuta
    text = re.sub(r'ـ',      '',  text)   # remove tatweel
    return text.lower()


# Arabic stop words (after _normalize — alef unified, tashkeel stripped)
_AR_STOPWORDS: List[str] = [
    # prepositions / conjunctions
    'في', 'من', 'الي', 'علي', 'عن', 'مع', 'بين', 'حتي', 'لكن',
    'او', 'ان', 'اذا', 'لو', 'بل', 'ثم',
    # pronouns / demonstratives
    'هو', 'هي', 'هم', 'هن', 'انا', 'نحن', 'انت',
    'هذا', 'هذه', 'ذلك', 'تلك', 'الذي', 'التي', 'الذين',
    # auxiliaries
    'كان', 'كانت', 'يكون', 'تكون', 'لا', 'ما', 'لم', 'لن', 'قد',
    'كل', 'بعض', 'جميع', 'غير',
    # compound prepositions (normalised)
    'وفي', 'وعلي', 'ومن', 'وهو', 'وهي', 'وان',
]


def _tokenize_fallback(text: str) -> List[str]:
    """rank-bm25 tokeniser with stop word filtering."""
    tokens = re.split(r'[^\w؀-ۿ]+', _normalize(text))
    return [t for t in tokens if len(t) > 1 and t not in _AR_STOPWORDS]


def prepare_query(query: str) -> str:
    """
    Remove Arabic / English question openers, append a normalised (hamza-free)
    variant for BM25, and surface any quoted English terms so BM25 can match them.
    """
    q = query.strip()
    q = re.sub(
        r'^(ما هو|ما هي|كيف يعمل|كيف تعمل|كيف|لماذا|متى|أين|من هو|من هي'
        r'|ماذا|اشرح|شرح|عرّف|عرف|اذكر|ما معنى|ما المقصود'
        r'|مفهوم|لخص|تعريف|ما دور|ما وظيفة|وظيفة|دور'
        # Iraqi-dialect openers — without these, "شنو هي X" kept dialect noise
        # that weakened retrieval vs the MSA-stripped form
        r'|شنو هي|شنو هو|شنو يعني|شنو هيه|شنو|شلون|شگد|چم|ليش|وين|يعني شنو|گلي شنو|اگلي)\s+',
        '', q, flags=re.IGNORECASE,
    )
    norm = re.sub(r'[أإآٱ]', 'ا', q)
    # Extract quoted English/Latin terms (e.g. 'The Stepper') and append them
    # so BM25 can match them in English-language book content.
    quoted = re.findall(r"['\"‘’“”]([\w][^\'\"‘’“”]{1,40})['\"‘’“”]", q)
    extras = " ".join(quoted) if quoted else ""
    base   = f"{q} {norm}" if norm != q else q
    return f"{base} {extras}".strip() if extras else base


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Orchestrates the full 7-step retrieval pipeline.
    All stages degrade gracefully — missing deps / offline LLM never crash.
    """

    RRF_K     = 60
    DEFAULT_K = int(os.getenv("RAG_K", "6"))   # env-configurable: 6=fast, 10=comprehensive
    MAX_CHARS = 200

    def __init__(
        self,
        vector_store,
        k:               int  = DEFAULT_K,
        use_multi_query: bool = _ENABLE_MULTI_QUERY,
        use_hyde:        bool = _ENABLE_HYDE,
        use_reranker:    bool = _ENABLE_RERANKER,
    ):
        self.vector_store    = vector_store
        self.k               = k
        self.use_multi_query = use_multi_query
        self.use_hyde        = use_hyde
        self.use_reranker    = use_reranker

        self._corpus:       List[Document] = []
        self._bm25                         = None
        self._bm25_lib: str                = "none"  # "bm25s" | "rank_bm25" | "none"
        self._known_titles: Set[str]        = set()
        # user_id → ((corpus_id, corpus_len), titles) — see _user_known_titles
        self._user_titles: dict             = {}
        # user_id → (corpus_state, bm25_index, user_corpus) — see _get_user_bm25.
        # Per-user sub-indices so BM25 ranks within a user's own books instead of
        # a global corpus one user can dominate (97%/2%/1% split observed live).
        self._user_bm25: dict               = {}
        self._user_bm25_lock                = threading.Lock()

        # Build BM25 in background so the server starts immediately.
        # _bm25_search() returns [] while building — FAISS handles vector search until ready.
        t = threading.Thread(target=self._build_bm25, daemon=True, name="bm25-build")
        t.start()

    # ── BM25 index ────────────────────────────────────────────────────────────

    def _build_bm25(self):
        # Detect available library (bm25s preferred)
        _has_bm25s = False
        try:
            import bm25s  # noqa: F401
            _has_bm25s = True
        except ImportError:
            pass

        _has_rank = False
        if not _has_bm25s:
            try:
                from rank_bm25 import BM25Okapi  # noqa: F401
                _has_rank = True
            except ImportError:
                pass

        if not _has_bm25s and not _has_rank:
            logger.warning("No BM25 library found — install bm25s or rank-bm25")
            return

        try:
            # ── Fast path: load from disk cache ──────────────────────────────
            if _has_bm25s and self._load_bm25_cache():
                self._known_titles = {
                    m.get("book_title", "") for m in
                    (d.metadata for d in self._corpus) if m.get("book_title")
                } - {""}
                return

            # ── Slow path: build from Qdrant ──────────────────────────────────
            from backend.database.vector_db import scroll_all
            res   = scroll_all(include_documents=True)
            texts = res.get("documents") or []
            metas = res.get("metadatas") or []
            if not texts:
                logger.info("BM25: vector store empty, index skipped")
                return

            self._corpus = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metas, strict=False)
                if t and t.strip() and m.get("fact_type") != "user_fact"
            ]

            if _has_bm25s:
                import bm25s as _b
                # Filter docs that are empty after normalization before indexing.
                # bm25s internally skips empty strings and produces fewer arrays than
                # the corpus list, causing a numpy broadcast shape error at index time.
                normed_raw = [_normalize(d.page_content) for d in self._corpus]
                pairs = [(d, n) for d, n in zip(self._corpus, normed_raw) if n.strip()]
                if not pairs:
                    logger.warning("BM25: all documents empty after normalization — index skipped")
                    return
                pre_len      = len(self._corpus)
                self._corpus = [p[0] for p in pairs]
                normed       = [p[1] for p in pairs]
                if len(self._corpus) < pre_len:
                    logger.info(f"BM25: filtered {pre_len - len(self._corpus)} empty docs before indexing")
                tokens = _b.tokenize(normed, stopwords=_AR_STOPWORDS, show_progress=False)
                self._bm25 = _b.BM25()
                self._bm25.index(tokens, show_progress=False)
                self._bm25_lib = "bm25s"
                self._save_bm25_cache()
            else:
                from rank_bm25 import BM25Okapi
                tokenized  = [_tokenize_fallback(d.page_content) for d in self._corpus]
                self._bm25 = BM25Okapi(tokenized)
                self._bm25_lib = "rank_bm25"

            self._known_titles = {
                d.metadata.get("book_title", "") for d in self._corpus
                if d.metadata.get("book_title")
            } - {""}

            logger.info(
                f"BM25 index ({self._bm25_lib}): {len(self._corpus)} chunks"
            )
        except Exception as e:
            logger.warning(f"BM25 index build failed: {e}", exc_info=True)

    def rebuild(self, new_docs: "list | None" = None):
        """
        Rebuild BM25 + update FAISS after new documents are ingested.

        Incremental path (new_docs provided, BM25 already loaded):
          • BM25:  appends new_docs to in-RAM corpus, re-tokenises from RAM
                   (~5-10 s instead of 30 s Qdrant scroll_all)
          • FAISS: scrolls ONLY the new book's vectors (~1500 vs 329K),
                   adds them to the existing HNSW index in background (~3-5 s)

        Full rebuild path (new_docs=None OR BM25 not loaded):
          • Used after book deletion or first-ever build.
          • Invalidates both caches and rebuilds from scratch.
        """
        from backend.database.vector_db import faiss_index

        book_title = (
            new_docs[0].metadata.get("book_title")
            if new_docs else None
        )

        # Per-user sub-indices are keyed on the corpus (id,len); both paths below
        # change the corpus, so drop them to free memory and force a clean rebuild
        # on next query (the state key would invalidate them anyway).
        with self._user_bm25_lock:
            self._user_bm25.clear()

        if new_docs and self._bm25 is not None and self._corpus:
            # ── Incremental path ───────────────────────────────────────────────
            self._incremental_bm25_update(new_docs)
            if book_title:
                faiss_index.add_vectors_in_background(book_title)
        else:
            # ── Full rebuild path ──────────────────────────────────────────────
            self._bm25         = None
            self._corpus       = []
            self._bm25_lib     = "none"
            self._known_titles = set()
            self._invalidate_bm25_cache()
            self._build_bm25()
            faiss_index.invalidate()
            faiss_index.build_in_background()

    def _incremental_bm25_update(self, new_docs: list) -> None:
        """
        Extend in-RAM corpus with new_docs and rebuild BM25 from RAM.
        Avoids Qdrant scroll_all (30 s) — re-tokenises from in-RAM corpus (~5-10 s).
        """
        import time as _t
        try:
            import bm25s as _b
        except ImportError:
            logger.warning("bm25s not installed — falling back to full BM25 rebuild")
            self._invalidate_bm25_cache()
            self._build_bm25()
            return

        t0 = _t.perf_counter()

        clean_new = [
            d for d in new_docs
            if d.page_content and d.page_content.strip()
            and d.metadata.get("fact_type") != "user_fact"
        ]
        if not clean_new:
            return

        # Extend corpus + known_titles in place (no Qdrant I/O)
        self._corpus.extend(clean_new)
        self._known_titles.update(
            {d.metadata.get("book_title") for d in clean_new
             if d.metadata.get("book_title")} - {""}
        )

        # Re-tokenise entire corpus from RAM — faster than Qdrant scroll_all
        normed = [_normalize(d.page_content) for d in self._corpus]
        pairs  = [(d, n) for d, n in zip(self._corpus, normed) if n.strip()]
        self._corpus = [p[0] for p in pairs]
        normed       = [p[1] for p in pairs]

        tokens = _b.tokenize(normed, stopwords=_AR_STOPWORDS, show_progress=False)
        bm25   = _b.BM25()
        bm25.index(tokens, show_progress=False)

        self._bm25     = bm25
        self._bm25_lib = "bm25s"
        self._save_bm25_cache()

        logger.info(
            f"BM25 incremental: +{len(clean_new)} docs → "
            f"total={len(self._corpus)} in {(_t.perf_counter()-t0)*1000:.0f}ms"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _find_mentioned_book(query: str, known_titles: Set[str]) -> Optional[str]:
        """
        Return the book title if it appears in the query. Matching is done on the
        Arabic-normalised forms (ة/ه, أإآ/ا, ى/ي, tashkeel) so a title like
        "الكسور الجزئيه" is still detected in "ما هي الكسور الجزئية؟". Sorts by
        length descending so longer titles match before shorter sub-titles.
        """
        if not known_titles:
            return None
        q = _normalize(query)
        for title in sorted(known_titles, key=len, reverse=True):
            if _normalize(title) in q:
                return title
        return None

    def _user_known_titles(self, user_id: str) -> set:
        """Return known book titles for a specific user.

        Cached per user — a full-corpus scan on every chat request costs tens
        of ms once the corpus reaches 100K+ chunks. The cache key is
        (id, len) of the corpus list: every mutation either reassigns the
        list (new id) or extends it (new len), so stale entries self-expire
        without touching the mutation sites.
        """
        state  = (id(self._corpus), len(self._corpus))
        cached = self._user_titles.get(user_id)
        if cached is not None and cached[0] == state:
            return cached[1]
        titles = {
            d.metadata.get("book_title", "") for d in self._corpus
            if d.metadata.get("user_id") == user_id and d.metadata.get("book_title")
        } - {""}
        if len(self._user_titles) > 10_000:   # bound memory across many users
            self._user_titles.clear()
        self._user_titles[user_id] = (state, titles)
        return titles

    def _biencoder_score_pairs(self, pairs):
        """Lightweight relevance scorer using the embedding model already loaded
        for vector search — cosine similarity of (query, passage). Lets the
        self-calibrating gate run on devices that cannot host the heavy
        cross-encoder (no torch needed). Returns [] on failure."""
        try:
            emb = self.vector_store.embeddings
            qv = emb.embed_documents([p[0] for p in pairs])
            dv = emb.embed_documents([p[1] for p in pairs])
            out = []
            for a, b in zip(qv, dv):
                a = np.asarray(a, dtype=np.float32)
                b = np.asarray(b, dtype=np.float32)
                out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
            return out
        except Exception as e:
            logger.debug(f"[GATE] bi-encoder scoring failed: {e}")
            return []

    def book_chunks(self, book_title: str, user_id: Optional[str] = None,
                    limit: int = 24) -> List[Document]:
        """Return an evenly-spaced, diverse sample of a book's own chunks
        straight from the in-RAM corpus — no embedding, no relevance gate.
        Used by quiz generation, which needs BROAD coverage of the whole book
        (the relevance gate is the wrong tool there — it narrows to a query).
        Skips the first 2 chunks (usually cover/TOC/preface) for deeper content."""
        docs = [d for d in self._corpus
                if d.metadata.get("book_title") == book_title
                and (user_id is None or d.metadata.get("user_id") == user_id)
                and d.metadata.get("fact_type") != "user_fact"
                and not d.metadata.get("raptor_level")
                and len((d.page_content or "").strip()) > 80]
        if len(docs) > 4:
            docs = docs[2:]                      # drop cover/preface noise
        if len(docs) <= limit:
            return docs
        step = len(docs) / limit                 # even spread across the book
        return [docs[int(i * step)] for i in range(limit)]

    def _tenant_gate(self, user_id: Optional[str]):
        """Per-tenant calibrated relevance cutoff. Returns (mode, cutoff):
          - ('xenc', score)  → cross-encoder cutoff, applied during rerank
          - ('bienc', cos)   → embedding cosine cutoff, applied in vector search
                               (used when the reranker isn't available — e.g. a
                               low-end device with no torch)
          - (None, None)     → no per-tenant cutoff (fall back to global defaults)
        """
        if not user_id:
            return None, None
        try:
            from backend.rag.relevance_gate import gate, XENC_PROFILE, ScorerProfile
            _, u_corpus = self._get_user_bm25(user_id)
            if not u_corpus:
                u_corpus = [d for d in self._corpus
                            if d.metadata.get("user_id") == user_id]
            if not u_corpus:
                return None, None
            state = (id(self._corpus), len(self._corpus))
            if self.use_reranker:
                return "xenc", gate.cutoff_for(user_id, u_corpus, state, XENC_PROFILE)
            bienc = ScorerProfile("bi-encoder", self._biencoder_score_pairs,
                                  _RELEVANCE_THRESHOLD, 0.0, 0.95)
            return "bienc", gate.cutoff_for(user_id, u_corpus, state, bienc)
        except Exception as e:
            logger.debug(f"[GATE] tenant cutoff skipped: {e}")
            return None, None

    def _get_user_bm25(self, user_id: str):
        """Return (bm25_index, user_corpus) scoped to one user, built lazily.

        Ranking BM25 over a per-user sub-corpus is correct (a minority user's
        chunks can't be outranked off the result page by a 97%-corpus user) and
        faster (smaller index). Built on first query per user, cached in RAM,
        and auto-invalidated when the global corpus changes via the (id, len)
        state key. Returns (None, None) for the rank_bm25 fallback or on error,
        so the caller drops back to the global+filter path.
        """
        if self._bm25_lib != "bm25s" or not self._corpus:
            return None, None
        state  = (id(self._corpus), len(self._corpus))
        with self._user_bm25_lock:
            cached = self._user_bm25.get(user_id)
            if cached is not None and cached[0] == state:
                return cached[1], cached[2]
        try:
            import bm25s as _b
            user_corpus = [d for d in self._corpus
                           if d.metadata.get("user_id") == user_id]
            if not user_corpus:
                with self._user_bm25_lock:
                    self._user_bm25[user_id] = (state, None, [])
                return None, None
            normed = [_normalize(d.page_content) for d in user_corpus]
            pairs  = [(d, n) for d, n in zip(user_corpus, normed) if n.strip()]
            if not pairs:
                return None, None
            user_corpus = [p[0] for p in pairs]
            tokens = _b.tokenize([p[1] for p in pairs],
                                 stopwords=_AR_STOPWORDS, show_progress=False)
            idx = _b.BM25()
            idx.index(tokens, show_progress=False)
            with self._user_bm25_lock:
                if len(self._user_bm25) > 200:   # bound memory across many users
                    self._user_bm25.clear()
                self._user_bm25[user_id] = (state, idx, user_corpus)
            logger.info(f"[BM25] built per-user index: {len(user_corpus)} chunks "
                        f"for user={user_id[:8]}…")
            return idx, user_corpus
        except Exception as e:
            logger.warning(f"[BM25] per-user index build failed ({e}) — "
                           f"falling back to global+filter")
            return None, None

    def retrieve(self, query: str, k: Optional[int] = None,
                 active_filter: Optional[Filter] = None,
                 book_hint: Optional[str] = None,
                 use_multi_query: Optional[bool] = None,
                 user_id: Optional[str] = None,
                 _xlate_retry: bool = False) -> List[Document]:
        import time as _t
        _t_start = _t.perf_counter()

        k       = k or self.k
        # Over-fetch widely so small books (e.g. a 22-chunk title in a corpus of
        # thousands) still reach the reranker's candidate pool instead of being
        # crowded out by large books before reranking. The cross-encoder then
        # picks the truly relevant chunks, so a bigger pool helps recall without
        # hurting precision.
        fetch_n = k * 6

        # Step 1 — Query variants
        _use_mq = self.use_multi_query if use_multi_query is None else use_multi_query
        if _use_mq:
            from backend.rag.multi_query import expand_query
            queries = expand_query(query, use_llm=False)   # template-only, instant
        else:
            queries = [query]


        # Build user-aware active filter and known titles
        if active_filter is None:
            if user_id:
                active_filter = Filter(must=[
                    FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
                ], must_not=[
                    FieldCondition(key="metadata.fact_type", match=MatchValue(value="user_fact")),
                ])
            else:
                active_filter = self._BOOK_FILTER

        known = self._user_known_titles(user_id) if user_id else self._known_titles
        if book_hint is None:
            book_hint = self._find_mentioned_book(query, known)
        if book_hint:
            logger.debug(f"[RAG] book-hint: '{book_hint}' — extra BM25 boost, global search active")

        # Self-calibrating per-tenant relevance gate. With a reranker → an
        # 'xenc' cutoff applied at rerank; without one → a 'bienc' cosine cutoff
        # applied here at vector-search time (so the gate works on any device).
        gate_mode, gate_cutoff = self._tenant_gate(user_id)
        vec_threshold = gate_cutoff if gate_mode == "bienc" else None

        # Step 2 — BM25 + Vector for every variant (parallel via C++ thread pool)
        # Main BM25 always global (no book_title_hint) — extra book BM25 added after.
        candidate_lists: List[List[Document]] = []
        _t_search_start = _t.perf_counter()

        pool = _get_native_pool()
        if pool is not None:
            # Each query variant runs BM25+Vector concurrently on the C++ thread pool.
            # ORT releases the GIL during GPU compute, so BM25 overlaps with vector search.
            pair_results: List[List] = [[None, None] for _ in queries]
            done_events  = [threading.Event() for _ in queries]

            def _make_search_task(idx: int, q: str):
                def _task():
                    try:
                        pair_results[idx][0] = self._vector_search(q, fetch_n, active_filter, user_id=user_id, threshold=vec_threshold)
                        pair_results[idx][1] = self._bm25_search(q, fetch_n, None, user_id=user_id)
                    except Exception as _e:
                        logger.warning(f"[RAG] search task {idx} failed: {_e}")
                    finally:
                        done_events[idx].set()
                return _task

            for i, q in enumerate(queries):
                pool.send_to_pipe(_make_search_task(i, q))
            for ev in done_events:
                ev.wait(timeout=10.0)

            for v, b in pair_results:
                if v:
                    candidate_lists.append(v)
                if b:
                    candidate_lists.append(b)
        else:
            for q in queries:
                v = self._vector_search(q, fetch_n, active_filter, user_id=user_id, threshold=vec_threshold)
                b = self._bm25_search(q, fetch_n, None, user_id=user_id)
                if v:
                    candidate_lists.append(v)
                if b:
                    candidate_lists.append(b)

        # Extra book-specific BM25 list — boosts mentioned book in RRF
        if book_hint:
            extra = self._bm25_search(query, fetch_n, book_hint)
            if extra:
                candidate_lists.append(extra)
                logger.debug(f"[RAG] extra BM25 boost: {len(extra)} chunks from '{book_hint}'")

        logger.info(
            f"[RAG] variants={len(queries)} search={(_t.perf_counter()-_t_search_start)*1000:.0f}ms "
            f"pool={'C++' if pool is not None else 'sequential'}"
        )

        # Step 3 — HyDE with hard timeout (transparent: adds 0ms if Ollama absent,
        #          adds quality if Ollama is fast enough within _HYDE_TIMEOUT)
        if self.use_hyde:
            h = self._hyde_with_timeout(query, fetch_n)
            if h:
                candidate_lists.append(h)

        if not candidate_lists:
            return []

        # Step 4 — RRF fusion — feed a wide pool to the reranker (see fetch_n note)
        rrf_top = max(k * 6, 24)
        merged  = self._rrf_fuse_multi(candidate_lists, k=rrf_top)

        # Step 5 — Cross-encoder rerank (also applies the relevance cutoff, so
        # run it whenever there are candidates — not only when len > k — so an
        # off-topic query gets its weak matches filtered out, not just reordered)
        _t_rerank = _t.perf_counter()
        if self.use_reranker and merged:
            from backend.rag.reranker import rerank
            # 'xenc' cutoff from the self-calibrating gate; 'bienc' already
            # applied at vector-search time, so don't double-filter here.
            rr_cutoff = gate_cutoff if gate_mode == "xenc" else None
            result = rerank(query, merged, top_k=k, min_score=rr_cutoff)
            logger.info(
                f"[RAG] rerank={(_t.perf_counter()-_t_rerank)*1000:.0f}ms "
                f"total={(_t.perf_counter()-_t_start)*1000:.0f}ms kept={len(result)} "
                f"cutoff={'auto' if rr_cutoff is None else f'{rr_cutoff:.2f}'}"
            )
        else:
            logger.info(
                f"[RAG] total={(_t.perf_counter()-_t_start)*1000:.0f}ms (no rerank) "
                f"gate={'bienc %.3f'%gate_cutoff if gate_mode=='bienc' else 'off'}"
            )
            result = merged[:k]

        # Cross-lingual fallback (conditional — only when the same-language pass
        # found nothing). Translate the query to the other language and retry
        # once, so an Arabic question can still reach an English-content book
        # (and vice versa). Conditional keeps the ~translation latency off the
        # common case where the book is found in its own language.
        if (not result and _ENABLE_QUERY_TRANSLATION and not _xlate_retry):
            xlated = _translate_query(query)
            if xlated and xlated.lower() != query.lower():
                logger.info(f"[XLATE] no same-language match — retrying with: '{xlated[:40]}'")
                return self.retrieve(
                    xlated, k=k, active_filter=active_filter, book_hint=None,
                    use_multi_query=use_multi_query, user_id=user_id, _xlate_retry=True,
                )
        return result

    def _build_context_string(self, docs: List[Document]) -> Tuple[str, bool]:
        """
        Format a list of retrieved docs into a labeled context string.
        P13 ordering: RAPTOR summaries first, leaf chunks shortest→longest.
        """
        if not docs:
            return "", False

        raptor_docs = [d for d in docs if d.metadata.get("raptor_level")]
        leaf_docs   = [d for d in docs if not d.metadata.get("raptor_level")]
        leaf_docs.sort(key=lambda d: len(d.page_content))
        ordered_docs = raptor_docs + leaf_docs

        parts: List[str] = []
        for doc in ordered_docs:
            title = doc.metadata.get("book_title", "")
            page  = doc.metadata.get("page", "")
            lvl   = doc.metadata.get("raptor_level")

            if lvl:
                pr    = doc.metadata.get("page_range", "")
                label = (
                    f"[{title} — ملخص فصلي ص{pr} (مستوى {lvl})]"
                    if title and pr else
                    f"[{title} — ملخص (مستوى {lvl})]"
                )
            elif title and page:
                label = f"[{title} — ص{page}]"
            elif title:
                label = f"[{title}]"
            else:
                label = "[مرجع]"

            text = doc.page_content
            if len(text) > self.MAX_CHARS:
                cut = text[:self.MAX_CHARS]
                for sep in ('۔', '。', '.', '؟', '!', '\n'):
                    idx = cut.rfind(sep)
                    if idx > self.MAX_CHARS * 0.6:
                        cut = cut[:idx + 1]
                        break
                text = cut
            parts.append(f"{label}\n{text}")

        return "\n\n---\n\n".join(parts), True

    def get_context(self, query: str, k: Optional[int] = None,
                    user_id: Optional[str] = None,
                    book_hint: Optional[str] = None) -> Tuple[str, bool]:
        """
        Returns (context_string, found_in_books).
        When user_id is provided, retrieval is filtered to that user's books only.
        `book_hint` (e.g. the session's current book) boosts that book for short
        follow-up queries; a book explicitly named in the query takes priority.
        """
        known     = self._user_known_titles(user_id) if user_id else self._known_titles
        mentioned = self._find_mentioned_book(query, known) or book_hint
        docs = self.retrieve(query, k=k, book_hint=mentioned, user_id=user_id)
        if not docs:
            logger.debug("[RAG] all docs below threshold — no book context")
            return "", False
        return self._build_context_string(docs)

    def get_context_with_docs(
        self, query: str, k: Optional[int] = None,
        user_id: Optional[str] = None, book_hint: Optional[str] = None,
    ) -> Tuple[str, List[Document], bool]:
        """
        Like get_context() but also returns the raw docs list.
        Returns (context_string, docs, found_in_books).
        """
        known     = self._user_known_titles(user_id) if user_id else self._known_titles
        mentioned = self._find_mentioned_book(query, known) or book_hint
        docs = self.retrieve(query, k=k, book_hint=mentioned, user_id=user_id)
        if not docs:
            return "", [], False
        context, found = self._build_context_string(docs)
        return context, docs, found

    # ── Private helpers ───────────────────────────────────────────────────────

    def _hyde_with_timeout(self, query: str, fetch_n: int) -> List[Document]:
        """
        Run HyDE in a background task with a hard timeout.

        Uses the C++ thread pool when available (no ThreadPoolExecutor creation overhead).
        Falls back to a daemon thread if native_engine is absent.

        - Ollama not running → returns [] in < 50ms (ping cached as False)
        - Ollama fast (GPU)  → returns docs within _HYDE_TIMEOUT seconds
        - Ollama slow (CPU)  → times out, returns [] immediately
        """
        from backend.rag.hyde import retrieve_with_hyde
        result = [None]
        done   = threading.Event()

        def _run():
            try:
                result[0] = retrieve_with_hyde(query, self.vector_store, fetch_n)
            except Exception as e:
                logger.debug(f"HyDE error: {e}")
            finally:
                done.set()

        pool = _get_native_pool()
        if pool is not None:
            pool.send_to_pipe(_run)
        else:
            threading.Thread(target=_run, daemon=True).start()

        done.wait(timeout=_HYDE_TIMEOUT)
        return result[0] or []

    # Exclude user_fact documents from book retrieval.
    # Qdrant payload layout: {"page_content": "...", "metadata": {...}}
    # so fact_type lives at "metadata.fact_type".
    _BOOK_FILTER = Filter(
        must_not=[FieldCondition(key="metadata.fact_type", match=MatchValue(value="user_fact"))]
    )

    def _vector_search(self, query: str, n: int,
                       search_filter: Optional[Filter] = None,
                       user_id: Optional[str] = None,
                       threshold: Optional[float] = None) -> List[Document]:
        """
        Scored similarity search with relevance threshold filtering.
        Fast path: FAISS HNSW (C++ kernel, <5ms).
        Fallback: Qdrant local brute-force (Python, ~8-13s) while FAISS is building.

        `threshold` overrides the global cosine cutoff with a per-tenant value
        from the self-calibrating gate (used when no reranker is available).
        """
        import time as _t
        from backend.database.vector_db import faiss_index
        thr = _RELEVANCE_THRESHOLD if threshold is None else threshold

        # ── Fast path: FAISS HNSW ──────────────────────────────────────────────
        if faiss_index.ready:
            try:
                _t0 = _t.perf_counter()
                query_vec = self.vector_store.embeddings.embed_query(query)

                def filter_fn(meta, _uid=user_id):
                    if meta.get("fact_type") == "user_fact":
                        return False
                    return _uid is None or meta.get("user_id") == _uid
                results = faiss_index.search(
                    query_vec,
                    k=n,
                    filter_fn=filter_fn,
                    threshold=thr,
                )
                docs = [doc for doc, _score in results]
                logger.info(
                    f"[VEC] faiss={(_t.perf_counter()-_t0)*1000:.0f}ms results={len(docs)} thr={thr:.2f}"
                )
                return docs
            except Exception as e:
                logger.warning(f"[VEC] FAISS search failed, falling back to Qdrant: {e}")

        # ── Fallback: Qdrant local (slow, used while FAISS is building) ────────
        f = search_filter if search_filter is not None else self._BOOK_FILTER
        try:
            _t0 = _t.perf_counter()
            results = self.vector_store.similarity_search_with_score(query, k=n, filter=f)
            logger.info(f"[VEC] qdrant_search={(_t.perf_counter()-_t0)*1000:.0f}ms results={len(results)}")
            return [doc for doc, score in results if score >= thr]
        except Exception:
            try:
                return self.vector_store.similarity_search(query, k=n, filter=f)
            except Exception as e:
                logger.warning(f"Vector search error: {e}")
                return []

    def _bm25_search(self, query: str, n: int,
                     book_title: Optional[str] = None,
                     user_id: Optional[str] = None) -> List[Document]:
        """BM25 keyword search.

        When user_id is given, search that user's own sub-index so ranking is
        correct regardless of how the global corpus is split across users
        (otherwise a minority user's chunks get outranked off the page and
        post-filtering returns too few/zero results). Falls back to the global
        index + post-filter if the per-user index isn't available.
        """
        if not self._bm25 or not self._corpus:
            return []
        try:
            # ── Per-user sub-index (preferred when user_id provided) ──────────
            if user_id:
                u_idx, u_corpus = self._get_user_bm25(user_id)
                if u_idx is not None:
                    import bm25s as _b
                    q_tok   = _b.tokenize([_normalize(query)],
                                          stopwords=_AR_STOPWORDS, show_progress=False)
                    top_n   = min(n * (3 if book_title else 1), len(u_corpus))
                    res, sc = u_idx.retrieve(q_tok, k=top_n, show_progress=False)
                    results = [u_corpus[int(idx)]
                               for idx, score in zip(res[0], sc[0], strict=False)
                               if float(score) > 0]
                    if book_title:
                        results = [d for d in results
                                   if d.metadata.get("book_title") == book_title]
                    return results[:n]
                # else: fall through to global index + post-filter below

            # ── Global index + post-filter (no user_id, or fallback) ─────────
            multiplier = 3 if (book_title or user_id) else 1
            if self._bm25_lib == "bm25s":
                import bm25s as _b
                q_tok   = _b.tokenize(
                    [_normalize(query)], stopwords=_AR_STOPWORDS, show_progress=False
                )
                top_n   = min(n * multiplier, len(self._corpus))
                res, sc = self._bm25.retrieve(q_tok, k=top_n, show_progress=False)
                results = [
                    self._corpus[int(idx)]
                    for idx, score in zip(res[0], sc[0], strict=False)
                    if float(score) > 0
                ]
            else:   # rank_bm25
                tokens  = _tokenize_fallback(query)
                scores  = self._bm25.get_scores(tokens)
                top_idx = np.argsort(scores)[::-1][:n * multiplier]
                results = [self._corpus[i] for i in top_idx if scores[i] > 0]

            if user_id:
                results = [d for d in results if d.metadata.get("user_id") == user_id]
            if book_title:
                results = [d for d in results if d.metadata.get("book_title") == book_title]
            return results[:n]
        except Exception as e:
            logger.warning(f"BM25 search error: {e}")
            return []

    def _rrf_fuse_multi(
        self,
        lists: List[List[Document]],
        k: int,
    ) -> List[Document]:
        """Reciprocal Rank Fusion across N result lists."""
        rrf:     Dict[str, float]    = {}
        doc_map: Dict[str, Document] = {}

        import hashlib as _hl
        for doc_list in lists:
            for rank, doc in enumerate(doc_list):
                # MD5 of full content — no false collisions even for passages
                # that share a long common prefix (chapter intros, RAPTOR headers).
                # Non-cryptographic use; the flag also works under FIPS mode.
                key          = _hl.md5(doc.page_content.encode(),
                                       usedforsecurity=False).hexdigest()
                rrf[key]     = rrf.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
                doc_map[key] = doc

        sorted_keys = sorted(rrf, key=lambda x: rrf[x], reverse=True)
        return [doc_map[k] for k in sorted_keys[:k]]

    # ── BM25 disk cache ───────────────────────────────────────────────────────

    def _load_bm25_cache(self) -> bool:
        """Try to load BM25 index + corpus from disk. Returns True on success."""
        try:
            meta_path   = os.path.join(BM25_CACHE_DIR, "meta.json")
            corpus_path = os.path.join(BM25_CACHE_DIR, "corpus.json")
            index_path  = os.path.join(BM25_CACHE_DIR, "index")

            if not all(os.path.isfile(p) for p in [meta_path, corpus_path]):
                return False

            with open(meta_path) as f:
                meta = json.load(f)

            # Staleness check: compare stored Qdrant non-fact count with current
            from backend.database.vector_db import qdrant_client, COLLECTION_NAME
            current = qdrant_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(must_not=[
                    FieldCondition(key="metadata.fact_type", match=MatchValue(value="user_fact"))
                ]),
                exact=False,
            ).count

            if current != meta.get("qdrant_count", -1):
                logger.info(
                    f"BM25 cache stale (Qdrant={current} vs cache={meta.get('qdrant_count')}) — rebuilding"
                )
                return False

            with open(corpus_path, encoding="utf-8") as f:
                corpus_data = json.load(f)

            import bm25s as _b
            self._corpus   = [Document(page_content=d["c"], metadata=d["m"]) for d in corpus_data]
            self._bm25     = _b.BM25.load(index_path, load_corpus=False)
            self._bm25_lib = "bm25s"
            logger.info(f"BM25 cache loaded ({len(self._corpus)} chunks) — startup skipped Qdrant scroll")
            return True
        except Exception as e:
            logger.debug(f"BM25 cache load failed: {e}")
            return False

    def _save_bm25_cache(self):
        """Persist BM25 index + corpus to disk for fast future startups."""
        if self._bm25_lib != "bm25s" or not self._bm25:
            return
        try:
            from backend.database.vector_db import qdrant_client, COLLECTION_NAME
            os.makedirs(BM25_CACHE_DIR, exist_ok=True)

            current = qdrant_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(must_not=[
                    FieldCondition(key="metadata.fact_type", match=MatchValue(value="user_fact"))
                ]),
                exact=False,
            ).count

            self._bm25.save(os.path.join(BM25_CACHE_DIR, "index"))

            corpus_data = [{"c": d.page_content, "m": d.metadata} for d in self._corpus]
            with open(os.path.join(BM25_CACHE_DIR, "corpus.json"), "w", encoding="utf-8") as f:
                json.dump(corpus_data, f, ensure_ascii=False)

            with open(os.path.join(BM25_CACHE_DIR, "meta.json"), "w") as f:
                json.dump({"qdrant_count": current, "corpus_count": len(self._corpus)}, f)

            logger.info(f"BM25 cache saved ({len(self._corpus)} chunks)")
        except Exception as e:
            logger.warning(f"BM25 cache save failed: {e}")

    def _invalidate_bm25_cache(self):
        """Delete disk cache so next startup rebuilds from Qdrant."""
        import shutil
        try:
            for fname in ["meta.json", "corpus.json"]:
                p = os.path.join(BM25_CACHE_DIR, fname)
                if os.path.isfile(p):
                    os.remove(p)
            # Remove the index directory so bm25s starts with a clean slate on save.
            index_dir = os.path.join(BM25_CACHE_DIR, "index")
            if os.path.isdir(index_dir):
                shutil.rmtree(index_dir, ignore_errors=True)
        except Exception:
            pass

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a status dict for the /health endpoint."""
        return {
            "bm25_library":   self._bm25_lib,
            "bm25_chunks":    len(self._corpus),
            "multi_query":    self.use_multi_query,
            "hyde":           self.use_hyde,
            "reranker":       self.use_reranker,
        }
