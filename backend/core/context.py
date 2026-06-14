"""
RAG context resolution, book-list cache, small-talk detection,
internet check, and web-search fallback.
Owns all caches so that ingestion/startup can call the invalidation helpers
without importing from server_backend.
"""
import os
import socket
import time as _time_mod
import threading
import logging

logger = logging.getLogger("backend")

# ── DuckDuckGo availability ────────────────────────────────────────────────────
try:
    import ddgs as _ddgs_check  # noqa: F401
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False

# ── Clojure bridge config ─────────────────────────────────────────────────────
_CLOJURE_URL    = os.getenv("CLOJURE_REASONER_URL", "http://localhost:7654")
_ENABLE_CLOJURE = os.getenv("ENABLE_CLOJURE", "false").lower() == "true"

# ── Self-verify config ────────────────────────────────────────────────────────
_ENABLE_SELF_VERIFY = os.getenv("ENABLE_SELF_VERIFY", "false").lower() == "true"

# ── RAG result cache ──────────────────────────────────────────────────────────
_rag_cache:      dict[str, tuple[str, str, float]] = {}   # key → (ctx, src, ts)
_rag_cache_lock  = threading.Lock()
_RAG_CACHE_TTL   = 60    # seconds
_RAG_CACHE_MAX   = 200   # max entries before LRU eviction


def invalidate_rag_cache():
    with _rag_cache_lock:
        _rag_cache.clear()


def prune_rag_cache():
    cutoff = _time_mod.time() - _RAG_CACHE_TTL
    with _rag_cache_lock:
        expired = [k for k, v in _rag_cache.items() if v[2] < cutoff]
        for k in expired:
            del _rag_cache[k]
        if len(_rag_cache) > _RAG_CACHE_MAX:
            oldest = sorted(_rag_cache.items(), key=lambda x: x[1][2])
            for k, _ in oldest[:len(_rag_cache) - _RAG_CACHE_MAX]:
                del _rag_cache[k]


# ── Book-list cache (per-user dict) ──────────────────────────────────────────
_BOOK_LIST_TTL = 300   # refresh every 5 minutes
_book_list_lock = threading.Lock()

# Per-user caches: {user_id → (value_str, timestamp)}
_book_list_cache:   dict[str, tuple[str, float]] = {}
# Per-user books API cache: {user_id → (list, timestamp)}
_books_api_cache:   dict[str, tuple[list, float]] = {}


def invalidate_book_list_cache(user_id: str = ""):
    """Invalidate cache for a specific user (or all users if user_id is empty)."""
    with _book_list_lock:
        if user_id:
            _book_list_cache.pop(user_id, None)
            _books_api_cache.pop(user_id, None)
        else:
            _book_list_cache.clear()
            _books_api_cache.clear()


def get_books_api_cache(user_id: str = "") -> "list | None":
    """Return cached books list for a user if still fresh, else None."""
    with _book_list_lock:
        entry = _books_api_cache.get(user_id)
        if entry and (_time_mod.monotonic() - entry[1]) < _BOOK_LIST_TTL:
            return list(entry[0])
    return None


def set_books_api_cache(data: list, user_id: str = "") -> None:
    """Write the books API cache for a user."""
    with _book_list_lock:
        _books_api_cache[user_id] = (data, _time_mod.monotonic())


# Public alias for external modules that need the TTL constant
BOOK_LIST_TTL = _BOOK_LIST_TTL


def get_book_list(user_id: str = "") -> str:
    now = _time_mod.monotonic()
    with _book_list_lock:
        entry = _book_list_cache.get(user_id)
        if entry and (now - entry[1]) < _BOOK_LIST_TTL:
            return entry[0]
    try:
        from backend.core.state import hybrid_retriever
        if user_id:
            titles = sorted(hybrid_retriever._user_known_titles(user_id))
        else:
            titles = sorted(hybrid_retriever._known_titles) if hybrid_retriever._known_titles else []

        if not titles:
            # Fallback: scroll Qdrant filtered by user_id
            from backend.database.vector_db import scroll_user_books, scroll_all
            if user_id:
                metadatas = scroll_user_books(user_id)["metadatas"]
            else:
                metadatas = scroll_all()["metadatas"]
            titles = sorted({
                m.get("book_title") or m.get("source", "")
                for m in metadatas
                if (m.get("book_title") or m.get("source"))
                   and not m.get("raptor_level")
            } - {""})

        if titles:
            value = "الكتب المتاحة:\n" + "\n".join(f"- {t}" for t in titles)
            with _book_list_lock:
                _book_list_cache[user_id] = (value, _time_mod.monotonic())
            return value
    except Exception:
        pass
    return "الكتب المتاحة: لا يوجد كتب مرفوعة."


# ── Small-talk / library-query detection (extracted to smalltalk.py) ──────────
from backend.core.smalltalk import is_small_talk, is_library_query  # noqa: F401 (re-exported)


def ddgs_available() -> bool:
    """Public accessor for DuckDuckGo availability flag."""
    return _DDGS_AVAILABLE


# ── Internet availability check (cached 30 s) ─────────────────────────────────
_online_cache: bool | None = None
_online_checked_at: float  = 0.0
_online_lock = threading.Lock()


def is_online() -> bool:
    global _online_cache, _online_checked_at
    now = _time_mod.monotonic()
    with _online_lock:
        if _online_cache is not None and now - _online_checked_at < 30:
            return _online_cache
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.5)
        s.connect(("8.8.8.8", 53))
        s.close()
        result = True
    except Exception:
        result = False
    with _online_lock:
        _online_cache      = result
        _online_checked_at = now
    return result


# ── DuckDuckGo web search fallback ────────────────────────────────────────────
def web_search(query: str, num_results: int = 5) -> str:
    if not _DDGS_AVAILABLE:
        return ""
    try:
        from ddgs import DDGS
        with DDGS(timeout=6) as ddgs:
            hits = list(ddgs.text(query, max_results=num_results))
        parts = []
        for h in hits:
            title = (h.get("title") or "").strip()
            body  = (h.get("body")  or "").strip()[:500]
            if body:
                header = f"[{title}]" if title else "[نتيجة]"
                parts.append(f"{header}\n{body}")
        result = "\n\n---\n\n".join(parts)
        logger.info(f"[WEB] {len(hits)} results for: {query[:60]}")
        return result
    except Exception as e:
        logger.warning(f"Web search error: {e}")
        return ""


# ── Clojure context-processor bridge ─────────────────────────────────────────
def clojure_process_context(docs, query: str) -> str | None:
    if not _ENABLE_CLOJURE or not docs:
        return None
    try:
        import requests as _req
        payload = {
            "query": query,
            "chunks": [
                {
                    "content":      d.page_content,
                    "title":        d.metadata.get("book_title", ""),
                    "page":         d.metadata.get("page", ""),
                    "raptor_level": d.metadata.get("raptor_level"),
                }
                for d in docs
            ],
        }
        r = _req.post(f"{_CLOJURE_URL}/process", json=payload, timeout=2.0)
        if r.status_code == 200:
            ctx = r.json().get("context", "")
            if ctx:
                logger.debug(f"[Clojure] processed {len(docs)} chunks → {len(ctx)} chars")
                return ctx
    except Exception as e:
        logger.debug(f"[Clojure] unavailable: {e}")
    return None


# ── Self-verification loop ────────────────────────────────────────────────────
def self_verify_response(response: str, context: str, query: str) -> str:
    if not _ENABLE_SELF_VERIFY or not context or not response:
        return response
    verify_prompt = (
        "أنت مدقق علمي دقيق. راجع الجواب التالي وتحقق إذا كل ادعاء فيه مدعوم "
        "بالسياق المرفق من الكتب.\n\n"
        f"السؤال: {query}\n\n"
        f"السياق من الكتب:\n{context[:3000]}\n\n"
        f"الجواب المقترح:\n{response}\n\n"
        "تعليمات:\n"
        "• إذا الجواب دقيق ومدعوم بالسياق — أعده كما هو بدون أي تغيير.\n"
        "• إذا فيه معلومات مخترعة أو غير موجودة بالسياق — صحح الجواب "
        "واحذف الاختراع واذكر 'تصحيح: ...' في البداية.\n"
        "الجواب النهائي:"
    )
    try:
        import backend.llm.offline_llm as _llm
        verified = _llm.chat("أنت مدقق علمي. أجب بالعربية.", verify_prompt)
        if verified and len(verified) > 50:
            logger.debug(f"[Verify] response verified ({len(verified)} chars)")
            return verified
    except Exception as e:
        logger.debug(f"[Verify] failed: {e}")
    return response


# ── Context resolution (books → web → none) ──────────────────────────────────
# Words that carry no standalone topic: question words, anaphora, and
# continuation/elaboration markers. A query made up ONLY of these refers back to
# the ongoing topic; if any *content* word survives, the query stands on its own
# (e.g. "شنو هي الكسور الجزئية؟" → "الكسور الجزئية" remains → NOT a follow-up).
_FOLLOWUP_STOPWORDS = frozenset((
    # question words
    "شنو", "شو", "وش", "ايش", "إيش", "ما", "ماذا", "ماهو", "ماهي", "ماهية",
    "كيف", "شلون", "اشلون", "ليش", "لماذا", "متى", "اين", "وين", "هل", "كم", "من",
    # anaphora / continuation / elaboration markers
    "مثال", "امثلة", "أمثلة", "مثل", "اكمل", "أكمل", "كمل", "كمله", "استمر",
    "وضح", "وضّح", "اشرح", "اشرحه", "اشرحها", "بالتفصيل", "تفصيل", "اكثر", "أكثر",
    "زياده", "زيادة", "يعني", "ذلك", "هذا", "هذه", "هاي", "هاد", "بسطها", "بسّط",
    "اعطني", "أعطني", "عطني", "زيدني", "زود", "كمان", "ايضا", "أيضا", "اخر", "آخر",
    "ثاني", "تاني", "عليه", "عليها", "عنه", "عنها", "له", "لها", "هو", "هي", "هم",
    # generic particles
    "على", "عن", "في", "الى", "إلى", "و", "يا", "لي", "حول", "عند", "ال", "بشكل",
    # filler adjectives that commonly modify "مثال/شرح" in a follow-up
    "بسيط", "بسيطة", "سهل", "سهلة", "سريع", "سريعة", "صغير", "صغيرة", "قصير",
    "قصيرة", "واضح", "واضحة", "عملي", "عملية", "حقيقي", "اخرى", "أخرى", "جديد",
    "جديدة", "اضافي", "إضافي", "اضافية", "مفصل", "مفصلة", "اوسع", "دقيق", "تفصيلي",
    "اوضح", "أوضح", "افضل", "أفضل", "اعمق", "أعمق", "اكتر",
))


def _strip_punct(tok: str) -> str:
    return tok.strip("؟?!،,.؛;:()[]\"'")


def _is_followup_query(query: str) -> bool:
    """True if the query has NO standalone topic of its own — i.e. every token is
    a question word, anaphora, or continuation marker — so it needs the previous
    turn's topic/book to retrieve correctly. A query with any content word (even a
    short one like "الكسور") is treated as a new question, not a follow-up."""
    content = [
        t for t in (_strip_punct(w) for w in query.split())
        if len(t) > 2 and t not in _FOLLOWUP_STOPWORDS
    ]
    return len(content) == 0


def resolve_context(query: str, user_id: str = "", prev_user_text: str = "",
                    preferred_book: str = "") -> tuple[str, str]:
    """Returns (context_text, source) — source in {'small_talk','books','web','none'}.

    `prev_user_text` (recent user turns) anchors retrieval on the conversation
    topic when the current query is a short/anaphoric follow-up — without it,
    queries like "أعطني مثالاً" drift to an unrelated book.
    `preferred_book` (the session's current book) boosts that book for the same
    follow-ups — session affinity, a second guard against topic drift.
    """
    if is_small_talk(query):
        return "", "small_talk"

    # For follow-ups, fold the recent topic into the SEARCH query only (the LLM
    # still answers the user's literal question; small-talk/language checks above
    # already ran on the original query).
    is_followup  = _is_followup_query(query)
    search_input = query
    if prev_user_text and is_followup:
        search_input = f"{prev_user_text} {query}"
    book_hint = preferred_book if (preferred_book and is_followup) else None

    cache_key = f"{user_id}:{search_input.strip().lower()}"
    now = _time_mod.time()
    with _rag_cache_lock:
        hit = _rag_cache.get(cache_key)
        if hit and now - hit[2] < _RAG_CACHE_TTL:
            logger.debug(f"[RAG-CACHE] hit for '{cache_key[:40]}'")
            return hit[0], hit[1]

    from backend.core.state import hybrid_retriever
    from backend.rag.hybrid_retriever import prepare_query
    search_q = prepare_query(search_input)

    if _ENABLE_CLOJURE:
        context, docs, found = hybrid_retriever.get_context_with_docs(
            search_q, user_id=user_id, book_hint=book_hint)
        if found:
            clojure_ctx = clojure_process_context(docs, query)
            result = ((clojure_ctx if clojure_ctx else context), "books")
        else:
            result = None
    else:
        context, found = hybrid_retriever.get_context(
            search_q, user_id=user_id, book_hint=book_hint)
        result = (context, "books") if found else None

    if result is None:
        if is_online():
            logger.info("[CTX] No book match — trying web search")
            web_ctx = web_search(query)
            if web_ctx:
                logger.info(f"[CTX] Web fallback: {len(web_ctx)} chars")
                result = web_ctx, "web"
        if result is None:
            result = "", "none"

    with _rag_cache_lock:
        _rag_cache[cache_key] = (*result, now)
        if len(_rag_cache) > _RAG_CACHE_MAX:
            oldest = min(_rag_cache.items(), key=lambda x: x[1][2])
            del _rag_cache[oldest[0]]

    return result
