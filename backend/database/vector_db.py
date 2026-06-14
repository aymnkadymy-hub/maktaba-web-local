"""
Vector store — Qdrant local mode (replaces ChromaDB).

Why Qdrant:
  ChromaDB uses in-memory HNSW + SQLite → collapses at ~2000 books (RAM).
  Qdrant stores vectors on disk with mmap, keeps search latency flat at any scale.
  Local mode (path=...) requires no server process.

Helpers exposed alongside `vector_store`:
  scroll_all()         — iterate all points (replaces ChromaDB .get())
  delete_by_title()    — filter-delete a whole book in one call
  count_by_title()     — chunk count for a given book_title
"""
import os
import sys
import logging
import warnings

# Suppress Qdrant's "Local mode >20k points" warning — we acknowledge the trade-off;
# migrating to server mode requires Docker which is optional for local dev.
warnings.filterwarnings("ignore", message=".*Local mode is not recommended.*", category=UserWarning)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    Filter, FieldCondition, MatchValue,
    FilterSelector,
)
from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger("vector_db")

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# QDRANT_URL: set to remote Qdrant URL for cloud/Docker deployment (e.g. http://qdrant:6333)
# Leave empty to use local file-based mode (default for local server).
QDRANT_URL      = os.getenv("QDRANT_URL", "")
QDRANT_PATH     = os.getenv("QDRANT_PATH", os.path.join(PROJECT_ROOT, "qdrant_data"))
COLLECTION_NAME = "books_collection"
EMBED_DIM       = 384   # paraphrase-multilingual-MiniLM-L12-v2 output dim

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Embedding model ───────────────────────────────────────────────────────────

def _build_embeddings():
    try:
        from backend.rag.native_embeddings import try_build_native_embeddings
        native = try_build_native_embeddings()
        if native is not None:
            logger.info("Embeddings: ONNX Runtime (native_engine) — MAX_LENGTH=512, batch=32")
            return native
    except Exception as e:
        logger.debug(f"NativeEmbeddings unavailable: {e}")

    logger.info("Embeddings: HuggingFace CPU")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ── Qdrant client + collection ────────────────────────────────────────────────

def _clear_stale_lock():
    """Remove a Qdrant .lock file left over from a hard kill.

    Must run BEFORE QdrantClient(path=...) — a stale lock makes the client
    constructor raise "already accessed by another instance" at import time.
    If a live process still holds the lock, the OS (Windows) refuses the
    delete and the client fails with the same clear error as before.
    """
    try:
        lock_file = os.path.join(QDRANT_PATH, ".lock")
        if os.path.isfile(lock_file):
            os.remove(lock_file)
            logger.info("Removed stale Qdrant .lock file from previous session")
    except Exception as e:
        logger.debug(f"Qdrant lock cleanup skipped: {e}")


def _init_client() -> QdrantClient:
    if QDRANT_URL:
        # Remote mode: cloud, Docker, or any external Qdrant instance
        client = QdrantClient(url=QDRANT_URL)
        logger.info(f"Qdrant: remote mode → {QDRANT_URL}")
    else:
        # Local file mode: zero-dependency, works offline
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _clear_stale_lock()
        client = QdrantClient(path=QDRANT_PATH)
        logger.info(f"Qdrant: local mode → {QDRANT_PATH}")

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Qdrant: created collection '{COLLECTION_NAME}'")
    else:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(
            f"Qdrant: loaded '{COLLECTION_NAME}' "
            f"({info.points_count} points)"
        )
    return client


embeddings    = _build_embeddings()
qdrant_client = _init_client()

vector_store  = QdrantVectorStore(
    client          = qdrant_client,
    collection_name = COLLECTION_NAME,
    embedding       = embeddings,
)


# ── Scroll helpers (replace ChromaDB .get()) ──────────────────────────────────

_SCROLL_BATCH = 500


def _scroll(scroll_filter=None, include_documents: bool = False) -> dict:
    """Shared scroll loop over all points matching `scroll_filter`.
    Returns {"ids": [...], "metadatas": [...][, "documents": [...]]}."""
    ids, metas, docs = [], [], []
    offset = None
    while True:
        hits, offset = qdrant_client.scroll(
            collection_name = COLLECTION_NAME,
            scroll_filter   = scroll_filter,
            limit           = _SCROLL_BATCH,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
        )
        for hit in hits:
            payload = hit.payload or {}
            ids.append(str(hit.id))
            metas.append(payload.get("metadata", {}))
            if include_documents:
                docs.append(payload.get("page_content", ""))
        if not offset:
            break

    result = {"ids": ids, "metadatas": metas}
    if include_documents:
        result["documents"] = docs
    return result


def scroll_all(include_documents: bool = False) -> dict:
    """Return all points. Equivalent to ChromaDB .get(include=[...])."""
    return _scroll(None, include_documents)


def scroll_book_chunks(title: str, include_documents: bool = False) -> dict:
    """Return all points belonging to a specific book_title."""
    return _scroll(Filter(must=[
        FieldCondition(key="metadata.book_title", match=MatchValue(value=title))
    ]), include_documents)


def scroll_user_books(user_id: str) -> dict:
    """Return all points belonging to a specific user (no vectors)."""
    return _scroll(Filter(must=[
        FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
    ]))


def _delete_where(conditions: list) -> int:
    """Count then delete all points matching the conditions. Returns count."""
    f = Filter(must=conditions)
    n = qdrant_client.count(
        collection_name=COLLECTION_NAME, count_filter=f, exact=True,
    ).count
    if n > 0:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(filter=f),
        )
    return n


def delete_by_title(title: str) -> int:
    """Delete all chunks for a book in one filter-based operation."""
    return _delete_where([
        FieldCondition(key="metadata.book_title", match=MatchValue(value=title)),
    ])


def delete_by_user_title(user_id: str, title: str) -> int:
    """Delete all chunks for a specific user's book. Returns number deleted."""
    return _delete_where([
        FieldCondition(key="metadata.user_id",    match=MatchValue(value=user_id)),
        FieldCondition(key="metadata.book_title", match=MatchValue(value=title)),
    ])


def count_all() -> int:
    return qdrant_client.count(collection_name=COLLECTION_NAME, exact=False).count


# ── FAISS in-memory HNSW index ────────────────────────────────────────────────
# QdrantLocal uses Python brute-force (O(N)) → 8-13s for 329K vectors.
# FAISS HNSW (C++ kernel) → <5ms for the same collection.
# Built once in background at startup, rebuilt after new ingestion.

import threading as _threading
import numpy as _np


_FAISS_CACHE_DIR = os.path.join(
    os.getenv("BM25_CACHE_DIR", os.path.join(PROJECT_ROOT, "bm25_cache")), "faiss"
)


class _FaissIndex:
    """
    Background-built HNSW index over all Qdrant vectors for sub-millisecond search.
    Disk-cached after first build — subsequent restarts load in <15s instead of 5 min.
    """

    def __init__(self):
        self._index    = None
        self._docs: list = []   # (page_content, metadata) at each FAISS position
        self._ready    = False
        self._building = False
        self._lock     = _threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    def build_in_background(self):
        """Start background build (idempotent). Tries disk cache first; rebuilds if stale."""
        with self._lock:
            if self._building or self._ready:
                return
            self._building = True
        t = _threading.Thread(target=self._build, daemon=True, name="faiss-build")
        t.start()

    def invalidate(self):
        """Mark stale (call after new ingestion). Deletes disk cache so next build is fresh."""
        import shutil
        with self._lock:
            self._ready    = False
            self._building = False
            self._index    = None
            self._docs     = []
        try:
            if os.path.isdir(_FAISS_CACHE_DIR):
                shutil.rmtree(_FAISS_CACHE_DIR, ignore_errors=True)
        except Exception:
            pass

    # ── Disk cache helpers ────────────────────────────────────────────────────

    def _cache_is_fresh(self) -> bool:
        import json
        meta_path = os.path.join(_FAISS_CACHE_DIR, "meta.json")
        if not os.path.isfile(meta_path):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            current  = qdrant_client.count(collection_name=COLLECTION_NAME, exact=False).count
            cached   = meta.get("qdrant_count", -1)
            # Tolerate up to 500 new vectors — user_facts and small RAPTOR batches
            # shouldn't trigger a full 410s FAISS rebuild (rebuild() in HybridRetriever
            # invalidates explicitly when a new book is ingested).
            return abs(current - cached) < 500
        except Exception:
            return False

    def _load_from_disk(self) -> bool:
        import time as _t
        import pickle
        try:
            import faiss
        except ImportError:
            return False
        index_path = os.path.join(_FAISS_CACHE_DIR, "index.bin")
        docs_path  = os.path.join(_FAISS_CACHE_DIR, "docs.pkl")
        if not (os.path.isfile(index_path) and os.path.isfile(docs_path)):
            return False
        t0 = _t.perf_counter()
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        with self._lock:
            self._index    = index
            self._docs     = docs
            self._ready    = True
            self._building = False
        logger.info(
            f"FAISS: loaded from disk cache in {_t.perf_counter()-t0:.1f}s "
            f"({index.ntotal} vectors) ✓"
        )
        return True

    def _save_to_disk(self, index, docs, qdrant_count: int):
        import pickle
        import json
        try:
            import faiss
        except ImportError:
            return
        try:
            os.makedirs(_FAISS_CACHE_DIR, exist_ok=True)
            faiss.write_index(index, os.path.join(_FAISS_CACHE_DIR, "index.bin"))
            with open(os.path.join(_FAISS_CACHE_DIR, "docs.pkl"), "wb") as f:
                pickle.dump(docs, f, protocol=4)
            with open(os.path.join(_FAISS_CACHE_DIR, "meta.json"), "w") as f:
                json.dump({"qdrant_count": qdrant_count}, f)
            logger.info(f"FAISS: index saved to disk cache ({len(docs)} docs)")
        except Exception as e:
            logger.warning(f"FAISS: disk cache save failed (non-fatal): {e}")

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        import time as _t
        try:
            import faiss  # noqa: F401
        except ImportError:
            logger.warning("faiss not installed — FAISS index disabled. pip install faiss-cpu")
            with self._lock:
                self._building = False
            return

        # Fast path: load from disk if cache is fresh
        if self._cache_is_fresh():
            logger.info("FAISS: disk cache fresh — loading…")
            if self._load_from_disk():
                return

        t0 = _t.perf_counter()
        logger.info("FAISS: building HNSW index from Qdrant local storage…")

        vectors = []
        docs    = []
        offset  = None
        n_total = 0

        while True:
            hits, next_offset = qdrant_client.scroll(
                collection_name = COLLECTION_NAME,
                limit           = 1000,
                offset          = offset,
                with_payload    = True,
                with_vectors    = True,
            )
            for hit in hits:
                vec = hit.vector
                if isinstance(vec, dict):
                    vec = next(iter(vec.values()), None)
                if not vec:
                    continue
                payload  = hit.payload or {}
                metadata = payload.get("metadata", {})
                content  = payload.get("page_content", "")
                vectors.append(vec)
                docs.append((content, metadata))

            n_total += len(hits)
            offset   = next_offset
            if not offset:
                break
            if n_total % 20000 == 0:
                logger.info(f"FAISS: {n_total} vectors loaded…")

        if not vectors:
            logger.warning("FAISS: no vectors found — index not built")
            with self._lock:
                self._building = False
            return

        t_load = _t.perf_counter()
        logger.info(f"FAISS: {n_total} vectors loaded in {t_load - t0:.1f}s — building HNSW…")

        import faiss as _faiss
        matrix = _np.array(vectors, dtype=_np.float32)
        norms  = _np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= _np.maximum(norms, 1e-9)

        dim   = matrix.shape[1]
        index = _faiss.IndexHNSWFlat(dim, 32, _faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch       = 64
        index.add(matrix)

        t_done = _t.perf_counter()

        # Save to disk for fast future restarts
        qdrant_count = qdrant_client.count(collection_name=COLLECTION_NAME, exact=False).count
        self._save_to_disk(index, docs, qdrant_count)

        with self._lock:
            self._index    = index
            self._docs     = docs
            self._ready    = True
            self._building = False

        logger.info(
            f"FAISS: HNSW ready — {n_total} vectors, "
            f"load={t_load-t0:.1f}s build={t_done-t_load:.1f}s total={t_done-t0:.1f}s ✓"
        )

    def add_vectors_in_background(self, book_title: str) -> None:
        """
        Incrementally add vectors for a newly ingested book — runs in a daemon thread.
        Scrolls ONLY the new book's vectors from Qdrant (~1500 vectors) instead of
        rebuilding from all 329K+.  Saves updated cache to disk when done.
        Call this instead of invalidate() + build_in_background() after book ingestion.
        """
        t = _threading.Thread(
            target=self._add_vectors_for_book,
            args=(book_title,),
            daemon=True,
            name=f"faiss-incr-{book_title[:20]}",
        )
        t.start()

    def _add_vectors_for_book(self, book_title: str) -> None:
        """Worker: scroll only new book vectors, add to existing HNSW index."""
        import time as _t
        if not self._ready or self._index is None:
            logger.info(f"FAISS: not ready — skipping incremental add for '{book_title}'")
            return

        t0 = _t.perf_counter()
        new_vectors: list = []
        new_docs:    list = []
        offset = None
        book_filter = Filter(must=[
            FieldCondition(key="metadata.book_title", match=MatchValue(value=book_title))
        ])

        while True:
            hits, next_offset = qdrant_client.scroll(
                collection_name = COLLECTION_NAME,
                scroll_filter   = book_filter,
                limit           = 1000,
                offset          = offset,
                with_payload    = True,
                with_vectors    = True,
            )
            for hit in hits:
                vec = hit.vector
                if isinstance(vec, dict):
                    vec = next(iter(vec.values()), None)
                if not vec:
                    continue
                payload  = hit.payload or {}
                metadata = payload.get("metadata", {})
                content  = payload.get("page_content", "")
                new_vectors.append(vec)
                new_docs.append((content, metadata))
            offset = next_offset
            if not offset:
                break

        if not new_vectors:
            logger.info(f"FAISS: no vectors found for '{book_title}' — incremental add skipped")
            return

        try:
            import faiss  # noqa: F401  — availability guard, not used directly
        except ImportError:
            return

        matrix = _np.array(new_vectors, dtype=_np.float32)
        norms  = _np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= _np.maximum(norms, 1e-9)

        with self._lock:
            self._index.add(matrix)
            self._docs.extend(new_docs)
            total = self._index.ntotal

        qdrant_count = qdrant_client.count(collection_name=COLLECTION_NAME, exact=False).count
        self._save_to_disk(self._index, self._docs, qdrant_count)

        logger.info(
            f"FAISS: incremental +{len(new_vectors)} vectors for '{book_title}' "
            f"in {_t.perf_counter()-t0:.1f}s (total={total}) ✓"
        )

    def search(
        self,
        query_vec: list,
        k: int,
        filter_fn=None,
        threshold: float = 0.0,
        fetch_multiplier: int = 8,
    ) -> list:
        """
        Returns list of (Document, cosine_score) sorted by score desc.
        Over-fetches to compensate for post-filtering.
        fetch_multiplier: increase for very selective filters (e.g. user_fact retrieval).
        """
        if not self._ready or self._index is None:
            return []

        from langchain_core.documents import Document

        q = _np.array(query_vec, dtype=_np.float32)
        q /= _np.maximum(_np.linalg.norm(q), 1e-9)
        q  = q.reshape(1, -1)

        with self._lock:
            # Re-check inside the lock: invalidate() (book delete/ingest) may
            # have nulled the index between the guard above and here.
            if self._index is None:
                return []
            # Snapshot the docs list reference under the lock. invalidate()
            # rebinds self._docs to a NEW empty list; binding `docs` here keeps
            # index access below consistent with the index we searched, instead
            # of racing to IndexError.
            docs = self._docs
            fetch_k = min(k * fetch_multiplier if filter_fn else k * 2, len(docs))
            distances, indices = self._index.search(q, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                break
            score = float(dist)
            if score < threshold:
                break  # sorted by score, can break early
            content, meta = docs[int(idx)]
            if filter_fn and not filter_fn(meta):
                continue
            results.append((Document(page_content=content, metadata=meta), score))
            if len(results) >= k:
                break

        return results


faiss_index = _FaissIndex()
