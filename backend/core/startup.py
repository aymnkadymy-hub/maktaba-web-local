"""
Application startup: DB schema, parallel warmups, background ingestion,
RAPTOR enrichment, and periodic cleanup thread.
"""
import os
import asyncio
import time as _time_mod
import threading
import collections
import logging

logger = logging.getLogger("backend")


# Stale-lock cleanup moved to backend/database/vector_db.py:_clear_stale_lock()
# — it must run BEFORE QdrantClient is constructed (which happens at import
# time), and running it here after the client was live deleted the *active*
# lock file on Linux, silently dropping the single-process protection.


async def startup(system_prompt: str):
    """Called from lifespan — runs warmups, kicks off background ingestion."""
    import backend.llm.offline_llm as llm
    llm.reset_cache()
    loop = asyncio.get_running_loop()

    from backend.api.notes import ensure_highlights_table
    from backend.api.flashcards import ensure_flashcards_table
    from backend.memory.storage import ensure_chat_schema
    await ensure_chat_schema()      # SQLite chat.db
    await ensure_highlights_table()
    await ensure_flashcards_table() # SQLite flashcards.db

    await asyncio.gather(
        loop.run_in_executor(None, _warm_up_embeddings),
        loop.run_in_executor(None, lambda: _warm_up_ollama(system_prompt)),
        loop.run_in_executor(None, _warm_up_reranker),
        loop.run_in_executor(None, _warm_up_books_cache),
    )

    # Start FAISS index build after warmup — replaces Qdrant's 8-13s Python brute-force
    # with <5ms HNSW search. Runs in background; Qdrant used as fallback until ready.
    try:
        from backend.database.vector_db import faiss_index
        faiss_index.build_in_background()
        logger.info("FAISS: background index build started")
    except Exception as e:
        logger.warning(f"FAISS index start failed (non-fatal): {e}")

    _ingest_loop = asyncio.get_running_loop()

    def _ingestion_done(t: asyncio.Task):
        if not t.exception():
            return
        logger.error(f"Background ingestion crashed: {t.exception()} — retrying in 60s")
        async def _delayed_retry():
            await asyncio.sleep(60)
            retry = _ingest_loop.create_task(_ingest_pending_books())
            retry.add_done_callback(_ingestion_done)
        _ingest_loop.create_task(_delayed_retry())

    task = _ingest_loop.create_task(_ingest_pending_books())
    task.add_done_callback(_ingestion_done)

    _start_cleanup_thread()

    from backend.core.context import ddgs_available
    if not ddgs_available():
        logger.warning("ddgs not installed — web search fallback disabled. pip install duckduckgo-search")


# ── Warmup helpers ────────────────────────────────────────────────────────────
def _warm_up_embeddings():
    try:
        from backend.database.vector_db import vector_store
        vector_store.similarity_search("warmup", k=1)
        logger.info("Embeddings warmed up — CUDA graph compiled")
    except Exception as e:
        logger.debug(f"Embeddings warm-up skipped: {e}")


def _warm_up_books_cache():
    """Populate books API cache from the BM25 corpus already in RAM."""
    import backend.core.context as _ctx
    from backend.core.state import hybrid_retriever
    try:
        if not hybrid_retriever._corpus:
            return
        leaf_counts:   dict = collections.defaultdict(int)
        raptor_counts: dict = collections.defaultdict(int)
        first_meta:    dict = {}
        for doc in hybrid_retriever._corpus:
            meta = doc.metadata
            if meta.get("fact_type"):
                continue
            title = meta.get("book_title") or meta.get("source", "")
            if not title:
                continue
            if meta.get("raptor_level"):
                raptor_counts[title] += 1
            else:
                leaf_counts[title] += 1
                if title not in first_meta:
                    first_meta[title] = meta
        cached_all = [
            {
                "title":         title,
                "filename":      first_meta[title].get("source", ""),
                "chunks":        leaf_counts[title],
                "raptor_chunks": raptor_counts[title],
            }
            for title in sorted(first_meta)
        ]
        _ctx.set_books_api_cache(cached_all)
        logger.info(f"Books cache warmed up from BM25 corpus — {len(cached_all)} books")
    except Exception as e:
        logger.debug(f"Books cache warm-up skipped: {e}")


def _warm_up_ollama(system_prompt: str):
    try:
        import backend.llm.offline_llm as _llm
        if not _llm._ping_ollama():
            return
        sess = _llm._get_http_session()
        sess.post(
            f"{_llm.OLLAMA_URL}/api/chat",
            json={
                "model": _llm.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": "مرحبا"},
                ],
                "stream":  False,
                "options": {**_llm._OLLAMA_OPTS, "num_predict": 1},
            },
            timeout=60,
        )
        logger.info(f"Ollama warmed up with system-prompt KV cache (num_ctx={_llm.NUM_CTX})")
    except Exception as e:
        logger.debug(f"Ollama warm-up skipped: {e}")


def _warm_up_reranker():
    import os
    if os.getenv("ENABLE_RERANKER", "true").lower() != "true":
        return  # skip warm-up when reranker is disabled
    try:
        from backend.rag.reranker import warm_up
        warm_up()
        _warm_up_relevance_gate()
    except Exception as e:
        logger.debug(f"Reranker warm-up: {e}")


def _warm_up_relevance_gate():
    """Pre-calibrate the per-tenant relevance gate for tenants already in the
    corpus, so the first query per tenant pays no calibration cost. Best-effort:
    calibration is cached, and any failure just defers it to first query."""
    try:
        from backend.core.state import hybrid_retriever as hr
        from backend.rag.relevance_gate import gate
        corpus = getattr(hr, "_corpus", None)
        if not corpus:
            return
        users = {d.metadata.get("user_id") for d in corpus if d.metadata.get("user_id")}
        state = (id(corpus), len(corpus))
        for uid in list(users)[:50]:   # bound startup work
            u_corpus = [d for d in corpus if d.metadata.get("user_id") == uid]
            if u_corpus:
                gate.cutoff_for(uid, u_corpus, state)
        if users:
            logger.info(f"[GATE] pre-calibrated {len(users)} tenant(s) at startup")
    except Exception as e:
        logger.debug(f"Relevance-gate warm-up: {e}")


# ── Background ingestion ──────────────────────────────────────────────────────
async def _ingest_pending_books():
    from backend.core.state import hybrid_retriever, BOOKS_DIR
    from backend.core.ingestion import run_ingestion
    from backend.core.context import invalidate_rag_cache
    from backend.utils.ingestion_ledger import compute_sha256, is_complete, is_failed

    if not os.path.isdir(BOOKS_DIR):
        return
    loop       = asyncio.get_running_loop()
    ingested_n = 0

    # Books are stored in books/{user_id}/book.pdf — scan all user subdirectories
    entries = sorted(os.listdir(BOOKS_DIR))
    for entry in entries:
        entry_path = os.path.join(BOOKS_DIR, entry)
        # Per-user subdirectory
        if os.path.isdir(entry_path):
            user_id = entry
            for fname in sorted(os.listdir(entry_path)):
                if not fname.lower().endswith(".pdf"):
                    continue
                fpath     = os.path.join(entry_path, fname)
                book_name = os.path.splitext(fname)[0]
                try:
                    book_hash = await loop.run_in_executor(None, compute_sha256, fpath)
                except Exception:
                    continue
                if is_complete(book_name, book_hash, user_id):
                    logger.info(f"✅ '{book_name}' ({user_id}) already ingested — skip")
                    continue
                if is_failed(book_name, book_hash, user_id):
                    logger.info(f"⛔ '{book_name}' ({user_id}) previously failed — skip")
                    continue
                logger.info(f"📖 Ingesting: '{book_name}' for user '{user_id}'")
                try:
                    await loop.run_in_executor(None, run_ingestion, fpath, user_id)
                    ingested_n += 1
                except Exception as e:
                    logger.error(f"Ingest failed '{book_name}': {e}")
        # Legacy: flat PDF directly in books/ (pre-per-user era — skip silently)
        # elif entry.lower().endswith(".pdf"):  # intentionally not re-ingested

    if ingested_n > 0:
        logger.info(f"🔄 Rebuilding BM25 index after {ingested_n} new books…")
        await loop.run_in_executor(None, hybrid_retriever.rebuild)
        invalidate_rag_cache()

    # Run RAPTOR in background after a delay — QdrantLocal is not thread-safe,
    # so running scroll_all() at startup serializes every chat search for 20-39s.
    asyncio.ensure_future(_delayed_raptor())


async def _delayed_raptor():
    """Schedule RAPTOR 5 minutes after startup — avoids Qdrant lock contention at boot."""
    await asyncio.sleep(300)
    logger.info("RAPTOR: starting background enrichment check (delayed 5 min)…")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _raptor_for_existing_books)


def _raptor_for_existing_books():
    try:
        import backend.llm.offline_llm as _llm
        if not _llm._ping_ollama():
            return

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from backend.database.vector_db import qdrant_client, COLLECTION_NAME, scroll_book_chunks
        from backend.rag.raptor import ingest_raptor
        from backend.core.state import hybrid_retriever
        from langchain_core.documents import Document

        # Use in-RAM BM25 corpus titles — avoids expensive scroll_all() (20s+ for 329K records)
        # QdrantLocal is not thread-safe; scroll_all() blocks every chat search while running.
        titles = set(hybrid_retriever._known_titles)

        if not titles:
            return

        from backend.database.vector_db import vector_store
        any_added = False
        for title in sorted(titles):
            raptor_count = qdrant_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(must=[
                    FieldCondition(key="metadata.book_title", match=MatchValue(value=title)),
                    FieldCondition(key="metadata.raptor_level", match=MatchValue(value=1)),
                ]),
                exact=True,
            ).count

            if raptor_count > 0:
                logger.info(f"✅ RAPTOR: '{title}' has {raptor_count} summaries — skip")
                continue

            logger.info(f"🌿 RAPTOR: building summaries for '{title}'...")
            res  = scroll_book_chunks(title, include_documents=True)
            docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(res.get("documents", []), res.get("metadatas", []))
                if not meta.get("raptor_level")
            ]
            if not docs:
                continue

            try:
                _RC = 2000
                raptor_docs = docs[:_RC] if len(docs) > _RC else docs
                added = ingest_raptor(raptor_docs, title, vector_store, max_levels=2)
                if added:
                    logger.info(f"RAPTOR: {added} summaries for '{title}' (input capped at {len(raptor_docs)})")
                    any_added = True
            except Exception as e:
                logger.warning(f"RAPTOR failed for '{title}' (non-fatal): {e}")

        if any_added:
            hybrid_retriever.rebuild()

    except Exception as e:
        logger.warning(f"RAPTOR startup check failed (non-fatal): {e}")


# ── Periodic cleanup thread ───────────────────────────────────────────────────
def _start_cleanup_thread():
    def _loop():
        while True:
            _time_mod.sleep(1800)
            from backend.core.session import prune_ram_history
            from backend.core.guards import prune_register_attempts, prune_chat_attempts, prune_dedup
            from backend.core.ingestion import prune_ingest_progress
            from backend.core.context import prune_rag_cache
            prune_ram_history()
            prune_register_attempts()
            prune_ingest_progress()
            prune_rag_cache()
            prune_chat_attempts()
            prune_dedup()
    threading.Thread(target=_loop, daemon=True, name="cleanup").start()
