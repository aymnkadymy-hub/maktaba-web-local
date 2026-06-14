"""PDF ingestion engine — runs in a background thread, updates progress tracker."""
import os
import time as _time_mod
import threading
import logging

logger = logging.getLogger("backend")

# ── Ingestion progress tracker ────────────────────────────────────────────────
_ingest_progress:      dict[str, dict] = {}
_ingest_progress_lock  = threading.Lock()


def prune_ingest_progress():
    cutoff = _time_mod.time() - 3600
    with _ingest_progress_lock:
        stale = [
            name for name, info in _ingest_progress.items()
            if info.get("status") in ("done", "failed")
            and info.get("ts", 0) < cutoff
        ]
        for name in stale:
            del _ingest_progress[name]
    if stale:
        logger.debug(f"Pruned {len(stale)} old ingest-progress entries")


# ── PDF compression helper ────────────────────────────────────────────────────

def _compress_pdf(file_path: str) -> None:
    """
    Rewrite the PDF with PyMuPDF garbage-collection + deflate compression.
    Reduces file size 20-60 % with zero quality loss for text-based PDFs.
    Runs in-place: writes to a temp file then replaces the original.
    Non-fatal — any error is logged and the original file is left intact.
    """
    import fitz
    tmp_path = file_path + "._compress_tmp.pdf"
    try:
        doc = fitz.open(file_path)
        doc.save(
            tmp_path,
            garbage=4,      # remove unused objects + cross-references
            deflate=True,   # zlib-compress content streams
            clean=True,     # sanitise content streams
        )
        doc.close()
        os.replace(tmp_path, file_path)
        logger.info(f"PDF compressed: {os.path.basename(file_path)}")
    except Exception as e:
        logger.debug(f"PDF compression skipped ({e})")
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Core ingestion function ───────────────────────────────────────────────────
def run_ingestion(file_path: str, user_id: str = "") -> list:
    import fitz
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from backend.utils.arabic_normalizer import smart_normalize
    from backend.database.vector_db import vector_store
    from backend.utils.ingestion_ledger import (
        compute_sha256, mark_started, mark_complete, mark_failed
    )
    from backend.core.context import invalidate_book_list_cache

    name      = os.path.splitext(os.path.basename(file_path))[0]
    book_hash = compute_sha256(file_path)

    doc         = fitz.open(file_path)
    total_pages = len(doc)
    mark_started(name, book_hash, total_pages, user_id)
    with _ingest_progress_lock:
        _ingest_progress[name] = {
            "status": "ingesting", "pages_done": 0,
            "total_pages": total_pages, "ts": _time_mod.time(),
        }

    pages: list[Document] = []
    skipped_pages = 0
    try:
        for i in range(total_pages):
            try:
                page = doc[i]
                text = smart_normalize(page.get_text("text"))
            except Exception as page_err:
                logger.warning(f"[{name}] page {i+1} unreadable ({page_err}) — skipping")
                skipped_pages += 1
                with _ingest_progress_lock:
                    _ingest_progress[name]["pages_done"] = i + 1
                continue
            if text and len(text) > 20:
                pages.append(Document(
                    page_content=text,
                    metadata={
                        "source":     file_path,
                        "book_title": name,
                        "page":       i + 1,
                        "author":     "غير محدد",
                        "user_id":    user_id,
                    },
                ))
            with _ingest_progress_lock:
                _ingest_progress[name]["pages_done"] = i + 1
    finally:
        doc.close()

    if skipped_pages:
        logger.warning(f"[{name}] {skipped_pages} corrupted pages skipped out of {total_pages}")

    if not pages:
        msg = f"No readable pages extracted (all {total_pages} pages were corrupt/empty)"
        mark_failed(name, msg, user_id)
        raise ValueError(msg)

    _native_chunker = None
    try:
        import native_engine as _ne
        _native_chunker = _ne.S24DocumentOptimizer()
    except Exception:
        pass

    if _native_chunker is not None:
        from langchain_core.documents import Document as _Doc
        chunks: list = []
        for page_doc in pages:
            parts = _native_chunker.smart_chunk(page_doc.page_content, 500, 100)
            for part in parts:
                if part.strip():
                    chunks.append(_Doc(page_content=part, metadata=page_doc.metadata))
        logger.debug(f"smart_chunk (C++): {len(chunks)} chunks from {len(pages)} pages")
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "؟", "!", "،", "؛", " ", ""],
        )
        chunks = splitter.split_documents(pages)

    with _ingest_progress_lock:
        _ingest_progress[name]["total_chunks"] = len(chunks)
        _ingest_progress[name]["chunks_done"]  = 0

    try:
        _BATCH = 500
        for _i in range(0, len(chunks), _BATCH):
            vector_store.add_documents(chunks[_i:_i + _BATCH])
            with _ingest_progress_lock:
                _ingest_progress[name]["chunks_done"] = min(_i + _BATCH, len(chunks))
    except Exception as e:
        mark_failed(name, str(e), user_id)
        with _ingest_progress_lock:
            _ingest_progress[name] = {
                "status": "failed", "error": str(e),
                "total_pages": total_pages, "ts": _time_mod.time(),
            }
        raise

    mark_complete(name, len(chunks), user_id)
    invalidate_book_list_cache(user_id)
    with _ingest_progress_lock:
        _ingest_progress[name] = {
            "status": "done", "pages_done": total_pages,
            "total_pages": total_pages, "chunks": len(chunks), "ts": _time_mod.time(),
        }
    logger.info(f"Ingested {len(chunks)} chunks from '{name}' ({total_pages} pages)")

    _raptor_inline = os.getenv("RAPTOR_DURING_INGEST", "true").lower() == "true"
    if _raptor_inline:
        try:
            from backend.rag.raptor import ingest_raptor
            _RAPTOR_CAP  = 2000
            raptor_input = chunks[:_RAPTOR_CAP] if len(chunks) > _RAPTOR_CAP else chunks
            added = ingest_raptor(raptor_input, name, vector_store, max_levels=1)
            if added:
                logger.info(f"RAPTOR: {added} chunks for '{name}'")
        except Exception as e:
            logger.warning(f"RAPTOR ingestion failed (non-fatal): {e}")

    fire_ingest_webhook(name, len(chunks), total_pages)

    # Return leaf chunks so the caller can do incremental BM25/FAISS updates
    # without an additional Qdrant scroll_all.
    return chunks


# ── Webhook on ingest completion ─────────────────────────────────────────────
_INGEST_WEBHOOK_URL = os.getenv("INGEST_WEBHOOK_URL", "")


def fire_ingest_webhook(book_name: str, chunks: int, pages: int):
    if not _INGEST_WEBHOOK_URL:
        return
    def _send():
        try:
            import requests as _req
            _req.post(
                _INGEST_WEBHOOK_URL,
                json={"event": "ingest_done", "book": book_name, "chunks": chunks, "pages": pages},
                timeout=5,
            )
            logger.debug(f"[WEBHOOK] fired ingest_done for '{book_name}'")
        except Exception as e:
            logger.debug(f"[WEBHOOK] failed: {e}")
    threading.Thread(target=_send, daemon=True).start()
