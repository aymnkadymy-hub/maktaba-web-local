"""
Books management endpoints: /books, /books/{title}, /ingest, /ingest/progress, /ingest/status.
Registered in server_backend.py via make_books_router().
"""
import os
import asyncio
import threading
import collections
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse

logger = logging.getLogger("backend")


def make_books_router(get_current_user, admin_username: str):
    from backend.core.messages import _MSG
    from backend.core.state import hybrid_retriever, BOOKS_DIR
    from backend.core.context import (
        invalidate_book_list_cache, invalidate_rag_cache,
        get_books_api_cache, set_books_api_cache,
    )
    from backend.core.ingestion import (
        run_ingestion, _compress_pdf, _ingest_progress, _ingest_progress_lock,
    )
    from backend.database.vector_db import scroll_user_books, delete_by_user_title
    from backend.utils.ingestion_ledger import remove as _ledger_remove

    router = APIRouter()

    @router.get("/books")
    async def list_books(
        page:     int = 1,
        per_page: int = 50,
        search:   str = "",
        current=Depends(get_current_user),
    ):
        per_page = min(per_page, 200)
        try:
            loop = asyncio.get_running_loop()
            user_id = current.user_id

            cached_all = get_books_api_cache(user_id)

            if cached_all is None:
                metadatas = (await loop.run_in_executor(None, scroll_user_books, user_id))["metadatas"]
                leaf_counts:   dict[str, int]  = collections.defaultdict(int)
                raptor_counts: dict[str, int]  = collections.defaultdict(int)
                first_meta:    dict[str, dict] = {}

                for meta in metadatas:
                    if meta.get("fact_type"):
                        continue
                    title = meta.get("book_title") or meta.get("source", "غير معروف")
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
                set_books_api_cache(cached_all, user_id)

            all_books = cached_all
            if search:
                sq = search.strip().lower()
                all_books = [b for b in all_books if sq in b["title"].lower()]

            total = len(all_books)
            start = (page - 1) * per_page
            paged = all_books[start: start + per_page]
            return {
                "total":    total,
                "page":     page,
                "per_page": per_page,
                "pages":    max(1, (total + per_page - 1) // per_page),
                "books":    paged,
            }
        except Exception as e:
            logger.error(f"Books list error: {e}")
            raise HTTPException(status_code=500, detail="تعذّر تحميل قائمة الكتب")

    @router.delete("/books/{title}")
    async def delete_book(title: str, current=Depends(get_current_user)):
        user_id = current.user_id
        try:
            loop = asyncio.get_running_loop()
            n    = await loop.run_in_executor(None, delete_by_user_title, user_id, title)
            if n == 0:
                raise HTTPException(status_code=404, detail=_MSG.BOOK_NOT_FOUND(title))
            _ledger_remove(title, user_id)

            # Delete PDF from disk — path-traversal guard: verify resolved path
            # stays inside user_books_dir before deleting anything.
            user_books_dir = os.path.join(BOOKS_DIR, user_id)
            real_books_dir = os.path.realpath(user_books_dir)
            for ext in (".pdf", ".PDF"):
                pdf_path = os.path.join(user_books_dir, title + ext)
                real_pdf  = os.path.realpath(pdf_path)
                # Only delete if the resolved path starts with the user's dir
                if not real_pdf.startswith(real_books_dir + os.sep):
                    logger.warning(f"Path traversal blocked for title='{title}'")
                    break
                if os.path.isfile(real_pdf):
                    try:
                        os.remove(real_pdf)
                        logger.info(f"Deleted PDF: {real_pdf}")
                    except OSError as oe:
                        logger.warning(f"Could not delete PDF '{real_pdf}': {oe}")
                    break

            await loop.run_in_executor(None, hybrid_retriever.rebuild)
            invalidate_book_list_cache(user_id)
            invalidate_rag_cache()
            return {"message": f"✅ تم حذف '{title}' ({n} قطعة)."}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete book error: {e}")
            raise HTTPException(status_code=500, detail=_MSG.DELETE_FAILED(e))

    _MAX_BOOKS_PER_USER = 20

    @router.post("/ingest")
    async def ingest_book(file: UploadFile = File(...), current=Depends(get_current_user)):
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=_MSG.NOT_PDF)
        safe_name = os.path.basename(file.filename or "upload.pdf")
        if not safe_name or safe_name.startswith('.'):
            raise HTTPException(status_code=400, detail="اسم الملف غير صالح")

        user_id        = current.user_id
        user_books_dir = os.path.join(BOOKS_DIR, user_id)

        # Enforce per-user book limit before saving
        if os.path.isdir(user_books_dir):
            current_count = sum(
                1 for f in os.listdir(user_books_dir)
                if f.lower().endswith(".pdf")
            )
            if current_count >= _MAX_BOOKS_PER_USER:
                raise HTTPException(
                    status_code=400,
                    detail=f"book_limit_reached:{_MAX_BOOKS_PER_USER}",
                )

        save_path      = os.path.join(user_books_dir, safe_name)
        try:
            os.makedirs(user_books_dir, exist_ok=True)
            # Defense-in-depth: verify resolved path stays inside user_books_dir
            real_books = os.path.realpath(user_books_dir)
            real_save  = os.path.realpath(save_path)
            if not real_save.startswith(real_books + os.sep) and real_save != real_books:
                raise HTTPException(status_code=400, detail="اسم الملف غير صالح")
            header = await file.read(4)
            if header[:4] != b'%PDF':
                raise HTTPException(status_code=400, detail=_MSG.BAD_PDF_MAGIC)
            await file.seek(0)
            with open(save_path, "wb") as f:
                while chunk := await file.read(65536):
                    f.write(chunk)
            book_name = os.path.splitext(safe_name)[0]

            _compress_pdf(save_path)

            def _bg_ingest():
                try:
                    new_docs = run_ingestion(save_path, user_id)
                    hybrid_retriever.rebuild(new_docs=new_docs)
                    invalidate_book_list_cache(user_id)
                    invalidate_rag_cache()
                except Exception as exc:
                    logger.error(f"Background ingest error for '{book_name}': {exc}")

            threading.Thread(target=_bg_ingest, name=f"ingest-{book_name}", daemon=True).start()
            return JSONResponse(
                status_code=202,
                content={
                    "status":       "ingesting",
                    "book":         book_name,
                    "progress_url": f"/ingest/progress/{book_name}",
                    "message":      f"⏳ جارٍ استيعاب '{file.filename}' — تابع التقدم أدناه.",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ingest error: {e}")
            raise HTTPException(status_code=500, detail=_MSG.INGEST_FAILED(e))

    @router.get("/ingest/progress/{book_name}")
    async def ingest_progress(book_name: str, current=Depends(get_current_user)):
        with _ingest_progress_lock:
            ram = _ingest_progress.get(book_name)
        if ram:
            return ram
        from backend.utils.ingestion_ledger import get_all as _ledger_all
        # Ledger key is now "user_id/book_name"
        entry = _ledger_all().get(f"{current.user_id}/{book_name}")
        if entry:
            return {
                "status":      entry.get("status", "unknown"),
                "pages_done":  entry.get("total_pages", 0),
                "total_pages": entry.get("total_pages", 0),
                "chunks":      entry.get("chunks_added", 0),
            }
        return {"status": "unknown"}

    @router.get("/ingest/status")
    async def ingest_bulk_status(current=Depends(get_current_user)):
        from backend.utils.ingestion_ledger import get_all as _ledger_all
        user_id  = current.user_id
        prefix   = f"{user_id}/"
        all_entries = _ledger_all()
        # Filter to this user's entries only
        ledger   = {k[len(prefix):]: v for k, v in all_entries.items() if k.startswith(prefix)}

        complete = [n for n, v in ledger.items() if v.get("status") == "complete"]
        failed   = [n for n, v in ledger.items() if v.get("status") == "failed"]
        in_prog  = [n for n, v in ledger.items() if v.get("status") == "in_progress"]

        user_books_dir = os.path.join(BOOKS_DIR, user_id)
        total_on_disk  = 0
        if os.path.isdir(user_books_dir):
            total_on_disk = sum(1 for f in os.listdir(user_books_dir) if f.lower().endswith(".pdf"))

        total_chunks = sum(v.get("chunks_added", 0) for v in ledger.values())
        total_pages  = sum(v.get("total_pages",  0) for v in ledger.values())

        with _ingest_progress_lock:
            active = {k: v for k, v in _ingest_progress.items() if v.get("status") == "ingesting"}

        return {
            "total_books":    total_on_disk,
            "complete":       len(complete),
            "in_progress":    len(in_prog),
            "failed":         len(failed),
            "remaining":      max(0, total_on_disk - len(complete) - len(failed)),
            "total_chunks":   total_chunks,
            "total_pages":    total_pages,
            "active":         active,
            "failed_books":   failed,
            "complete_books": sorted(complete),
        }

    return router
