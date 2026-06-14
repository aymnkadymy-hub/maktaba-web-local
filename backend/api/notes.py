"""
Notes / Highlights system — student selects bot text and saves it as a note.
Registered in server_backend.py with prefix /notes.

Storage: SQLite via aiosqlite (works without PostgreSQL).
DB file: notes.db in project root (or NOTES_DB_PATH env var).
"""
import os
import uuid
import logging
import aiosqlite
from contextlib import asynccontextmanager
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("notes")
router = APIRouter(prefix="/notes", tags=["Notes"])

_DB_PATH = os.getenv(
    "NOTES_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "notes.db"),
)
_DB_PATH = os.path.normpath(_DB_PATH)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS highlights (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL DEFAULT '',
    book_title  TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_hl_user ON highlights(user_id);
CREATE INDEX IF NOT EXISTS idx_hl_book ON highlights(book_title);
"""


@asynccontextmanager
async def _db():
    async with aiosqlite.connect(_DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def ensure_highlights_table(get_db_ctx=None):
    """Idempotent schema init — get_db_ctx param kept for API compatibility."""
    try:
        async with _db() as conn:
            await conn.executescript(_CREATE_SQL)
            await conn.commit()
        logger.info(f"highlights table ready (SQLite: {_DB_PATH})")
    except Exception as e:
        logger.warning(f"highlights table init failed: {e}")


# ── Pydantic models ───────────────────────────────────────────────────────────

class NoteCreate(BaseModel):
    content:    str
    note:       str = ""
    session_id: str = ""
    book_title: str = ""


class NoteOut(BaseModel):
    id:         str
    content:    str
    note:       str
    session_id: str
    book_title: str
    created_at: str


class NoteUpdate(BaseModel):
    note: str


# ── Router factory ────────────────────────────────────────────────────────────

def make_notes_router(get_db_ctx=None, get_current_user=None):

    @router.post("/", response_model=NoteOut, status_code=201)
    async def create_note(body: NoteCreate,
                          user=Depends(get_current_user)):
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or ""
        note_id = str(uuid.uuid4())
        async with _db() as conn:
            await conn.execute(
                """INSERT INTO highlights
                   (id, user_id, session_id, book_title, content, note)
                   VALUES (?,?,?,?,?,?)""",
                (note_id, user_id, body.session_id, body.book_title,
                 body.content, body.note),
            )
            await conn.commit()
            row = await (await conn.execute(
                "SELECT id, content, note, session_id, book_title, created_at "
                "FROM highlights WHERE id=?", (note_id,)
            )).fetchone()
        return NoteOut(**dict(row))

    @router.get("/", response_model=list[NoteOut])
    async def list_notes(user=Depends(get_current_user),
                         book_title: str = ""):
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or ""
        async with _db() as conn:
            if book_title:
                rows = await (await conn.execute(
                    "SELECT id, content, note, session_id, book_title, created_at "
                    "FROM highlights WHERE user_id=? AND book_title=? "
                    "ORDER BY created_at DESC",
                    (user_id, book_title),
                )).fetchall()
            else:
                rows = await (await conn.execute(
                    "SELECT id, content, note, session_id, book_title, created_at "
                    "FROM highlights WHERE user_id=? ORDER BY created_at DESC",
                    (user_id,),
                )).fetchall()
        return [NoteOut(**dict(r)) for r in rows]

    @router.patch("/{note_id}", response_model=NoteOut)
    async def update_note(note_id: str, body: NoteUpdate,
                          user=Depends(get_current_user)):
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or ""
        async with _db() as conn:
            await conn.execute(
                "UPDATE highlights SET note=? WHERE id=? AND user_id=?",
                (body.note, note_id, user_id),
            )
            await conn.commit()
            row = await (await conn.execute(
                "SELECT id, content, note, session_id, book_title, created_at "
                "FROM highlights WHERE id=? AND user_id=?",
                (note_id, user_id),
            )).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="الملاحظة غير موجودة")
        return NoteOut(**dict(row))

    @router.delete("/{note_id}", status_code=204)
    async def delete_note(note_id: str, user=Depends(get_current_user)):
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or ""
        async with _db() as conn:
            cur = await conn.execute(
                "DELETE FROM highlights WHERE id=? AND user_id=?",
                (note_id, user_id),
            )
            await conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="الملاحظة غير موجودة")

    return router
