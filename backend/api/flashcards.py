"""
Flashcards + spaced repetition (SM-2) — evidence-based active-recall study.

Students generate flashcards from a book (broad content sample → LLM), then
review them on an SM-2 schedule (see backend/srs.py). Everything is offline:
generation reuses the local LLM; scheduling is deterministic math; storage is
SQLite. Registered in server_backend.py with prefix /flashcards.

Endpoints:
  POST   /flashcards/generate        — make cards from a book
  GET    /flashcards/due             — cards due for review now
  POST   /flashcards/{id}/review     — grade a card (0-5) → reschedule (SM-2)
  GET    /flashcards                 — list the user's cards (optional ?book_title=)
  GET    /flashcards/stats           — counts: total / due / per book
  DELETE /flashcards/{id}            — delete a card
"""
import os
import re
import uuid
import asyncio
import inspect
import logging
import aiosqlite
from contextlib import asynccontextmanager
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.srs import CardState, review as srs_review
from backend.api.quiz import _extract_json_array  # reuse robust JSON extraction

logger = logging.getLogger("flashcards")
router = APIRouter(prefix="/flashcards", tags=["Flashcards"])

_DB_PATH = os.path.normpath(os.getenv(
    "FLASHCARDS_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "flashcards.db"),
))

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS cards (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    book_title  TEXT NOT NULL DEFAULT '',
    front       TEXT NOT NULL,
    back        TEXT NOT NULL,
    repetitions INTEGER NOT NULL DEFAULT 0,
    interval    INTEGER NOT NULL DEFAULT 0,
    ease        REAL    NOT NULL DEFAULT 2.5,
    due_date    TEXT    NOT NULL DEFAULT (date('now')),
    reviews     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cards_user ON cards(user_id, due_date);
CREATE INDEX IF NOT EXISTS idx_cards_book ON cards(user_id, book_title);
"""

_TITLE_INJECTION_RE = re.compile(r'[<>\[\]{}\x00-\x1f]|--|\*\*|<<|>>')
_TOKENS_PER_CARD = 90
_PER_CALL_SEC    = int(os.getenv("FLASHCARDS_TIMEOUT", "150"))


@asynccontextmanager
async def _db():
    async with aiosqlite.connect(_DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def ensure_flashcards_table():
    try:
        async with _db() as conn:
            await conn.executescript(_CREATE_SQL)
            await conn.commit()
        logger.info(f"flashcards table ready (SQLite: {_DB_PATH})")
    except Exception as e:
        logger.warning(f"flashcards table init failed: {e}")


# ── Models ────────────────────────────────────────────────────────────────────
class GenerateReq(BaseModel):
    book_title: str = Field(..., min_length=1, max_length=200)
    n_cards:    int = Field(default=8, ge=1, le=50)

    @field_validator("book_title")
    @classmethod
    def _safe_title(cls, v: str) -> str:
        v = v.strip()
        if _TITLE_INJECTION_RE.search(v):
            raise ValueError("عنوان الكتاب يحتوي على رموز غير مسموحة")
        return v


class ReviewReq(BaseModel):
    grade: int = Field(..., ge=0, le=5)


# ── Card generation prompt ────────────────────────────────────────────────────
def _build_prompt(n: int, context: str) -> str:
    return (
        f'أنشئ بالضبط {n} بطاقة تعليمية (flashcards) من النص التالي لمساعدة طالب '
        f'على الحفظ بالاستدعاء النشط.\n\n'
        f'النص:\n{context}\n\n'
        'أخرج JSON فقط بهذا الشكل بدون أي نص آخر:\n'
        '[{"front":"سؤال أو مصطلح","back":"الإجابة أو التعريف المختصر"}]\n\n'
        'قواعد: front = سؤال قصير أو مصطلح محدد من النص؛ back = إجابة دقيقة '
        f'وموجزة. كل بطاقة عن معلومة محددة (تعريف/حقيقة/سبب/خطوة). أنشئ {n} بطاقة بالضبط.'
    )


def _valid_card(c) -> bool:
    return (isinstance(c, dict) and isinstance(c.get("front"), str)
            and isinstance(c.get("back"), str)
            and len(c["front"].strip()) >= 3 and len(c["back"].strip()) >= 1)


# ── Router factory ────────────────────────────────────────────────────────────
def make_flashcards_router(retriever, llm_ask, get_current_user):

    def _uid(user):
        return getattr(user, "user_id", None) or getattr(user, "id", None) or ""

    @router.post("/generate")
    async def generate(body: GenerateReq, user=Depends(get_current_user)):
        uid = _uid(user)
        loop = asyncio.get_running_loop()
        # Broad, even sample of the whole book (same source as quiz)
        docs = []
        if hasattr(retriever, "book_chunks"):
            docs = await loop.run_in_executor(
                None, lambda: retriever.book_chunks(body.book_title, uid, max(12, body.n_cards * 3)))
        if not docs:
            raise HTTPException(404, detail=f"لم يُعثر على محتوى للكتاب: {body.book_title}")

        context = "\n\n---\n\n".join(d.page_content[:500] for d in docs)[:2500]
        prompt = _build_prompt(body.n_cards, context)

        sig = inspect.signature(llm_ask).parameters
        kw = {}
        if "num_predict" in sig:
            kw["num_predict"] = body.n_cards * _TOKENS_PER_CARD
        if "num_ctx" in sig:
            kw["num_ctx"] = int(os.getenv("NUM_CTX", "1536"))
        if "timeout" in sig:
            kw["timeout"] = _PER_CALL_SEC
        # Retry once: LLM JSON output is occasionally malformed/truncated
        cards = []
        for attempt in range(2):
            try:
                raw = await llm_ask(prompt, **kw)
            except Exception as e:
                logger.warning(f"flashcards LLM error (attempt {attempt+1}): {e}")
                if attempt == 0:
                    continue
                raise HTTPException(503, detail="تعذّر توليد البطاقات — حاول مرة أخرى")
            cards = [c for c in _extract_json_array(raw) if _valid_card(c)][:body.n_cards]
            if cards:
                break
            logger.info(f"flashcards: no valid cards (attempt {attempt+1})")
        if not cards:
            raise HTTPException(502, detail="لم تُولَّد بطاقات صالحة — حاول مجدداً")

        created = []
        async with _db() as conn:
            for c in cards:
                cid = str(uuid.uuid4())
                await conn.execute(
                    "INSERT INTO cards (id, user_id, book_title, front, back) VALUES (?,?,?,?,?)",
                    (cid, uid, body.book_title, c["front"].strip(), c["back"].strip()),
                )
                created.append({"id": cid, "front": c["front"].strip(), "back": c["back"].strip()})
            await conn.commit()
        logger.info(f"flashcards: generated {len(created)} for '{body.book_title}' user={uid[:8]}")
        return {"created": len(created), "cards": created}

    @router.get("/due")
    async def due(user=Depends(get_current_user), limit: int = 20):
        uid = _uid(user)
        async with _db() as conn:
            rows = await (await conn.execute(
                "SELECT id, book_title, front, back, repetitions, interval, ease, reviews "
                "FROM cards WHERE user_id=? AND due_date <= date('now') "
                "ORDER BY due_date LIMIT ?", (uid, min(limit, 100))
            )).fetchall()
        return {"due": len(rows), "cards": [dict(r) for r in rows]}

    @router.post("/{card_id}/review")
    async def review_card(card_id: str, body: ReviewReq, user=Depends(get_current_user)):
        uid = _uid(user)
        async with _db() as conn:
            row = await (await conn.execute(
                "SELECT repetitions, interval, ease FROM cards WHERE id=? AND user_id=?",
                (card_id, uid)
            )).fetchone()
            if not row:
                raise HTTPException(404, detail="البطاقة غير موجودة")
            nxt = srs_review(CardState(row["repetitions"], row["interval"], row["ease"]), body.grade)
            await conn.execute(
                "UPDATE cards SET repetitions=?, interval=?, ease=?, reviews=reviews+1, "
                "due_date=date('now', ?) WHERE id=? AND user_id=?",
                (nxt.repetitions, nxt.interval, nxt.ease, f"+{nxt.interval} days", card_id, uid),
            )
            await conn.commit()
        return {"id": card_id, "interval_days": nxt.interval,
                "repetitions": nxt.repetitions, "ease": nxt.ease}

    @router.get("")
    async def list_cards(user=Depends(get_current_user), book_title: str = ""):
        uid = _uid(user)
        q = ("SELECT id, book_title, front, back, repetitions, interval, due_date, reviews "
             "FROM cards WHERE user_id=?")
        args = [uid]
        if book_title:
            q += " AND book_title=?"
            args.append(book_title)
        q += " ORDER BY created_at DESC"
        async with _db() as conn:
            rows = await (await conn.execute(q, tuple(args))).fetchall()
        return {"total": len(rows), "cards": [dict(r) for r in rows]}

    @router.get("/stats")
    async def stats(user=Depends(get_current_user)):
        uid = _uid(user)
        async with _db() as conn:
            total = (await (await conn.execute(
                "SELECT COUNT(*) FROM cards WHERE user_id=?", (uid,))).fetchone())[0]
            due = (await (await conn.execute(
                "SELECT COUNT(*) FROM cards WHERE user_id=? AND due_date <= date('now')",
                (uid,))).fetchone())[0]
            per_book = await (await conn.execute(
                "SELECT book_title, COUNT(*) c FROM cards WHERE user_id=? GROUP BY book_title",
                (uid,))).fetchall()
        return {"total": total, "due": due,
                "per_book": {r["book_title"]: r["c"] for r in per_book}}

    @router.delete("/{card_id}")
    async def delete_card(card_id: str, user=Depends(get_current_user)):
        uid = _uid(user)
        async with _db() as conn:
            cur = await conn.execute("DELETE FROM cards WHERE id=? AND user_id=?", (card_id, uid))
            await conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, detail="البطاقة غير موجودة")
        return {"deleted": card_id}

    return router
