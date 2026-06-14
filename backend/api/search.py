"""
Conversation search — smart Arabic + English fuzzy search across chat history.
Searches both user questions and bot answers. No score cutoff — always shows
all results sorted by relevance. Falls back to recent chats when no match found.
"""
import os
import re
import logging
import difflib
import aiosqlite
from contextlib import asynccontextmanager
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

logger = logging.getLogger("search")
router = APIRouter(prefix="/search", tags=["Search"])

_DB_PATH = os.path.normpath(os.getenv("CHAT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "chat.db")))


# ── Text normalization ────────────────────────────────────────────────────────
_RE_ALEF  = re.compile(r'[أإآٱ]')
_RE_YA    = re.compile(r'[يى]')
_RE_TA_M  = re.compile(r'ة')
_RE_DIACR = re.compile(r'[ً-ٰٟـ]')
_RE_SPLIT = re.compile(r'[\s،,؟?!.\-/\(\)\[\]"\']+')


def _normalize(text: str) -> str:
    """Arabic: unify letter variants + strip diacritics. English: lowercase."""
    text = _RE_ALEF.sub('ا', text)
    text = _RE_YA.sub('ي', text)
    text = _RE_TA_M.sub('ه', text)
    text = _RE_DIACR.sub('', text)
    return text.lower().strip()


def _words(text: str) -> list[str]:
    return [w for w in _RE_SPLIT.split(text) if len(w) >= 2]


def _score(query_norm: str, q_words: list[str], text: str) -> float:
    """
    Score 0.0–1.0. Four signals:
    1. Exact normalized substring  → 1.0 immediately
    2. Word coverage ratio
    3. Fuzzy sequence similarity (handles typos)
    4. Any single word present bonus
    Works for both Arabic and English.
    """
    if not text or not query_norm:
        return 0.0

    text_norm = _normalize(text)

    if query_norm in text_norm:
        return 1.0

    word_hits  = sum(1 for w in q_words if w in text_norm)
    word_score = word_hits / max(len(q_words), 1)

    window = text_norm[:max(len(query_norm) * 8, 400)]
    fuzzy  = difflib.SequenceMatcher(None, query_norm, window, autojunk=False).ratio()

    any_hit = any(w in text_norm for w in q_words) if q_words else False

    return min(word_score * 0.45 + fuzzy * 0.40 + (0.15 if any_hit else 0.0), 1.0)


# ── DB ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def _db():
    async with aiosqlite.connect(_DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


# ── Models ────────────────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    session_id:   str
    topic:        str
    user_message: str
    bot_message:  str
    book_title:   str
    created_at:   str
    score:        float = 0.0


class SearchResponse(BaseModel):
    query:       str
    total:       int
    results:     list[SearchResult]
    has_history: bool = True
    any_match:   bool = True


# ── Router ────────────────────────────────────────────────────────────────────
def make_search_router(get_db_ctx, get_current_user):

    @router.get("/conversations", response_model=SearchResponse)
    async def search_conversations(
        q:     str = Query(default="", description="كلمة البحث"),
        limit: int = Query(default=50, ge=1, le=200),
        user:  object = Depends(get_current_user),
    ):
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
        try:
            q_text  = q.strip()
            q_norm  = _normalize(q_text)
            q_words = _words(q_norm)

            async with _db() as conn:
                # Does this user have any history?
                cnt_cur     = await conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,)
                )
                cnt_row     = await cnt_cur.fetchone()
                has_history = bool(cnt_row and cnt_row[0] > 0)

                if not has_history:
                    return SearchResponse(
                        query=q_text, total=0, results=[],
                        has_history=False, any_match=False,
                    )

                # Fetch all exchanges (most recent first)
                cur = await conn.execute("""
                    SELECT
                        a.session_id,
                        COALESCE(s.topic, 'محادثة')                           AS topic,
                        COALESCE(u.content, '')                                AS user_message,
                        a.content                                              AS bot_message,
                        COALESCE(json_extract(a.metadata, '$.book_title'), '') AS book_title,
                        a.created_at                                           AS created_at
                    FROM messages a
                    JOIN sessions s ON s.id = a.session_id
                    LEFT JOIN messages u ON u.id = (
                        SELECT id FROM messages
                        WHERE session_id = a.session_id
                          AND role = 'user'
                          AND id < a.id
                        ORDER BY id DESC
                        LIMIT 1
                    )
                    WHERE s.user_id = ?
                      AND a.role = 'assistant'
                    ORDER BY a.id DESC
                    LIMIT 2000
                """, (user_id,))
                raw_rows = await cur.fetchall()

            # Convert to plain dicts
            exchanges = [dict(r) for r in raw_rows]

            # Empty query → show recent conversations as-is
            if not q_text:
                for ex in exchanges:
                    ex['score'] = 0.0
                top = exchanges[:limit]
                return SearchResponse(
                    query='', total=len(top),
                    results=[SearchResult(**r) for r in top],
                    has_history=True, any_match=False,
                )

            # Score every exchange
            for ex in exchanges:
                s_bot   = _score(q_norm, q_words, ex.get('bot_message', ''))
                s_user  = _score(q_norm, q_words, ex.get('user_message', ''))
                s_topic = _score(q_norm, q_words, ex.get('topic', ''))
                ex['score'] = round(max(s_bot, s_user, s_topic), 4)

            # Sort: best score first; ties preserve most-recent order (stable sort)
            exchanges.sort(key=lambda x: x['score'], reverse=True)

            any_match = any(ex['score'] >= 0.12 for ex in exchanges)

            # Always return all results: good matches first, then rest (no cutoff)
            top = exchanges[:limit]

            return SearchResponse(
                query=q_text, total=len(top),
                results=[SearchResult(**r) for r in top],
                has_history=True, any_match=any_match,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return SearchResponse(
                query=q.strip(), total=0, results=[],
                has_history=True, any_match=False,
            )

    return router
