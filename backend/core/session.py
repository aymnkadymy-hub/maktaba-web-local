"""RAM session history — get/save history, session ownership, pruning."""
import asyncio
import os
import re
import time as _time_mod
import logging

logger = logging.getLogger("backend")

# ── In-memory session store ───────────────────────────────────────────────────
_ram_history:    dict[str, list[dict]] = {}
_ram_history_ts: dict[str, float]      = {}
_session_book:   dict[str, str]        = {}   # session_id → last book answered from
_RAM_HISTORY_LIMIT  = 20
_HISTORY_CTX_LIMIT  = 8
_MAX_SESSIONS_RAM   = int(os.getenv("MAX_SESSIONS_RAM", "500"))
_RAM_SESSION_TTL    = 7200   # 2 hours

_session_lock = asyncio.Lock()


def set_session_book(session_id: str, book: str) -> None:
    """Remember the book a session is currently discussing — used to keep
    short follow-up retrieval anchored to the same book (session affinity)."""
    if book:
        _session_book[session_id] = book


def get_session_book(session_id: str) -> str:
    return _session_book.get(session_id, "")


def _enforce_session_cap():
    if len(_ram_history) <= _MAX_SESSIONS_RAM:
        return
    overage = len(_ram_history) - _MAX_SESSIONS_RAM
    oldest  = sorted(_ram_history_ts.items(), key=lambda x: x[1])[:overage]
    for sid, _ in oldest:
        _ram_history.pop(sid, None)
        _ram_history_ts.pop(sid, None)
        _session_book.pop(sid, None)
    logger.debug(f"Session cap: evicted {overage} oldest sessions")


def prune_ram_history():
    cutoff = _time_mod.monotonic() - _RAM_SESSION_TTL
    stale  = [sid for sid, ts in _ram_history_ts.items() if ts < cutoff]
    for sid in stale:
        _ram_history.pop(sid, None)
        _ram_history_ts.pop(sid, None)
        _session_book.pop(sid, None)
    if stale:
        logger.debug(f"Pruned {len(stale)} stale RAM sessions")


async def get_history(session_id: str) -> str:
    try:
        from backend.memory.summarizer import get_context_smart
        ctx = await get_context_smart(session_id)
        if ctx:
            return ctx
    except Exception:
        pass
    async with _session_lock:
        _ram_history_ts[session_id] = _time_mod.monotonic()
        msgs = _ram_history.get(session_id, [])[-_HISTORY_CTX_LIMIT:]
    return "\n".join(
        f"{'المستخدم' if m['role'] == 'user' else 'البوت'}: {m['content']}"
        for m in msgs
    ) if msgs else ""


_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')


async def get_recent_user_text(session_id: str, n: int = 2, bot_chars: int = 240) -> str:
    """Return recent conversation topic text (RAM-only, fast) for retrieval anchoring.

    Combines the last `n` user messages with the last bot reply (code stripped,
    truncated). Short follow-ups like "أعطني مثالاً" have no content words; even a
    chain of them ("مثال" → "وضّح أكثر") stays on-topic because the last bot reply
    always carries the subject. Without this, retrieval drifts to an unrelated book.
    """
    async with _session_lock:
        msgs = list(_ram_history.get(session_id, []))
    users = [m["content"] for m in msgs if m["role"] == "user"][-n:]
    last_bot = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    last_bot = _CODE_BLOCK_RE.sub(" ", last_bot)[:bot_chars]
    return (" ".join(users) + " " + last_bot).strip()


async def save_exchange(session_id: str, user_msg: str, bot_msg: str, user_id: str = "unknown"):
    async with _session_lock:
        _ram_history_ts[session_id] = _time_mod.monotonic()
        session = _ram_history.setdefault(session_id, [])
        session.extend([
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": bot_msg},
        ])
        if len(session) > _RAM_HISTORY_LIMIT:
            _ram_history[session_id] = session[-_RAM_HISTORY_LIMIT:]
        _enforce_session_cap()

    try:
        from backend.memory.storage import ensure_session, save_message, update_session_topic
        is_new = await ensure_session(session_id, user_id)
        if is_new:
            topic = user_msg[:45] + ("…" if len(user_msg) > 45 else "")
            await update_session_topic(session_id, topic)
        await save_message(session_id, "user",      user_msg)
        await save_message(session_id, "assistant", bot_msg)
    except Exception as e:
        logger.debug(f"SQLite save skipped: {e}")

    try:
        from backend.memory.fact_extractor import extract_and_store_async
        extract_and_store_async(user_msg, bot_msg, user_id, session_id)
    except Exception as e:
        logger.debug(f"Fact extraction skipped: {e}")


async def get_session_owner(session_id: str):
    """Return user_id for session, or None if not found."""
    try:
        from backend.memory.storage import get_session_owner as _db_owner
        return await _db_owner(session_id)
    except Exception as e:
        logger.warning(f"Session owner lookup failed: {e}")
        return None
