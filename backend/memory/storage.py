# backend/memory/storage.py
"""Chat history persistence using SQLite (aiosqlite).
Uses a single persistent WAL connection shared across all async callers
instead of opening/closing a connection per call — removes per-call overhead.
"""
import os
import json
import uuid
import logging
import asyncio
import functools
import aiosqlite
from typing import List, Dict, Optional

logger = logging.getLogger("memory.storage")

_DB_PATH = os.path.normpath(os.getenv("CHAT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "chat.db")))

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    topic      TEXT NOT NULL DEFAULT 'محادثة جديدة',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    metadata   TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_user   ON sessions (user_id, created_at);
"""

# ── Shared persistent connection ──────────────────────────────────────────────
_conn: "aiosqlite.Connection | None" = None
_conn_lock = asyncio.Lock()
# Serializes every DB operation. The single shared connection has ONE SQLite
# transaction; without this, a concurrent save_message().commit() could commit
# save_summary()'s open BEGIN mid-transaction, and interleaved execute/fetch
# pairs could read another coroutine's cursor. aiosqlite already runs all SQL
# on one worker thread, so this lock only prevents await-interleaving — no
# throughput cost (SQLite serializes writes regardless).
_op_lock = asyncio.Lock()


def _serialized(fn):
    """Run an async DB function under _op_lock so its execute/fetch/commit
    sequence cannot interleave with another coroutine's."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        async with _op_lock:
            return await fn(*args, **kwargs)
    return wrapper


async def _get_conn() -> aiosqlite.Connection:
    """Return (or lazily open) the shared WAL connection."""
    global _conn
    async with _conn_lock:
        if _conn is None:
            _conn = await aiosqlite.connect(_DB_PATH)
            _conn.row_factory = aiosqlite.Row
            await _conn.execute("PRAGMA journal_mode=WAL")
            await _conn.execute("PRAGMA foreign_keys=ON")
            await _conn.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL = safe + fast
            await _conn.execute("PRAGMA cache_size=-8000")    # 8 MB page cache
    return _conn


@_serialized
async def ensure_chat_schema():
    conn = await _get_conn()
    await conn.executescript(_CREATE_SQL)
    await conn.commit()
    logger.info(f"Chat history table ready (SQLite: {_DB_PATH})")


@_serialized
async def ensure_session(session_id: str, user_id: str) -> bool:
    """Create session if it doesn't exist. Returns True if newly created."""
    conn = await _get_conn()
    cur  = await conn.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


@_serialized
async def create_session(user_id: str) -> str:
    session_id = str(uuid.uuid4())
    conn = await _get_conn()
    await conn.execute(
        "INSERT INTO sessions (id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
    await conn.commit()
    return session_id


@_serialized
async def save_message(session_id: str, role: str, content: str, metadata: Optional[dict] = None):
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    conn = await _get_conn()
    await conn.execute(
        "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
        (session_id, role, content, meta_json),
    )
    await conn.commit()


@_serialized
async def get_session_context(session_id: str, limit: int = 5) -> List[Dict]:
    """Return last `limit` messages for a session, oldest first."""
    conn = await _get_conn()
    cur  = await conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    )
    rows = await cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


@_serialized
async def list_user_sessions(user_id: str) -> List[Dict]:
    conn = await _get_conn()
    cur  = await conn.execute(
        "SELECT id, topic, created_at FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = await cur.fetchall()
    return [{"id": r["id"], "topic": r["topic"], "created_at": r["created_at"]} for r in rows]


@_serialized
async def update_session_topic(session_id: str, topic: str):
    conn = await _get_conn()
    await conn.execute(
        "UPDATE sessions SET topic = ?, updated_at = datetime('now') WHERE id = ?",
        (topic, session_id),
    )
    await conn.commit()


@_serialized
async def get_message_count(session_id: str) -> int:
    conn = await _get_conn()
    cur  = await conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role != 'summary'",
        (session_id,),
    )
    row = await cur.fetchone()
    return row[0] if row else 0


@_serialized
async def save_summary(session_id: str, summary_text: str):
    """Upsert rolling summary — atomically replaces any previous summary."""
    conn = await _get_conn()
    await conn.execute("BEGIN")
    try:
        await conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND role = 'summary'",
            (session_id,),
        )
        await conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'summary', ?)",
            (session_id, summary_text),
        )
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise


@_serialized
async def get_latest_summary(session_id: str) -> str:
    conn = await _get_conn()
    cur  = await conn.execute(
        "SELECT content FROM messages WHERE session_id = ? AND role = 'summary' ORDER BY id DESC LIMIT 1",
        (session_id,),
    )
    row = await cur.fetchone()
    return row["content"] if row else ""


@_serialized
async def get_session_owner(session_id: str):
    """Return user_id for session, or None if not found."""
    conn = await _get_conn()
    cur  = await conn.execute(
        "SELECT user_id FROM sessions WHERE id = ?",
        (session_id,),
    )
    row = await cur.fetchone()
    return row["user_id"] if row else None
