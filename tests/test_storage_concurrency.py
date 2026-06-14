"""Concurrency safety for the shared aiosqlite connection.

These exercise the _op_lock serialization: many coroutines hitting the one
shared connection at once must not corrupt transactions or lose writes.
"""
import asyncio

from backend.memory import storage


def _run(coro):
    async def wrapper():
        # Fresh connection + lock per loop: asyncio.run() makes a new event
        # loop each call, and the module-level lock binds to the first loop
        # that uses it (in production there is only ever one loop).
        storage._conn = None
        storage._op_lock = asyncio.Lock()
        try:
            return await coro
        finally:
            if storage._conn is not None:
                await storage._conn.close()
                storage._conn = None
    return asyncio.run(wrapper())


def test_concurrent_save_summary_and_messages_stay_consistent():
    """save_summary's BEGIN/commit must not be corrupted by concurrent
    save_message commits on the same connection."""
    async def flow():
        await storage.ensure_chat_schema()
        sid = await storage.create_session("user-X")
        # Fire summaries and messages concurrently at the one connection
        await asyncio.gather(
            *[storage.save_summary(sid, f"summary-{i}") for i in range(10)],
            *[storage.save_message(sid, "user", f"m-{i}") for i in range(10)],
        )
        # Exactly one summary row must survive (upsert semantics held)
        summary = await storage.get_latest_summary(sid)
        assert summary.startswith("summary-"), summary
        # All 10 user messages must be persisted (no lost writes)
        count = await storage.get_message_count(sid)
        assert count == 10, f"expected 10 user messages, got {count}"
    _run(flow())


def test_concurrent_reads_and_writes_no_cursor_corruption():
    """Interleaved execute/fetch pairs on the shared connection must each
    return their own result set."""
    async def flow():
        await storage.ensure_chat_schema()
        sids = [await storage.create_session(f"user-{i}") for i in range(5)]
        for i, sid in enumerate(sids):
            for j in range(i + 1):           # user-i gets i+1 messages
                await storage.save_message(sid, "user", f"{i}-{j}")
        # Read all contexts concurrently; each must match its own message count
        results = await asyncio.gather(
            *[storage.get_session_context(sid, limit=50) for sid in sids]
        )
        assert [len(r) for r in results] == [1, 2, 3, 4, 5]
    _run(flow())
