"""Chat history persistence (aiosqlite, temp DB via CHAT_DB_PATH)."""
import asyncio

from backend.memory import storage


def _run(coro):
    """Run a storage flow on a fresh event loop with a fresh connection.

    The module-level shared connection must not outlive the loop it was
    created on (its aiosqlite worker thread is non-daemon and would hang
    the test process at exit), so open and close it inside the same loop.
    """
    async def wrapper():
        storage._conn = None
        storage._op_lock = asyncio.Lock()   # bind lock to this loop (see below)
        try:
            return await coro
        finally:
            if storage._conn is not None:
                await storage._conn.close()
                storage._conn = None
    return asyncio.run(wrapper())


def test_schema_and_session_lifecycle():
    async def flow():
        await storage.ensure_chat_schema()
        sid = await storage.create_session("user-A")
        assert sid
        # ensure_session: new id → True, existing id → False
        assert await storage.ensure_session("manual-id", "user-A") is True
        assert await storage.ensure_session("manual-id", "user-A") is False
        return sid
    _run(flow())


def test_messages_roundtrip_order_and_limit():
    async def flow():
        await storage.ensure_chat_schema()
        sid = await storage.create_session("user-B")
        for i in range(7):
            await storage.save_message(sid, "user" if i % 2 == 0 else "assistant",
                                       f"msg-{i}")
        last5 = await storage.get_session_context(sid, limit=5)
        assert [m["content"] for m in last5] == [f"msg-{i}" for i in range(2, 7)], \
            "must return last N messages oldest-first"
        return sid
    _run(flow())


def test_session_owner_and_user_isolation():
    async def flow():
        await storage.ensure_chat_schema()
        sid_a = await storage.create_session("owner-A")
        sid_b = await storage.create_session("owner-B")
        assert await storage.get_session_owner(sid_a) == "owner-A"
        assert await storage.get_session_owner("no-such-session") is None
        sessions_a = await storage.list_user_sessions("owner-A")
        ids_a = {s["id"] for s in sessions_a}
        assert sid_a in ids_a and sid_b not in ids_a, \
            "list_user_sessions must be scoped to the user"
    _run(flow())
