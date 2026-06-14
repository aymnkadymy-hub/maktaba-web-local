"""Rate limiting, prune helpers, dedup, and the Ollama circuit breaker."""
import time

from backend.core import guards as g


def test_chat_rate_limit_blocks_over_limit():
    g._chat_attempts.clear()
    assert all(g.check_chat_rate("u1") for _ in range(g._CHAT_MAX_PER_MINUTE))
    assert not g.check_chat_rate("u1")
    assert g.check_chat_rate("u2"), "other users unaffected"


def test_prune_chat_attempts_drops_stale_keeps_fresh():
    g._chat_attempts.clear()
    g._chat_attempts["stale"] = [time.time() - 999]
    g._chat_attempts["fresh"] = [time.time()]
    g.prune_chat_attempts()
    assert "stale" not in g._chat_attempts
    assert "fresh" in g._chat_attempts


def test_prune_enforces_hard_cap():
    g._chat_attempts.clear()
    now = time.time()
    for i in range(g._MAX_CHAT_USERS + 50):
        g._chat_attempts[f"u{i}"] = [now] * (1 if i < 50 else 2)
    g.prune_chat_attempts()
    assert len(g._chat_attempts) <= g._MAX_CHAT_USERS
    assert "u0" not in g._chat_attempts, "fewest-attempts evicted first"
    g._chat_attempts.clear()


def test_prune_register_attempts():
    g._register_attempts.clear()
    g._register_attempts["old_ip"] = [time.time() - 4000]
    g._register_attempts["new_ip"] = [time.time()]
    g.prune_register_attempts()
    assert "old_ip" not in g._register_attempts
    assert "new_ip" in g._register_attempts


def test_duplicate_request_detection():
    g._recent_msg.clear()
    assert not g.is_duplicate_request("s1", "hello")
    assert g.is_duplicate_request("s1", "hello")
    assert not g.is_duplicate_request("s1", "different")
    assert not g.is_duplicate_request("s2", "hello"), "scoped per session"


def test_circuit_breaker_opens_and_closes():
    g._ollama_failures = 0
    g._ollama_open_until = 0.0
    for _ in range(g._OLLAMA_THRESHOLD):
        g.ollama_record_failure()
    assert g.ollama_circuit_open()
    g.ollama_record_success()
    assert not g.ollama_circuit_open()
