"""Rate limiting, dedup guard, and Ollama circuit breaker — all pure stdlib, no circular imports."""
import os
import time as _time_mod
import threading
import collections
import logging

logger = logging.getLogger("backend")

# ── Ollama circuit breaker ────────────────────────────────────────────────────
_ollama_failures   = 0
_ollama_open_until = 0.0
_OLLAMA_THRESHOLD  = 5
_OLLAMA_RESET_SEC  = 30
_ollama_cb_lock    = threading.Lock()


def ollama_circuit_open() -> bool:
    global _ollama_failures, _ollama_open_until
    with _ollama_cb_lock:
        now = _time_mod.time()
        if _ollama_open_until and now >= _ollama_open_until:
            # Reset counter when the open window expires — allows fresh probe
            _ollama_failures   = 0
            _ollama_open_until = 0.0
            logger.info("[CIRCUIT] Ollama circuit half-open — allowing probe")
        return _ollama_open_until > now


def ollama_record_failure():
    global _ollama_failures, _ollama_open_until
    with _ollama_cb_lock:
        _ollama_failures += 1
        if _ollama_failures >= _OLLAMA_THRESHOLD:
            _ollama_open_until = _time_mod.time() + _OLLAMA_RESET_SEC
            logger.warning(
                f"[CIRCUIT] Ollama circuit OPEN for {_OLLAMA_RESET_SEC}s "
                f"after {_ollama_failures} consecutive failures"
            )


def ollama_record_success():
    global _ollama_failures, _ollama_open_until
    with _ollama_cb_lock:
        if _ollama_failures:
            logger.info("[CIRCUIT] Ollama circuit CLOSED — response received")
        _ollama_failures   = 0
        _ollama_open_until = 0.0


# ── Per-user chat rate limit ──────────────────────────────────────────────────
_chat_attempts: dict[str, list[float]] = collections.defaultdict(list)
_CHAT_MAX_PER_MINUTE = int(os.getenv("CHAT_MAX_PER_MINUTE", "30"))
_MAX_CHAT_USERS = 50_000   # hard cap


def check_chat_rate(user_id: str) -> bool:
    """Return False if user exceeds CHAT_MAX_PER_MINUTE requests in the last 60 s."""
    now    = _time_mod.time()
    window = [t for t in _chat_attempts[user_id] if now - t < 60]
    if len(window) >= _CHAT_MAX_PER_MINUTE:
        return False
    window.append(now)
    _chat_attempts[user_id] = window
    return True


def _prune_window(store: dict, window_sec: float, max_entries: int, label: str):
    """Drop entries whose timestamps all predate the window, then enforce the
    hard cap by evicting entries with the fewest attempts first."""
    cutoff = _time_mod.time() - window_sec
    stale  = [k for k, ts_list in store.items()
              if not any(t > cutoff for t in ts_list)]
    for k in stale:
        del store[k]
    if len(store) > max_entries:
        excess = sorted(store.items(), key=lambda x: len(x[1]))
        for k, _ in excess[:len(store) - max_entries]:
            del store[k]
    if stale:
        logger.debug(f"Pruned {len(stale)} {label} entries")


def prune_chat_attempts():
    _prune_window(_chat_attempts, 120, _MAX_CHAT_USERS, "chat rate-limit")


# ── Per-IP register rate limit ────────────────────────────────────────────────
_register_attempts: dict[str, list[float]] = collections.defaultdict(list)
_REGISTER_MAX_PER_HOUR = int(os.getenv("REGISTER_MAX_PER_HOUR", "5"))
_MAX_REGISTER_IPS = 10_000   # hard cap — prevents memory exhaustion


def prune_register_attempts():
    _prune_window(_register_attempts, 3600, _MAX_REGISTER_IPS, "rate-limit IP")


# ── Duplicate-request dedup ───────────────────────────────────────────────────
_recent_msg:      dict[str, tuple[str, float]] = {}
_recent_msg_lock  = threading.Lock()
_DEDUP_WINDOW_SEC = 3
_DEDUP_MAX        = 100_000   # hard cap — prevents memory exhaustion


def is_duplicate_request(session_id: str, msg: str) -> bool:
    import hashlib as _hl
    key  = _hl.sha256(msg.encode()).hexdigest()[:16]
    now  = _time_mod.time()
    with _recent_msg_lock:
        prev = _recent_msg.get(session_id)
        if prev and prev[0] == key and now - prev[1] < _DEDUP_WINDOW_SEC:
            return True
        _recent_msg[session_id] = (key, now)
        if len(_recent_msg) > _DEDUP_MAX:
            # Evict oldest half by timestamp
            oldest = sorted(_recent_msg.items(), key=lambda x: x[1][1])
            for sid, _ in oldest[:_DEDUP_MAX // 2]:
                del _recent_msg[sid]
    return False


def prune_dedup():
    dedup_cutoff = _time_mod.time() - 10
    with _recent_msg_lock:
        stale = [sid for sid, (_, ts) in _recent_msg.items() if ts < dedup_cutoff]
        for sid in stale:
            del _recent_msg[sid]
