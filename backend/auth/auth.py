"""
User authentication — SQLite + bcrypt + random session tokens.

Design:
  - Passwords hashed with bcrypt (cost=12)
  - Tokens are cryptographically random (32 bytes URL-safe base64)
  - Tokens stored in SQLite with 30-day expiry (refreshed on each use)
  - No JWT keys to manage; tokens can be revoked instantly

DB location: PROJECT_ROOT/auth.db

Tables:
  users  (id, username, password_hash, created_at)
  tokens (token, user_id, username, created_at, expires_at)
"""
import os
import hashlib
import sqlite3
import secrets
import time
import logging
import threading
from typing import Optional, Tuple
from contextlib import contextmanager

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("auth")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# AUTH_DB: configurable for cloud deployments (e.g. mount a persistent volume path)
AUTH_DB      = os.getenv("AUTH_DB", os.path.join(PROJECT_ROOT, "auth.db"))

# TOKEN_TTL_DAYS: 30 days default. Mobile apps may prefer longer (e.g. 90 days).
TOKEN_TTL    = int(os.getenv("TOKEN_TTL_DAYS", "30")) * 24 * 3600
BCRYPT_COST  = 12

# Dummy hash used when username is not found — ensures bcrypt always runs
# so response time is the same whether the username exists or not (prevents
# username enumeration via timing).
_DUMMY_HASH = bcrypt.hashpw(b"__dummy__", bcrypt.gensalt(rounds=BCRYPT_COST)).decode()


def _hash_token(token: str) -> str:
    """Tokens are stored as SHA-256 at rest — a stolen auth.db no longer
    yields usable session tokens. The raw token lives only in the cookie."""
    return hashlib.sha256(token.encode()).hexdigest()

_bearer = HTTPBearer(auto_error=False)


# ── Database setup ─────────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                username     TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at   REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tokens (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                username   TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_tokens_user_id  ON tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_tokens_expires  ON tokens(expires_at);
        """)
    logger.info(f"Auth DB ready: {AUTH_DB}")
    _schedule_cleanup()


def _schedule_cleanup():
    """Background thread: purge expired tokens every 24 hours."""
    def _run():
        while True:
            time.sleep(24 * 3600)
            _cleanup_expired()
    threading.Thread(target=_run, daemon=True).start()


# ── Core auth functions ────────────────────────────────────────────────────────

def register(username: str, password: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Create a new user account.
    Returns (user_id, None) on success or (None, error_message) on failure.
    """
    username = username.strip()
    if not username or len(username) < 3:
        return None, "اسم المستخدم يجب أن يكون 3 أحرف على الأقل"
    if len(username) > 50:
        return None, "اسم المستخدم طويل جداً"
    if not password or len(password) < 6:
        return None, "كلمة المرور يجب أن تكون 6 أحرف على الأقل"

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    user_id = secrets.token_urlsafe(16)

    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO users (id, username, password_hash, created_at) VALUES (?,?,?,?)",
                (user_id, username, pw_hash, time.time())
            )
        logger.info(f"New user registered: '{username}' ({user_id[:8]}…)")
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "اسم المستخدم مستخدم بالفعل"
    except Exception as e:
        logger.error(f"Register error: {e}")
        return None, "خطأ في إنشاء الحساب"


def login(username: str, password: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Authenticate a user and issue a new token.
    Returns (token, user_id, username, None) on success
    or      (None, None, None, error_message) on failure.
    """
    with _db() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username=? COLLATE NOCASE",
            (username.strip(),)
        ).fetchone()

    # Always run bcrypt regardless of whether user exists — prevents username
    # enumeration via response-time difference (timing oracle attack).
    stored_hash = row["password_hash"].encode() if row else _DUMMY_HASH.encode()
    password_ok = bcrypt.checkpw(password.encode(), stored_hash)

    if not row or not password_ok:
        return None, None, None, "اسم المستخدم أو كلمة المرور غير صحيحة"

    token = secrets.token_urlsafe(32)
    now   = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO tokens (token, user_id, username, created_at, expires_at) VALUES (?,?,?,?,?)",
            (_hash_token(token), row["id"], row["username"], now, now + TOKEN_TTL)
        )

    logger.info(f"Login: '{row['username']}'")
    return token, row["id"], row["username"], None


_TOKEN_CACHE:     dict[str, tuple[str, str, float]] = {}   # token → (user_id, username, cached_at)
_TOKEN_CACHE_TTL  = 30.0   # re-validate against SQLite every 30s; covers burst concurrency
_TOKEN_CACHE_MAX  = 10_000  # hard cap — evict oldest on overflow
_TOKEN_CACHE_LOCK = threading.Lock()


def verify_token(token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Verify token and return (user_id, username) or (None, None).
    Extends expiry on each valid use (sliding window).

    In-memory cache avoids SQLite round-trip on every request.
    Multiple concurrent requests with the same token hit the cache after the first DB call.
    """
    if not token:
        return None, None
    now = time.time()

    # Fast path: cached recently-verified token
    with _TOKEN_CACHE_LOCK:
        entry = _TOKEN_CACHE.get(token)
        if entry is not None and now - entry[2] < _TOKEN_CACHE_TTL:
            return entry[0], entry[1]

    # Slow path: check SQLite (tokens stored hashed; fall back to legacy
    # plaintext rows once and upgrade them in place)
    token_hash = _hash_token(token)
    with _db() as conn:
        row = conn.execute(
            "SELECT user_id, username, expires_at FROM tokens WHERE token=?",
            (token_hash,)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT user_id, username, expires_at FROM tokens WHERE token=?",
                (token,)
            ).fetchone()
            if row:
                conn.execute("UPDATE tokens SET token=? WHERE token=?",
                             (token_hash, token))
        if not row or row["expires_at"] < now:
            with _TOKEN_CACHE_LOCK:
                _TOKEN_CACHE.pop(token, None)
            return None, None
        # Slide expiry window forward
        conn.execute(
            "UPDATE tokens SET expires_at=? WHERE token=?",
            (now + TOKEN_TTL, token_hash)
        )

    user_id, username = row["user_id"], row["username"]
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE[token] = (user_id, username, now)
        if len(_TOKEN_CACHE) > _TOKEN_CACHE_MAX:
            # Evict the oldest half by cached_at timestamp
            oldest = sorted(_TOKEN_CACHE.items(), key=lambda x: x[1][2])
            for k, _ in oldest[:_TOKEN_CACHE_MAX // 2]:
                del _TOKEN_CACHE[k]
    return user_id, username


def revoke_token(token: str):
    """Delete a specific token (logout) — covers hashed and legacy rows."""
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE.pop(token, None)
    with _db() as conn:
        conn.execute("DELETE FROM tokens WHERE token IN (?, ?)",
                     (_hash_token(token), token))


def revoke_all_tokens(user_id: str):
    """Delete all tokens for a user (logout everywhere)."""
    with _db() as conn:
        conn.execute("DELETE FROM tokens WHERE user_id=?", (user_id,))


def _cleanup_expired():
    """Remove expired tokens from DB (called periodically)."""
    with _db() as conn:
        n = conn.execute("DELETE FROM tokens WHERE expires_at < ?", (time.time(),)).rowcount
    if n:
        logger.debug(f"Cleaned up {n} expired tokens")


def user_count() -> int:
    with _db() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# ── FastAPI dependency ─────────────────────────────────────────────────────────

class CurrentUser:
    def __init__(self, user_id: str, username: str):
        self.user_id  = user_id
        self.username = username


def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> CurrentUser:
    """
    FastAPI dependency — checks HttpOnly cookie first (web), then Bearer header (mobile).
    Raises 401 if neither is present or valid.
    """
    token = request.cookies.get("maktaba_token") or (creds.credentials if creds else None)
    user_id, username = verify_token(token or "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="يجب تسجيل الدخول أولاً",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return CurrentUser(user_id=user_id, username=username)


# Initialise DB on import
init_db()
