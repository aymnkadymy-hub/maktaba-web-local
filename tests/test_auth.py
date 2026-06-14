"""Auth: bcrypt registration/login, SHA-256 token storage, legacy upgrade."""
import hashlib
import secrets
import sqlite3
import time

from backend.auth import auth


def _db_tokens():
    con = sqlite3.connect(auth.AUTH_DB)
    try:
        return [r[0] for r in con.execute("SELECT token FROM tokens").fetchall()]
    finally:
        con.close()


def test_register_and_login():
    user_id, err = auth.register("testuser", "secret123")
    assert err is None and user_id
    token, uid, uname, err = auth.login("testuser", "secret123")
    assert err is None and token and uid == user_id and uname == "testuser"


def test_token_stored_hashed_not_plaintext():
    token, *_ = auth.login("testuser", "secret123")
    rows = _db_tokens()
    assert token not in rows, "raw token must never be stored"
    assert hashlib.sha256(token.encode()).hexdigest() in rows


def test_verify_token_roundtrip():
    token, uid, uname, _ = auth.login("testuser", "secret123")
    user_id, username = auth.verify_token(token)
    assert user_id == uid and username == "testuser"


def test_legacy_plaintext_token_upgraded():
    uid, _ = auth.register("legacyuser", "secret123") or (None, None)
    uid = uid or auth.login("legacyuser", "secret123")[1]
    legacy = secrets.token_urlsafe(32)
    now = time.time()
    con = sqlite3.connect(auth.AUTH_DB)
    con.execute("INSERT INTO tokens VALUES (?,?,?,?,?)",
                (legacy, uid, "legacyuser", now, now + 3600))
    con.commit()
    con.close()

    user_id, _ = auth.verify_token(legacy)
    assert user_id == uid, "legacy token must still verify"
    rows = _db_tokens()
    assert legacy not in rows, "legacy row must be upgraded in place"
    assert hashlib.sha256(legacy.encode()).hexdigest() in rows


def test_revoke_token():
    token, *_ = auth.login("testuser", "secret123")
    auth.revoke_token(token)
    auth._TOKEN_CACHE.clear()   # bypass the 30s verification cache
    assert auth.verify_token(token) == (None, None)


def test_bad_credentials_rejected():
    assert auth.login("testuser", "wrongpass")[0] is None
    assert auth.login("no_such_user", "whatever")[0] is None


def test_weak_inputs_rejected():
    assert auth.register("ab", "secret123")[0] is None      # username too short
    assert auth.register("validname", "12345")[0] is None    # password too short
    assert auth.register("testuser", "secret123")[0] is None  # duplicate username
