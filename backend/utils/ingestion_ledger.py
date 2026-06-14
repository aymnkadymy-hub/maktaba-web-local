"""
Ingestion ledger — tracks every book's ingest state to prevent data loss.

Stored at PROJECT_ROOT/ingestion_ledger.json.

States:
  in_progress  — ingest started but not finished (crash/power-loss → re-ingest)
  complete     — all chunks stored, sha256 verified
  failed       — ingest threw an exception (will retry on next startup)

The sha256 of the PDF file is the key: if you replace a book file with a
new version, the hash changes and the book is automatically re-ingested.
"""
import os
import json
import hashlib
import time
import threading
import logging

logger     = logging.getLogger("ingestion_ledger")
_lock      = threading.Lock()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER_PATH  = os.path.join(PROJECT_ROOT, "ingestion_ledger.json")


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load() -> dict:
    if os.path.isfile(LEDGER_PATH):
        # utf-8-sig strips BOM if present (PowerShell writes UTF-8 with BOM by default)
        with open(LEDGER_PATH, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def _save(ledger: dict):
    # Write to a temp file then rename — atomic on POSIX, best-effort on Windows.
    # Prevents a crash mid-write from leaving the ledger in a partial/corrupt state.
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, LEDGER_PATH)


def _key(user_id: str, title: str) -> str:
    return f"{user_id}/{title}"


def is_complete(title: str, sha256: str, user_id: str = "") -> bool:
    """True only if this exact file version was fully ingested."""
    with _lock:
        entry = _load().get(_key(user_id, title), {})
        return entry.get("status") == "complete" and entry.get("sha256") == sha256


def is_failed(title: str, sha256: str, user_id: str = "") -> bool:
    """True if this exact file version previously failed — skip retrying same corrupt file."""
    with _lock:
        entry = _load().get(_key(user_id, title), {})
        return entry.get("status") == "failed" and entry.get("sha256") == sha256


def mark_started(title: str, sha256: str, total_pages: int, user_id: str = ""):
    with _lock:
        d = _load()
        d[_key(user_id, title)] = {
            "status":       "in_progress",
            "sha256":       sha256,
            "total_pages":  total_pages,
            "chunks_added": 0,
            "started_at":   time.time(),
        }
        _save(d)
    logger.info(f"[LEDGER] started  '{title}' ({total_pages} pages)")


def mark_complete(title: str, chunks_added: int, user_id: str = ""):
    with _lock:
        d = _load()
        k = _key(user_id, title)
        # Upsert: persist the terminal state even if the started-entry is gone
        # (ledger reset/race) — otherwise completion is silently lost and the
        # book gets needlessly re-ingested on next reload.
        d.setdefault(k, {})
        d[k].update({
            "status":       "complete",
            "chunks_added": chunks_added,
            "completed_at": time.time(),
        })
        _save(d)
    logger.info(f"[LEDGER] complete '{title}' ({chunks_added} chunks)")


def mark_failed(title: str, error: str, user_id: str = ""):
    with _lock:
        d = _load()
        k = _key(user_id, title)
        d.setdefault(k, {})
        d[k].update({
            "status":    "failed",
            "error":     str(error)[:500],
            "failed_at": time.time(),
        })
        _save(d)
    logger.warning(f"[LEDGER] failed  '{title}': {error}")


def remove(title: str, user_id: str = ""):
    """Remove ledger entry."""
    with _lock:
        d = _load()
        d.pop(_key(user_id, title), None)
        _save(d)
    logger.info(f"[LEDGER] removed '{title}'")


def get_all() -> dict:
    with _lock:
        return _load()
