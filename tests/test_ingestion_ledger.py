"""Ingestion ledger — terminal state must always persist (pure stdlib, no deps)."""
import backend.utils.ingestion_ledger as led


def _fresh_ledger(tmp_path):
    """Point the ledger at a throwaway JSON file for one test."""
    led.LEDGER_PATH = str(tmp_path)
    return led


def test_normal_started_then_complete(tmp_path):
    led = _fresh_ledger(tmp_path / "l1.json")
    led.mark_started("book", "sha1", 10, user_id="u1")
    led.mark_complete("book", 42, user_id="u1")
    assert led.is_complete("book", "sha1", user_id="u1")
    entry = led.get_all()[led._key("u1", "book")]
    assert entry["status"] == "complete" and entry["chunks_added"] == 42


def test_complete_persists_even_without_started(tmp_path):
    """The bug: mark_complete used to no-op if the started-entry was missing,
    silently losing completion and forcing a re-ingest."""
    led = _fresh_ledger(tmp_path / "l2.json")
    led.mark_complete("orphan", 7, user_id="u1")          # no mark_started first
    entry = led.get_all().get(led._key("u1", "orphan"))
    assert entry is not None, "completion must be persisted, not dropped"
    assert entry["status"] == "complete" and entry["chunks_added"] == 7


def test_failed_persists_even_without_started(tmp_path):
    led = _fresh_ledger(tmp_path / "l3.json")
    led.mark_failed("orphan", "boom", user_id="u1")
    entry = led.get_all().get(led._key("u1", "orphan"))
    assert entry is not None and entry["status"] == "failed"
    assert entry["error"] == "boom"


def test_user_scoping(tmp_path):
    led = _fresh_ledger(tmp_path / "l4.json")
    led.mark_complete("shared", 1, user_id="userA")
    assert led.get_all().get(led._key("userA", "shared")) is not None
    assert led.get_all().get(led._key("userB", "shared")) is None
