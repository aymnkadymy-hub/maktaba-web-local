"""Per-user BM25 sub-indexing: a minority user's chunks must rank correctly
even when another user dominates the global corpus (live split: 97%/2%/1%).

Needs bm25s + numpy + qdrant-client + langchain-core, so skipped in the
lightweight test env and run in the project .venv.
"""
import threading

import pytest

pytest.importorskip("bm25s")
pytest.importorskip("numpy")
pytest.importorskip("qdrant_client")

from langchain_core.documents import Document
from backend.rag.hybrid_retriever import HybridRetriever, _normalize, _AR_STOPWORDS
import bm25s


def _make_retriever(corpus):
    """Build a HybridRetriever with a global bm25 index over `corpus`,
    bypassing the heavy __init__ (Qdrant scroll, background threads)."""
    r = object.__new__(HybridRetriever)
    r._corpus = corpus
    r._bm25_lib = "bm25s"
    r._user_bm25 = {}
    r._user_bm25_lock = threading.Lock()
    normed = [_normalize(d.page_content) for d in corpus]
    tokens = bm25s.tokenize(normed, stopwords=_AR_STOPWORDS, show_progress=False)
    idx = bm25s.BM25()
    idx.index(tokens, show_progress=False)
    r._bm25 = idx
    return r


def _corpus():
    # Reproduce the live failure mode deterministically: "big" dominates with
    # 300 chunks where the query term has very high term-frequency, so they all
    # outrank "small"'s 3 chunks where the same term appears once amid filler.
    # Globally, small's chunks fall outside the top-N and get filtered to ~zero.
    docs = []
    for i in range(300):
        docs.append(Document(page_content="كهرباء كهرباء كهرباء كهرباء كهرباء",
                             metadata={"user_id": "big", "book_title": "big_book"}))
    for i in range(3):
        filler = " ".join(f"كلمة{i}_{j}" for j in range(20))
        docs.append(Document(page_content=f"{filler} كهرباء {filler}",
                             metadata={"user_id": "small", "book_title": "small_book"}))
    return docs


def test_minority_user_gets_own_results():
    r = _make_retriever(_corpus())
    hits = r._bm25_search("كهرباء", n=5, user_id="small")
    assert hits, "minority user must get results from their own sub-index"
    assert all(d.metadata["user_id"] == "small" for d in hits), \
        "must never leak another user's chunks"


def test_per_user_beats_global_filter_for_minority():
    """A/B: with the per-user index the minority user gets their chunks; the
    global-index + post-filter path (simulated by disabling per-user) returns
    fewer — demonstrating both the bug and the fix in one test."""
    r = _make_retriever(_corpus())

    per_user = r._bm25_search("كهرباء", n=5, user_id="small")

    # Disable the per-user index → forces the old global+filter path
    def _no_user_index(_uid):
        return None, None
    r._get_user_bm25 = _no_user_index
    global_filtered = r._bm25_search("كهرباء", n=5, user_id="small")

    assert len(per_user) >= 3, "per-user index surfaces all 3 of small's chunks"
    assert len(global_filtered) < len(per_user), (
        f"global+filter buries the minority user "
        f"(global={len(global_filtered)} vs per-user={len(per_user)})"
    )


def test_big_user_unaffected():
    r = _make_retriever(_corpus())
    hits = r._bm25_search("كهرباء", n=5, user_id="big")
    assert hits and all(d.metadata["user_id"] == "big" for d in hits)


def test_book_title_filter_within_user():
    r = _make_retriever(_corpus())
    hits = r._bm25_search("كهرباء", n=5, user_id="small", book_title="small_book")
    assert all(d.metadata["book_title"] == "small_book" for d in hits)
    none = r._bm25_search("كهرباء", n=5, user_id="small", book_title="nonexistent")
    assert none == []


def test_unknown_user_returns_empty():
    r = _make_retriever(_corpus())
    assert r._bm25_search("كهرباء", n=5, user_id="ghost") == []


def test_index_caches_and_invalidates_on_corpus_change():
    r = _make_retriever(_corpus())
    r._bm25_search("كهرباء", n=5, user_id="small")
    assert "small" in r._user_bm25
    state_before = r._user_bm25["small"][0]
    # Simulate a corpus rebuild (new list object) → cache must not be reused
    r._corpus = list(r._corpus)
    idx, corpus = r._get_user_bm25("small")
    assert r._user_bm25["small"][0] != state_before
