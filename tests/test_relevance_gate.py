"""Self-calibrating relevance gate — per-tenant cutoff from score distribution."""
import sys
import types

from langchain_core.documents import Document
import backend.rag.relevance_gate as rg


def _doc(book, text):
    return Document(page_content=text, metadata={"book_title": book, "user_id": "u1"})


def _corpus():
    # Two books, several chunks each (enough to form in/out-domain pairs)
    a = [_doc("bookA", f"Alpha topic sentence number {i} about photosynthesis and leaves "
                       f"and chlorophyll energy conversion process detail {i}") for i in range(6)]
    b = [_doc("bookB", f"Beta topic sentence number {i} about quantum mechanics and "
                       f"wavefunctions and operators eigenvalues measurement detail {i}") for i in range(6)]
    return a + b


def _install_fake_reranker(monkeypatch, in_band, out_band):
    """Fake score_pairs: same-book(=longer shared prefix) → in_band, else out_band.
    We detect 'in-domain' by whether query and passage share the topic word."""
    def fake_score_pairs(pairs):
        scores = []
        for q, p in pairs:
            same = ("photosynthesis" in q and "photosynthesis" in p) or \
                   ("quantum" in q and "quantum" in p)
            scores.append(in_band if same else out_band)
        return scores
    fake = types.SimpleNamespace(score_pairs=fake_score_pairs)
    monkeypatch.setitem(sys.modules, "backend.rag.reranker", fake)


def test_calibrates_between_bands(monkeypatch):
    _install_fake_reranker(monkeypatch, in_band=3.0, out_band=-6.0)
    g = rg.RelevanceGate()
    corpus = _corpus()
    cut = g.cutoff_for("u1", corpus, (id(corpus), len(corpus)))
    # Cutoff must sit between the irrelevant (-6) and relevant (3) bands
    assert -6.0 <= cut <= 3.0
    assert cut > -6.0, "should be above the irrelevant band"
    assert cut < 3.0,  "should be below the relevant band"


def test_cache_by_corpus_state(monkeypatch):
    _install_fake_reranker(monkeypatch, in_band=2.0, out_band=-4.0)
    g = rg.RelevanceGate()
    corpus = _corpus()
    state = (id(corpus), len(corpus))
    c1 = g.cutoff_for("u1", corpus, state)
    assert g._cache[("u1", "cross-encoder")][0] == state
    c2 = g.cutoff_for("u1", corpus, state)
    assert c1 == c2


def test_small_corpus_uses_default(monkeypatch):
    _install_fake_reranker(monkeypatch, in_band=2.0, out_band=-4.0)
    g = rg.RelevanceGate()
    tiny = [_doc("bookA", "only one short chunk here")]
    cut = g.cutoff_for("u1", tiny, (1, 1))
    assert cut == rg._DEFAULT_CUTOFF


def test_single_book_uses_default(monkeypatch):
    _install_fake_reranker(monkeypatch, in_band=2.0, out_band=-4.0)
    g = rg.RelevanceGate()
    one_book = [_doc("bookA", f"chunk number {i} with enough words to pass the length floor "
                              f"and be sampled properly here now") for i in range(8)]
    cut = g.cutoff_for("u1", one_book, (id(one_book), len(one_book)))
    assert cut == rg._DEFAULT_CUTOFF  # needs >=2 books to calibrate


def test_scorer_agnostic_bi_encoder_profile():
    """The gate must calibrate in ANY scorer's space — here a cosine-like [0,1]
    space (the lightweight bi-encoder used on low-end devices, no cross-encoder)."""
    g = rg.RelevanceGate()
    corpus = _corpus()
    # Cosine-style scorer: self-match high (~0.85), cross-book low (~0.30)
    def cos_scorer(pairs):
        out = []
        for q, p in pairs:
            same = ("photosynthesis" in q and "photosynthesis" in p) or \
                   ("quantum" in q and "quantum" in p)
            out.append(0.85 if same else 0.30)
        return out
    prof = rg.ScorerProfile("bi-encoder", cos_scorer, default=0.30, lo=0.0, hi=0.95)
    cut = g.cutoff_for("u1", corpus, (id(corpus), len(corpus)), prof)
    assert 0.30 < cut < 0.85, f"cosine cutoff should sit in the gap, got {cut}"
    # cached under the bi-encoder key, separate from any cross-encoder entry
    assert ("u1", "bi-encoder") in g._cache


def test_never_raises_on_bad_model(monkeypatch):
    fake = types.SimpleNamespace(score_pairs=lambda pairs: [])  # model unavailable
    monkeypatch.setitem(sys.modules, "backend.rag.reranker", fake)
    g = rg.RelevanceGate()
    corpus = _corpus()
    cut = g.cutoff_for("u1", corpus, (id(corpus), len(corpus)))
    assert cut == rg._DEFAULT_CUTOFF
