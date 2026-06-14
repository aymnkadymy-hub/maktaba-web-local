"""
Self-calibrating, per-tenant retrieval relevance gate.

Problem this solves
-------------------
A cross-encoder reranker emits an unbounded relevance score per (query, passage)
pair. To decide "is any retrieved passage actually relevant, or should the model
answer from general knowledge instead of forcing an off-topic book into the
prompt?" the system needs a CUTOFF. A single hand-tuned global cutoff (we used
-5.0) is brittle: the score distribution shifts with the corpus, the language
mix, and the cross-encoder's weaker calibration on some languages (e.g. Arabic
scores run lower than English for equally-relevant passages). One global number
therefore over-filters some tenants and under-filters others.

The invention
-------------
Calibrate the cutoff PER TENANT, automatically, from the tenant's own content —
no labels, no manual tuning:

  1. Sample chunks from the tenant's library, grouped by book.
  2. Build two score populations with the cross-encoder:
       • in-domain  : (query derived from a chunk, another chunk of the SAME book)
                      → topically related → the tenant's "relevant" score band.
       • out-domain : (that query, a chunk from a DIFFERENT book)
                      → unrelated → the tenant's "irrelevant" score band.
  3. Place the cutoff in the gap between the two bands (a high percentile of the
     irrelevant band, not exceeding a low percentile of the relevant band).
  4. Cache per tenant; auto-invalidate when the corpus changes (corpus id,len).

The cutoff adapts to each tenant's corpus and language automatically. A query
whose best reranked score falls below the tenant's own cutoff is treated as a
miss → triggers the cross-lingual translation fallback, then general knowledge.

Falls back to the global default when the model is unavailable or the tenant's
corpus is too small/homogeneous to calibrate (single book, too few chunks).
"""
import os
import logging

logger = logging.getLogger("relevance_gate")

_DEFAULT_CUTOFF = float(os.getenv("RERANK_MIN_SCORE", "-5.0"))
_SAMPLES        = int(os.getenv("GATE_CALIB_SAMPLES", "8"))   # chunks sampled per side
_MIN_BOOKS      = 2     # need ≥2 books to form in/out-domain pairs
_MIN_CHUNKS     = 6     # below this, distribution is too noisy → use default
# Cutoff is clamped to a sane band so a pathological calibration can't make the
# gate accept-everything or reject-everything.
_CUTOFF_FLOOR   = -8.0
_CUTOFF_CEIL    = 2.0


def _query_from_chunk(text: str, n_words: int = 12) -> str:
    """Derive a short pseudo-query from a chunk: its first n_words."""
    return " ".join(text.split()[:n_words])


def _percentile(sorted_vals: list, pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _xenc_score_pairs(pairs):
    """Default scorer: the cross-encoder reranker (unbounded scores)."""
    from backend.rag.reranker import score_pairs
    return score_pairs(pairs)


# A "scorer profile" makes the gate model-agnostic. Each downstream relevance
# scorer lives in its own score space, so it needs its own default + clamp band.
# This is what lets the SAME self-calibration run with a heavy cross-encoder OR
# with the lightweight embedding model already present on every device — so the
# gate works even where torch / a reranker can't run (e.g. low-end student
# laptops and phones).
class ScorerProfile:
    def __init__(self, name, score_fn, default, lo, hi):
        self.name = name
        self.score_fn = score_fn
        self.default = default
        self.lo = lo
        self.hi = hi


XENC_PROFILE = ScorerProfile("cross-encoder", _xenc_score_pairs,
                             _DEFAULT_CUTOFF, _CUTOFF_FLOOR, _CUTOFF_CEIL)


class RelevanceGate:
    """Computes and caches a per-tenant relevance cutoff for a given scorer."""

    def __init__(self):
        # (tenant_id, scorer_name) -> (corpus_state, cutoff)
        self._cache: dict = {}

    def cutoff_for(self, tenant_id: str, corpus: list, corpus_state,
                   profile: "ScorerProfile" = XENC_PROFILE) -> float:
        """Return the calibrated cutoff for this tenant in `profile`'s score
        space, computing it once per (tenant, scorer, corpus-state). Never
        raises — returns the profile default on any issue."""
        key = (tenant_id, profile.name)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == corpus_state:
            return cached[1]

        cutoff = self._calibrate(corpus, profile)
        if len(self._cache) > 5000:        # bound memory across many tenants
            self._cache.clear()
        self._cache[key] = (corpus_state, cutoff)
        return cutoff

    def _calibrate(self, corpus: list, profile: "ScorerProfile") -> float:
        try:
            if len(corpus) < _MIN_CHUNKS:
                return profile.default

            # Group chunk texts by book
            by_book: dict = {}
            for d in corpus:
                t = (d.page_content or "").strip()
                if len(t) < 40:
                    continue
                book = d.metadata.get("book_title", "?")
                by_book.setdefault(book, []).append(t)
            books = [b for b, v in by_book.items() if v]
            if len(books) < _MIN_BOOKS:
                return profile.default

            # Deterministic, dependency-light sampling (no RNG → reproducible):
            # walk books round-robin and take evenly-spaced chunks.
            # in-domain  = query-from-chunk vs the SAME chunk → guaranteed
            #              relevant (a "same-book different-chunk" signal proved
            #              too weak: a big multi-topic book's chunks aren't
            #              mutually relevant at the chunk level).
            # out-domain = that query vs a chunk from a DIFFERENT book.
            in_pairs, out_pairs = [], []
            for i in range(_SAMPLES):
                b = books[i % len(books)]
                chunks = by_book[b]
                c0 = chunks[(i * 7) % len(chunks)]
                q  = _query_from_chunk(c0)
                in_pairs.append((q, c0[:250]))                 # self-match (relevant)
                ob = books[(i + 1) % len(books)]
                if ob != b and by_book[ob]:
                    oc = by_book[ob][(i * 5) % len(by_book[ob])]
                    out_pairs.append((q, oc[:250]))            # cross-book (irrelevant)

            if len(in_pairs) < 3 or len(out_pairs) < 3:
                return profile.default

            in_scores  = sorted(profile.score_fn(in_pairs))
            out_scores = sorted(profile.score_fn(out_pairs))
            if len(in_scores) < 3 or len(out_scores) < 3:
                return profile.default

            # Place the cutoff in the GAP between the irrelevant band (top of
            # out-domain) and the relevant band (bottom of in-domain). The
            # midpoint keeps genuine matches while rejecting off-topic chunks.
            # If the bands overlap (no clean gap — common when the cross-encoder
            # is weakly calibrated for the language), we can't trust a learned
            # cutoff, so fall back to the safe global default.
            out_hi = _percentile(out_scores, 75)
            in_lo  = _percentile(in_scores, 25)
            if in_lo > out_hi:
                # Sit a quarter into the gap from the irrelevant side. The
                # in-domain anchor is self-match (an upper bound on relevance),
                # so real user-query matches score BETWEEN the bands — a
                # recall-favouring cutoff near the irrelevant band keeps them.
                cutoff = out_hi + 0.25 * (in_lo - out_hi)
            else:
                cutoff = profile.default
            cutoff = max(profile.lo, min(profile.hi, cutoff))

            logger.info(
                f"[GATE] {profile.name} cutoff={cutoff:.3f} "
                f"(in[25%]={in_lo:.3f} out[75%]={out_hi:.3f} gap={'yes' if in_lo>out_hi else 'no'} "
                f"books={len(books)} samples={len(in_pairs)}/{len(out_pairs)})"
            )
            return cutoff
        except Exception as e:
            logger.warning(f"[GATE] calibration failed ({e}) — using default")
            return profile.default


# Module-level singleton (one gate, keyed internally by tenant)
gate = RelevanceGate()
