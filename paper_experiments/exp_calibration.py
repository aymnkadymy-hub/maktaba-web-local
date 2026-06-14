#!/usr/bin/env python3
"""
Experiment 1 — Per-tenant self-calibrated relevance cutoffs (reproduction + extension).

Faithfully re-implements backend/rag/relevance_gate.py:_calibrate (same deterministic
sampling, same self-match in-domain / cross-book out-domain anchors, same
Q75(out)/Q25(in) gap rule, same alpha=0.25 recall-favouring placement, same clamp),
then runs it over heterogeneous themed TENANTS with two interchangeable scorers:
  - cross-encoder  : cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  (logit scores)
  - embedding cos  : paraphrase-multilingual-MiniLM-L12-v2       (cosine scores)

Goal: show one global cutoff cannot be right for all tenants, and that the
calibration is scorer-agnostic. All models are loaded from the local HF cache
(offline); the corpus is the live bm25_cache/corpus.json.
"""
import os, json, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "bm25_cache", "corpus.json")

# ---- calibration constants (mirror relevance_gate.py) ----
SAMPLES = int(os.getenv("GATE_SAMPLES", "8"))
ALPHA = float(os.getenv("GATE_ALPHA", "0.25"))   # recall-favouring placement in the gap
MIN_BOOKS = 2
MIN_CHUNKS = 6

def query_from_chunk(text, n=12):
    return " ".join(text.split()[:n])

def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]

def calibrate(by_book, score_fn, default, lo, hi):
    """Exact port of RelevanceGate._calibrate for a corpus already grouped by book."""
    books = [b for b, v in by_book.items() if v]
    if len(books) < MIN_BOOKS or sum(len(v) for v in by_book.values()) < MIN_CHUNKS:
        return default, None
    in_pairs, out_pairs = [], []
    for i in range(SAMPLES):
        b = books[i % len(books)]
        chunks = by_book[b]
        c0 = chunks[(i * 7) % len(chunks)]
        q = query_from_chunk(c0)
        in_pairs.append((q, c0[:250]))
        ob = books[(i + 1) % len(books)]
        if ob != b and by_book[ob]:
            oc = by_book[ob][(i * 5) % len(by_book[ob])]
            out_pairs.append((q, oc[:250]))
    if len(in_pairs) < 3 or len(out_pairs) < 3:
        return default, None
    in_scores = sorted(score_fn(in_pairs))
    out_scores = sorted(score_fn(out_pairs))
    out_hi = percentile(out_scores, 75)
    in_lo = percentile(in_scores, 25)
    gap = in_lo > out_hi
    if gap:
        cutoff = out_hi + ALPHA * (in_lo - out_hi)
    else:
        cutoff = default
    cutoff = max(lo, min(hi, cutoff))
    return cutoff, dict(in_lo=in_lo, out_hi=out_hi, gap=gap,
                        in_scores=[round(s, 3) for s in in_scores],
                        out_scores=[round(s, 3) for s in out_scores])

# ---- load corpus, filter valid chunks (text >= 40 chars, like the gate) ----
def load_books_for(titles):
    raw = json.load(open(CORPUS))
    by_book = {}
    for x in raw:
        m = x.get("m", {}) if isinstance(x, dict) else {}
        b = m.get("book_title", "?")
        if b not in titles:
            continue
        t = (x.get("c", "") or "").strip()
        if len(t) < 40:
            continue
        by_book.setdefault(b, []).append(t)
    return {b: v for b, v in by_book.items() if v}

# ---- heterogeneous themed tenants (private libraries) ----
TENANTS = {
    "T1 EN ML/AI-search": [
        "generative-deep-learning-teaching-machines-to-paint-write-compose-and-play",
        "AI_Search_Algorithms_100_Questions", "but_how_do_it_know"],
    "T2 EN economics/publishing": [
        "Freakonomics", "100 Q&A About Scientific Publishing - Beginner's Guide"],
    "T3 AR + math": [
        "اساسيات البرمجة", "اخيرة عربي", "الكسور الجزئيه", "INVERSE HYPERBOLIC FUNCTIONS"],
    "T4 Mixed AR+EN": [
        "generative-deep-learning-teaching-machines-to-paint-write-compose-and-play",
        "Freakonomics", "اساسيات البرمجة", "اخيرة عربي"],
    "T5 STEM/electronics": [
        "Zener-Diode-and-Voltage-Regulation", "INVERSE HYPERBOLIC FUNCTIONS", "الكسور الجزئيه"],
}

# Resolve fuzzy title matches against the real corpus titles
def resolve(titles_wanted):
    raw_titles = set()
    for x in json.load(open(CORPUS)):
        m = x.get("m", {}) if isinstance(x, dict) else {}
        raw_titles.add(m.get("book_title", "?"))
    out = []
    for w in titles_wanted:
        if w in raw_titles:
            out.append(w); continue
        cand = [t for t in raw_titles if t and (w[:20] in t or t[:20] in w)]
        if cand:
            out.append(cand[0])
    return out

def main():
    print("Loading models (offline, local cache)...", flush=True)
    from sentence_transformers import CrossEncoder, SentenceTransformer
    import numpy as np
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    try:
        emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print(f"  [embedding model unavailable: {type(e).__name__}] — cross-encoder only for now")
        emb = None

    def ce_score(pairs):
        return [float(s) for s in ce.predict([[q, p] for q, p in pairs])]

    def cos_score(pairs):
        qs = emb.encode([q for q, _ in pairs], normalize_embeddings=True)
        ps = emb.encode([p for _, p in pairs], normalize_embeddings=True)
        return [float((qs[i] * ps[i]).sum()) for i in range(len(pairs))]

    print(f"\n{'tenant':<28} {'#books':>6} {'#chunks':>8} | "
          f"{'XENC cutoff':>12} {'gap':>4} | {'COS cutoff':>11} {'gap':>4}")
    print("-" * 90)
    results = {}
    for name, wanted in TENANTS.items():
        titles = resolve(wanted)
        by_book = load_books_for(set(titles))
        nb = len(by_book); nc = sum(len(v) for v in by_book.values())
        ce_cut, ce_d = calibrate(by_book, ce_score, default=-5.0, lo=-8.0, hi=2.0)
        if emb is not None:
            cos_cut, cos_d = calibrate(by_book, cos_score, default=0.30, lo=0.0, hi=0.95)
        else:
            cos_cut, cos_d = float("nan"), None
        results[name] = dict(books=nb, chunks=nc, xenc=ce_cut, xenc_d=ce_d,
                             cos=cos_cut, cos_d=cos_d, titles=titles)
        g1 = "yes" if (ce_d and ce_d["gap"]) else "no"
        g2 = "yes" if (cos_d and cos_d["gap"]) else "no"
        print(f"{name:<28} {nb:>6} {nc:>8} | {ce_cut:>12.3f} {g1:>4} | {cos_cut:>11.3f} {g2:>4}")

    print("\n=== Interpretation ===")
    xs = [r['xenc'] for r in results.values()]
    cs = [r['cos'] for r in results.values()]
    print(f"cross-encoder cutoffs span [{min(xs):.2f}, {max(xs):.2f}]  (one global -5.0 cannot fit all)")
    print(f"cosine cutoffs span        [{min(cs):.3f}, {max(cs):.3f}]  (one global 0.30 cannot fit all)")

    os.makedirs(os.path.join(ROOT, "paper_experiments", "results"), exist_ok=True)
    out = os.path.join(ROOT, "paper_experiments", "results", "exp1_cutoffs.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
