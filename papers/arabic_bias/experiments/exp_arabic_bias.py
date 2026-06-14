#!/usr/bin/env python3
"""
Arabic algorithmic-bias measurements — REAL numbers from the live models + corpus.
Deterministic (no RNG). Produces the empirical evidence for the Arabic-bias paper:

  Exp A  Cross-lingual SELF-MATCH score gap. A pseudo-query = the first words of a
         chunk, scored against THAT SAME chunk (a guaranteed-relevant, ~lexically
         identical pair). If the multilingual scorer were unbiased, Arabic and
         English self-matches would score equally. We measure the gap (cross-encoder
         logits AND embedding cosine) plus the irrelevant (cross-document) band.
  Exp B  Tokenization fertility: tokens-per-word for Arabic vs English under the
         reranker's own subword tokenizer (the "subword penalty").
  Exp C  Score-distribution stats (mean / median / std) per language — the
         "score-compression" claim, measured rather than asserted.

Corpus: the live local library (not redistributed). Models from the local HF cache.
"""
import os, json, statistics as st
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CORPUS = os.path.expanduser("~/Desktop/maktaba-web-local/bm25_cache/corpus.json")
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

def is_arabic(s, thresh=0.25):
    ar = sum(1 for c in s if '؀' <= c <= 'ۿ')
    letters = sum(1 for c in s if c.isalpha() or '؀' <= c <= 'ۿ')
    return letters > 0 and ar / max(1, letters) > thresh

def even(lst, n):
    if not lst: return []
    step = max(1, len(lst) // n)
    return [lst[(i * step) % len(lst)] for i in range(min(n, len(lst)))]

def stats(x):
    return dict(n=len(x), mean=round(st.mean(x), 3), median=round(st.median(x), 3),
                std=round(st.pstdev(x), 3), min=round(min(x), 3), max=round(max(x), 3))

def main():
    rows = json.load(open(CORPUS))
    AR, EN = [], []
    for x in rows:
        t = (x.get("c", "") or "").strip()
        if len(t) < 60: continue
        (AR if is_arabic(t) else EN).append(t)
    print(f"corpus: {len(AR)} Arabic chunks, {len(EN)} English chunks (>=60 chars)")
    N = min(40, len(AR))
    ar, en = even(AR, N), even(EN, N)
    print(f"sampled {len(ar)} AR + {len(en)} EN chunks (deterministic)\n")

    from sentence_transformers import CrossEncoder, SentenceTransformer
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    fw = lambda t, n=12: " ".join(t.split()[:n])
    ce_s = lambda pairs: [float(s) for s in ce.predict([[q, p] for q, p in pairs])]
    def cos_s(pairs):
        qs = emb.encode([q for q, _ in pairs], normalize_embeddings=True)
        ps = emb.encode([p for _, p in pairs], normalize_embeddings=True)
        return [float((qs[i] * ps[i]).sum()) for i in range(len(pairs))]

    def bands(chunks, other):
        inp = [(fw(c), c[:250]) for c in chunks]
        outp = [(fw(c), other[(i * 5) % len(other)][:250]) for i, c in enumerate(chunks)]
        return ce_s(inp), ce_s(outp), cos_s(inp), cos_s(outp)

    ar_in_ce, ar_out_ce, ar_in_cos, ar_out_cos = bands(ar, en)
    en_in_ce, en_out_ce, en_in_cos, en_out_cos = bands(en, ar)

    R = {}
    print("===== Exp A — cross-lingual SELF-MATCH (guaranteed-relevant) score gap =====")
    for space, (a, e) in [("cross_encoder", (ar_in_ce, en_in_ce)), ("cosine", (ar_in_cos, en_in_cos))]:
        gap = round(st.mean(e) - st.mean(a), 3)
        R[f"{space}_selfmatch"] = {"arabic": stats(a), "english": stats(e), "english_minus_arabic_mean": gap}
        print(f"  [{space}] AR self-match mean={st.mean(a):.3f}  EN self-match mean={st.mean(e):.3f}  "
              f"GAP(EN-AR)={gap:+.3f}  ({'Arabic scored LOWER' if gap>0 else 'no penalty'})")
    print("\n  (irrelevant cross-document band, for reference)")
    for space, (a, e) in [("cross_encoder", (ar_out_ce, en_out_ce)), ("cosine", (ar_out_cos, en_out_cos))]:
        R[f"{space}_crossdoc"] = {"arabic": stats(a), "english": stats(e)}
        print(f"  [{space}] AR irrelevant mean={st.mean(a):.3f}  EN irrelevant mean={st.mean(e):.3f}")

    print("\n===== Exp B — tokenization fertility (subword penalty) =====")
    tok = ce.tokenizer
    def fert(chunks):
        tw = sum(len(c.split()) for c in chunks)
        tt = sum(len(tok.tokenize(c)) for c in chunks)
        return tt / max(1, tw)
    fa, fe = fert(ar), fert(en)
    R["tokenization_fertility"] = {"arabic_tokens_per_word": round(fa, 3),
                                   "english_tokens_per_word": round(fe, 3),
                                   "arabic_over_english": round(fa / fe, 2)}
    print(f"  Arabic: {fa:.3f} tokens/word   English: {fe:.3f} tokens/word   "
          f"Arabic is {fa/fe:.2f}x more fragmented")

    print("\n===== Exp C — score-distribution compression (self-match) =====")
    R["compression"] = {
        "cross_encoder": {"arabic_std": stats(ar_in_ce)["std"], "english_std": stats(en_in_ce)["std"]},
        "cosine": {"arabic_std": stats(ar_in_cos)["std"], "english_std": stats(en_in_cos)["std"]},
    }
    print(f"  cross-encoder self-match std: AR={stats(ar_in_ce)['std']}  EN={stats(en_in_ce)['std']}")
    print(f"  cosine self-match std:        AR={stats(ar_in_cos)['std']}  EN={stats(en_in_cos)['std']}")

    out = os.path.join(HERE, "results", "exp_arabic_bias.json")
    json.dump(R, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
