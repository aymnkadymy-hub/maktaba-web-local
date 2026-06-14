#!/usr/bin/env python3
"""
Experiment 4 — Ablation of the gap-placement parameter alpha.

The cutoff is tau = out_hi + alpha*(in_lo - out_hi), clamped. alpha=0 sits the
cutoff right at the top of the irrelevant band (most lenient / highest recall);
alpha=1 raises it to the bottom of the relevant band (strictest / highest
precision). The system default is alpha=0.25 (recall-favouring).

This reuses the EXPENSIVE per-query best-scores already saved by exp2
(results/exp2_gate_accuracy.json) and only recomputes the cheap calibration
score-bands, so the whole sweep costs ~5s of cross-encoder time.
"""
import os, json, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_calibration import calibrate, resolve, TENANTS, CORPUS

HERE = os.path.dirname(os.path.abspath(__file__))
saved = json.load(open(os.path.join(HERE, "results", "exp2_gate_accuracy.json")))
per_tenant_scores = saved["per_tenant_scores"]   # name -> [[best_score, label], ...]

def metrics_pooled(cutoff_by_tenant):
    tp=fp=tn=fn=0
    for name, sl in per_tenant_scores.items():
        c = cutoff_by_tenant[name]
        for s, lab in sl:
            accept = s >= c
            if lab == 1: tp += accept; fn += (not accept)
            else: fp += accept; tn += (not accept)
    acc=(tp+tn)/max(1,tp+fp+tn+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
    f1=2*prec*rec/max(1e-9,prec+rec)
    return acc,prec,rec,f1

def main():
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    def ce_score(pairs): return [float(s) for s in ce.predict([[q,p] for q,p in pairs])] if pairs else []

    rows = []
    for x in json.load(open(CORPUS)):
        m = x.get("m", {}) if isinstance(x, dict) else {}
        t = (x.get("c","") or "").strip()
        if len(t) >= 40: rows.append((t, m.get("book_title","?")))

    # recompute calibration bands (in_lo, out_hi) per tenant
    bands = {}
    for name, wanted in TENANTS.items():
        if name not in per_tenant_scores: continue
        books = set(resolve(wanted)); by_book = {}
        for t,b in rows:
            if b in books: by_book.setdefault(b, []).append(t)
        _, det = calibrate(by_book, ce_score, default=-5.0, lo=-8.0, hi=2.0)
        bands[name] = det     # dict with in_lo, out_hi

    print(f"{'alpha':>6}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}   cutoffs (per tenant)")
    print("-"*78)
    results = {}
    for alpha in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
        cut = {}
        for name, det in bands.items():
            lo, hi = det["out_hi"], det["in_lo"]
            cut[name] = max(-8.0, min(2.0, lo + alpha*(hi-lo)))
        acc,prec,rec,f1 = metrics_pooled(cut)
        results[alpha] = dict(acc=acc,prec=prec,rec=rec,f1=f1,cutoffs={k:round(v,2) for k,v in cut.items()})
        tag = "  <- default" if abs(alpha-0.25)<1e-9 else ""
        cuts = " ".join(f"{v:+.1f}" for v in cut.values())
        print(f"{alpha:>6.2f}{acc:>8.3f}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}   {cuts}{tag}")

    json.dump(results, open(os.path.join(HERE, "results", "exp4_alpha_ablation.json"), "w"), indent=2)
    print("\nsaved -> results/exp4_alpha_ablation.json")
    print("Interpretation: low alpha = lenient (recall up, precision down); high alpha = strict.")

if __name__ == "__main__":
    main()
