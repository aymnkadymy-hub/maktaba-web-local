#!/usr/bin/env python3
"""
Experiment 2 — Labeled gate-decision accuracy (the core quantitative result).

The relevance gate makes a binary decision per query: ACCEPT (answer is grounded
in a retrieved passage) iff the best reranked passage score >= cutoff, else REJECT
(answer from the model's general knowledge / trigger fallback).

We build, per tenant, a LABELED query set with known ground truth:
  - IN  (relevant)   : a query = an 18-word span of a passage from the tenant's OWN
                       library; the source passage REMAINS in the candidate pool, so
                       this is an easy NEAR-SELF-MATCH. This is a known limitation that
                       inflates IN recall (it stays 1.0 throughout) — see the paper's
                       threats to validity. Ground truth: ACCEPT.
  - OUT (irrelevant) : a query about a topic ABSENT from the tenant's library
                       (derived from another tenant's books) + fixed
                       general-knowledge questions (EN+AR). Ground truth: REJECT.

For each query we retrieve candidates (BM25 over the tenant's chunks; cross-encoder
rerank of the top-k) and take the best score, exactly as the live gate would.

We then score four cutoff strategies on accept/reject correctness:
  (1) SELF-CALIBRATED  per-tenant tau  (LABEL-FREE — our method)
  (2) FIXED -5.0       the system's old hand-tuned global default
  (3) BEST GLOBAL      the single fixed cutoff maximizing pooled F1  (uses labels — oracle global)
  (4) PER-TENANT ORACLE the best cutoff per tenant                   (uses labels — upper bound)

Metrics: accuracy / precision / recall / F1 of ACCEPT (positive = relevant).
Claim proven if (1) label-free ~ (4) oracle and >> (2),(3).
"""
import os, json, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_calibration import calibrate, query_from_chunk, resolve, CORPUS, TENANTS

POOL_CAP = 120     # candidate passages scored per query (deterministic sample)
N_IN  = 12         # in-corpus (relevant) queries per tenant
N_OUT = 8          # out-of-corpus queries drawn from other tenants per tenant
TOPK  = 20         # BM25 candidates reranked per query

GK_QUERIES = [  # general-knowledge OUT queries (absent from every library)
    ("What is the capital of France?", "en"),
    ("Who wrote the play Hamlet?", "en"),
    ("ما هي عاصمة اليابان؟", "ar"),
    ("كم عدد كواكب المجموعة الشمسية؟", "ar"),
]

def load_corpus_with_meta():
    rows = []
    for x in json.load(open(CORPUS)):
        m = x.get("m", {}) if isinstance(x, dict) else {}
        t = (x.get("c", "") or "").strip()
        if len(t) < 40:
            continue
        rows.append((t, m.get("book_title", "?")))
    return rows

def even_sample(items, n, offset=0):
    if not items or n <= 0:
        return []
    step = max(1, len(items) // n)
    return [items[(offset + i * step) % len(items)] for i in range(min(n, len(items)))]

def metrics(scores_labels, cutoff):
    tp = fp = tn = fn = 0
    for s, lab in scores_labels:
        accept = s >= cutoff
        if lab == 1:   # relevant -> should accept
            tp += accept; fn += (not accept)
        else:          # irrelevant -> should reject
            fp += accept; tn += (not accept)
    acc = (tp + tn) / max(1, len(scores_labels))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)

def best_fixed_cutoff(all_sl):
    cands = sorted(set(round(s, 3) for s, _ in all_sl))
    best, bc = -1, cands[0] if cands else 0.0
    for c in cands:
        f1 = metrics(all_sl, c)["f1"]
        if f1 > best:
            best, bc = f1, c
    return bc

def main():
    print("Loading cross-encoder + BM25 (offline)...", flush=True)
    from sentence_transformers import CrossEncoder
    import bm25s, numpy as np
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    def ce_score(pairs):
        if not pairs:
            return []
        return [float(s) for s in ce.predict([[q, p] for q, p in pairs])]

    rows = load_corpus_with_meta()
    # resolve tenant book sets against real titles
    tenant_books = {name: set(resolve(wanted)) for name, wanted in TENANTS.items()}

    per_tenant = {}    # name -> list[(best_score, label)]
    cutoffs = {}       # name -> self-calibrated tau
    for name, books in tenant_books.items():
        T = [r for r in rows if r[1] in books]
        if len(T) < 6:
            continue
        by_book = {}
        for t, b in T:
            by_book.setdefault(b, []).append(t)
        tau, _ = calibrate(by_book, ce_score, default=-5.0, lo=-8.0, hi=2.0)
        cutoffs[name] = tau

        T_texts = [t for t, _ in T]
        # BM25 index over the tenant's chunks
        retr = bm25s.BM25()
        retr.index(bm25s.tokenize(T_texts, show_progress=False))

        # build queries
        in_src = even_sample(T, N_IN, offset=3)                  # held-out chunks
        other = [r for r in rows if r[1] not in books]
        out_src = even_sample(other, N_OUT, offset=5)
        # IN/OUT queries: an 18-word span from a passage (realistic length). For IN
        # the library DOES contain the answer (source kept in the pool); for OUT the
        # query is about a topic from a disjoint library, plus general knowledge.
        queries = [(query_from_chunk(t, 18), 1) for t, _ in in_src] \
                + [(query_from_chunk(t, 18), 0) for t, _ in out_src] \
                + [(q, 0) for q, _ in GK_QUERIES]

        sl = []
        for q, lab in queries:
            res, _ = retr.retrieve(bm25s.tokenize(q, show_progress=False),
                                   k=min(TOPK, len(T_texts)), show_progress=False)
            idxs = list(res[0])
            cand = [T_texts[i] for i in idxs]
            if not cand:
                cand = even_sample(T_texts, min(POOL_CAP, len(T_texts)))
            scores = ce_score([(q, c[:250]) for c in cand[:POOL_CAP]])
            best = max(scores) if scores else -8.0
            sl.append((best, lab))
        per_tenant[name] = sl
        print(f"  {name:<28} tau={tau:6.3f}  IN={sum(1 for _,l in sl if l==1)} OUT={sum(1 for _,l in sl if l==0)}", flush=True)

    # ---- evaluate the four strategies ----
    all_sl = [x for sl in per_tenant.values() for x in sl]
    global_best = best_fixed_cutoff(all_sl)
    strategies = {
        "(1) SELF-CALIBRATED (label-free)": "selfcal",
        "(2) FIXED -5.0 (old default)":     -5.0,
        f"(3) BEST GLOBAL fixed = {global_best:+.2f} (oracle, labels)": global_best,
        "(4) PER-TENANT ORACLE (labels)":   "oracle",
    }

    print("\n" + "=" * 78)
    print(f"{'strategy':<46}{'acc':>7}{'prec':>7}{'rec':>7}{'F1':>7}")
    print("-" * 78)
    summary = {}
    for label, strat in strategies.items():
        agg = []
        for name, sl in per_tenant.items():
            if strat == "selfcal":
                c = cutoffs[name]
            elif strat == "oracle":
                c = best_fixed_cutoff(sl)
            else:
                c = strat
            agg.extend([(1 if (s >= c) == (lab == 1) else 0) for s, lab in [(s, l) for s, l in sl]])
        m = metrics(all_sl, None) if False else None
        # recompute proper pooled metrics by re-deriving per-tenant decisions
        tp=fp=tn=fn=0
        for name, sl in per_tenant.items():
            if strat == "selfcal": c = cutoffs[name]
            elif strat == "oracle": c = best_fixed_cutoff(sl)
            else: c = strat
            for s, lab in sl:
                accept = s >= c
                if lab == 1: tp += accept; fn += (not accept)
                else: fp += accept; tn += (not accept)
        acc=(tp+tn)/max(1,tp+fp+tn+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
        f1=2*prec*rec/max(1e-9,prec+rec)
        summary[label] = dict(acc=acc,prec=prec,rec=rec,f1=f1,tp=tp,fp=fp,tn=tn,fn=fn)
        print(f"{label:<46}{acc:>7.3f}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}")
    print("=" * 78)

    os.makedirs(os.path.join(os.path.dirname(CORPUS), "..", "paper_experiments", "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp2_gate_accuracy.json")
    json.dump({"per_tenant_cutoffs": cutoffs,
               "per_tenant_scores": {k: v for k, v in per_tenant.items()},
               "global_best_cutoff": global_best,
               "summary": summary}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
