#!/usr/bin/env python3
"""
Paper 1 — Sparse-retrieval tenant starvation in a shared BM25 index.

Deterministic experiment on the REAL maktaba-web-local corpus. Reproduces the
two retrieval paths exactly as the deployed code implements them
(backend/rag/hybrid_retriever.py):

  GLOBAL path  : one BM25 index over ALL tenants' chunks; retrieve the global
                 top-F (F = over-fetch multiple of k), then POST-FILTER to the
                 querying tenant, then take top-k.   (search-then-filter)
  PER-USER path: a BM25 sub-index built over the querying tenant's own chunks;
                 retrieve top-k directly.            (filter-then-search)

Measures, for the MINORITY tenant, recall@k and zero-yield rate as a function
of (a) tenant dominance and (b) global over-fetch depth. Dominance is varied by
deterministically subsampling the dominant tenant (seeded; no other randomness).

Run:  ../../maktaba-web-local/.venv/bin/python exp_p1_starvation.py
"""
import json, os, re, random, time
import bm25s

CORPUS = os.environ.get("MAKTABA_CORPUS", os.path.expanduser("~/maktaba-web-local/bm25_cache/corpus.json"))
OUT    = os.path.join(os.path.dirname(__file__), "results", "exp_p1_starvation.json")
SEED   = 0
K      = 5

# ── faithful copy of the deployed Arabic normaliser + stopwords ───────────────
def _normalize(text: str) -> str:
    text = re.sub(r'[ً-ٰٟ]', '', text)   # strip tashkeel
    text = re.sub(r'[أإآٱ]', 'ا', text)  # unify alef
    text = re.sub(r'ى', 'ي', text)            # unify ya
    text = re.sub(r'ة', 'ه', text)            # unify ta-marbuta
    text = re.sub(r'ـ', '', text)                  # remove tatweel
    return text.lower()

_AR_STOPWORDS = ['في','من','الي','علي','عن','مع','بين','حتي','لكن','او','ان','اذا',
    'لو','بل','ثم','هو','هي','هم','هن','انا','نحن','انت','هذا','هذه','ذلك','تلك',
    'الذي','التي','الذين','كان','كانت','يكون','تكون','لا','ما','لم','لن','قد','كل',
    'بعض','جميع','غير','وفي','وعلي','ومن','وهو','وهي','وان']
_STOP = set(_AR_STOPWORDS)

def content_tokens(text):
    toks = re.split(r'[^\w؀-ۿ]+', _normalize(text))
    return [t for t in toks if len(t) > 1 and t not in _STOP]

def build_index(docs):
    """docs: list of dicts {c, m}. Returns (bm25, kept_docs) over non-empty norms."""
    normed = [_normalize(d["c"]) for d in docs]
    pairs = [(d, n) for d, n in zip(docs, normed) if n.strip()]
    kept = [p[0] for p in pairs]
    tokens = bm25s.tokenize([p[1] for p in pairs], stopwords=_AR_STOPWORDS, show_progress=False)
    idx = bm25s.BM25(); idx.index(tokens, show_progress=False)
    return idx, kept

def search(idx, kept, query, topn):
    q = bm25s.tokenize([_normalize(query)], stopwords=_AR_STOPWORDS, show_progress=False)
    topn = min(topn, len(kept))
    res, sc = idx.retrieve(q, k=topn, show_progress=False)
    return [(kept[int(i)], float(s)) for i, s in zip(res[0], sc[0]) if float(s) > 0]

def main():
    random.seed(SEED)
    data = json.load(open(CORPUS))
    # group by tenant
    by_uid = {}
    for e in data:
        m = e.get("m", {})
        md = m.get("metadata", m) if isinstance(m, dict) else {}
        uid = md.get("user_id")
        e2 = {"c": e.get("c", ""), "m": md, "id": id(e)}
        by_uid.setdefault(uid, []).append(e2)
    sizes = {u: len(v) for u, v in by_uid.items()}
    ranked = sorted([(u, n) for u, n in sizes.items() if u], key=lambda x: -x[1])
    DOM, MIN = ranked[0][0], ranked[1][0]
    A, B = by_uid[DOM], by_uid[MIN]
    # give every doc a stable identity for hit-testing
    for i, d in enumerate(A): d["_k"] = ("A", i)
    for i, d in enumerate(B): d["_k"] = ("B", i)

    # ── two minority-tenant probe sets (deterministic pseudo-queries) ──────────
    # DISTINCT  : query = a B chunk's leading content words (its own vocabulary).
    #             Reproduces the case where the minority topic is distinct.
    # SHARED    : query = a B chunk's tokens that ALSO occur frequently in the
    #             dominant tenant's corpus (shared/common academic vocabulary).
    #             Reproduces the realistic worst case that triggers starvation,
    #             since collection statistics are dominated by the 98%-tenant.
    # Target (relevant) for both = that same B chunk.
    import collections
    A_df = collections.Counter()
    for d in A:
        for t in set(content_tokens(d["c"])):
            A_df[t] += 1
    DF_MIN = 30  # token must appear in >=30 dominant-tenant chunks to count as "shared/common"

    distinct, shared = [], []
    for d in B:
        ct = content_tokens(d["c"])
        if len(ct) >= 4:
            distinct.append((" ".join(ct[:10]), d["_k"]))
        sh = [t for t in ct if A_df.get(t, 0) >= DF_MIN]
        # rank shared tokens by dominant-tenant frequency (most-common first)
        sh = sorted(dict.fromkeys(sh), key=lambda t: -A_df[t])[:8]
        if len(sh) >= 3:
            shared.append((" ".join(sh), d["_k"]))
    dseen=set(); distinct=[(q,k) for q,k in distinct if not (q in dseen or dseen.add(q))]
    sseen=set(); shared=[(q,k) for q,k in shared if not (q in sseen or sseen.add(q))]
    PROBESETS = {"distinct_vocab": distinct, "shared_common_vocab": shared}
    probes = shared  # primary set for the headline starvation curves

    report = {
        "corpus": {
            "total_entries": len(data),
            "tenants": sizes,
            "dominant_tenant": DOM, "dominant_chunks": len(A),
            "minority_tenant": MIN, "minority_chunks": len(B),
            "dominance_excl_null": round(len(A)/(len(A)+len(B)), 4),
            "dominance_incl_null": round(len(A)/len(data), 4),
        },
        "config": {"k": K, "seed": SEED, "df_min_shared": DF_MIN,
                   "n_distinct_queries": len(distinct),
                   "n_shared_queries": len(shared)},
    }

    M_DEPLOYED = 18  # deployed BM25 over-fetch: fetch_n=k*6 then *3 inside _bm25_search

    # per-user index (filter-then-search) — dominance-independent — for both sets
    bidx, bkept = build_index(B)
    report["per_user_recall_at_k"] = {}
    oracle_topk = {}   # query -> set of B doc-keys that the per-user index returns @k
    for name, pset in PROBESETS.items():
        h=0
        for q, tgt in pset:
            ks = [d["_k"] for d,_ in search(bidx,bkept,q,K)]
            oracle_topk[q] = set(ks)
            h += int(tgt in ks)
        report["per_user_recall_at_k"][name] = round(h/len(pset),4)

    def eval_global(gidx, gkept, pset, F):
        hits=zero=0; yields=[]; overlaps=[]
        for q, tgt in pset:
            ranked_docs = search(gidx, gkept, q, F)
            bfilt = [d["_k"] for d,_ in ranked_docs if d["_k"][0]=="B"][:K]
            hits += int(tgt in bfilt); yields.append(len(bfilt)); zero += int(len(bfilt)==0)
            orc = oracle_topk.get(q, set())
            if orc:
                overlaps.append(len(set(bfilt) & orc)/len(orc))
        n=len(pset)
        return hits/n, zero/n, sum(yields)/n, (sum(overlaps)/len(overlaps) if overlaps else None)

    # ── Exp 1A: recall@k vs dominance, deployed over-fetch, avg over 5 seeds ───
    def dom_to_nA(d):
        return min(len(A), round(len(B)*d/(1-d)))
    DOMS = [0.50, 0.80, 0.90, 0.95, 0.98, len(A)/(len(A)+len(B))]
    SEEDS = [0,1,2,3,4]
    F = M_DEPLOYED*K
    report["expA_recall_vs_dominance"] = {"over_fetch_multiple": M_DEPLOYED, "seeds": SEEDS, "by_probeset": {}}
    for name, pset in PROBESETS.items():
        rows=[]
        for d in DOMS:
            nA = dom_to_nA(d)
            accs=[]; zeros=[]; ys=[]; ovs=[]
            for s in SEEDS:
                if nA < len(A):
                    random.seed(s); Asub = random.sample(A, nA)
                else:
                    Asub = A
                gidx, gkept = build_index(Asub + B)
                r,z,y,ov = eval_global(gidx, gkept, pset, F)
                accs.append(r); zeros.append(z); ys.append(y)
                if ov is not None: ovs.append(ov)
            rows.append({"dominance": round(nA/(nA+len(B)),4), "nA": nA,
                         "global_recall_at_k": round(sum(accs)/len(accs),4),
                         "global_zero_yield_rate": round(sum(zeros)/len(zeros),4),
                         "mean_B_yield": round(sum(ys)/len(ys),3),
                         "oracle_overlap_at_k": round(sum(ovs)/len(ovs),4) if ovs else None})
        report["expA_recall_vs_dominance"]["by_probeset"][name] = rows

    # ── Exp 1B: recall@k vs over-fetch depth at FULL real dominance ────────────
    gidx, gkept = build_index(A + B)
    expB = []
    for m in [1,3,6,18,50,100]:
        F = m*K; hits=0; zero=0
        for q, tgt in probes:
            ranked_docs = search(gidx, gkept, q, F)
            bfilt = [d["_k"] for d,_ in ranked_docs if d["_k"][0]=="B"][:K]
            hits += int(tgt in bfilt); zero += int(len(bfilt)==0)
        expB.append({"over_fetch_multiple": m, "global_fetch_F": F,
                     "global_recall_at_k": round(hits/len(probes),4),
                     "global_zero_yield_rate": round(zero/len(probes),4)})
    report["expB_recall_vs_overfetch_full_dominance"] = expB

    # ── Exp 1C: rank-of-target distribution in the global list (full corpus) ───
    ranks=[]
    for q, tgt in probes:
        full = search(gidx, gkept, q, len(gkept))
        bpos = [i for i,(d,_) in enumerate(full) if d["_k"][0]=="B"]
        # global rank (1-based) of the target chunk among ALL tenants
        pos = next((i+1 for i,(d,_) in enumerate(full) if d["_k"]==tgt), None)
        ranks.append(pos)
    found = [r for r in ranks if r]
    found.sort()
    report["expC_global_rank_of_target"] = {
        "n_found": len(found), "n_total": len(probes),
        "median_global_rank": found[len(found)//2] if found else None,
        "p90_global_rank": found[int(len(found)*0.9)] if found else None,
        "max_global_rank": max(found) if found else None,
        "frac_beyond_deployed_F": round(sum(1 for r in found if r> M_DEPLOYED*K)/len(probes),4),
    }

    # ── latency (per-user build+query vs global query) ─────────────────────────
    t0=time.perf_counter(); build_index(B); t_pu_build=(time.perf_counter()-t0)*1000
    report["latency_ms"] = {"per_user_index_build_minority": round(t_pu_build,2),
                            "note": "global index already resident; per-user built lazily once per tenant, cached"}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT,"w"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
