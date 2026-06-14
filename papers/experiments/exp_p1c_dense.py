#!/usr/bin/env python3
"""
Paper 1 — MEASURED dense-retrieval baseline (turns the sparse-vs-dense argument
into evidence). Same real corpus, same shared-vocabulary minority probes, same
oracle-overlap@5 metric as the sparse experiment — but with dense embeddings.

Dense paths compared vs tenant dominance:
  dense_postfilter_8x : global FAISS, fetch 8k, post-filter to minority, top-k   (deployed dense path)
  dense_per_user      : FAISS over minority's own chunks, top-k                   (oracle; =1.0)
Because cosine similarity is corpus-composition-independent, the minority's
internal dense order does not change with dominance -> dense should stay near 1.0
where sparse collapsed to 0.46. Deterministic (fixed seeds, frozen model).
"""
import json, os, re, random
import numpy as np
CORP=os.environ.get("MAKTABA_CORPUS", os.path.expanduser("~/maktaba-web-local/bm25_cache/corpus.json"))
OUT=os.path.join(os.path.dirname(__file__),"results","exp_p1c_dense.json")
K=5; DF_MIN=30; SEEDS=[0,1,2,3,4]
def _n(t):
    t=re.sub(r'[ً-ٰٟ]','',t);t=re.sub(r'[أإآٱ]','ا',t);t=re.sub(r'ى','ي',t);t=re.sub(r'ة','ه',t);t=re.sub(r'ـ','',t);return t.lower()
STOP=set(['في','من','الي','علي','عن','مع','بين','حتي','لكن','او','ان','اذا','لو','بل','ثم','هو','هي','هم','هن','انا','نحن','انت','هذا','هذه','ذلك','تلك','الذي','التي','الذين','كان','كانت','يكون','تكون','لا','ما','لم','لن','قد','كل','بعض','جميع','غير','وفي','وعلي','ومن','وهو','وهي','وان'])
def ctoks(t): return [x for x in re.split(r'[^\w؀-ۿ]+',_n(t)) if len(x)>1 and x not in STOP]
def main():
    import collections
    from sentence_transformers import SentenceTransformer
    import faiss
    data=json.load(open(CORP))
    by={}
    for e in data:
        m=e.get("m",{}); md=m.get("metadata",m) if isinstance(m,dict) else {}; u=md.get("user_id")
        by.setdefault(u,[]).append({"c":e.get("c",""),"m":md})
    rk=sorted([(u,len(v)) for u,v in by.items() if u],key=lambda x:-x[1]); A,B=by[rk[0][0]],by[rk[1][0]]
    for i,d in enumerate(A): d["_k"]=("A",i)
    for i,d in enumerate(B): d["_k"]=("B",i)
    Adf=collections.Counter()
    for d in A:
        for t in set(ctoks(d["c"])): Adf[t]+=1
    probes=[]
    for d in B:
        sh=[t for t in ctoks(d["c"]) if Adf.get(t,0)>=DF_MIN]; sh=sorted(dict.fromkeys(sh),key=lambda t:-Adf[t])[:8]
        if len(sh)>=3: probes.append((" ".join(sh),d["_k"]))
    seen=set(); probes=[(q,k) for q,k in probes if not (q in seen or seen.add(q))]
    print(f"probes={len(probes)} | embedding {len(A)+len(B)} chunks + queries (CPU)...")
    model=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    def emb(texts): return np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False),dtype="float32")
    A_vec=emb([d["c"] for d in A]); B_vec=emb([d["c"] for d in B]); Q_vec=emb([q for q,_ in probes])
    def faiss_ip(vecs):
        idx=faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs); return idx
    # oracle = per-user dense top-k (B only)
    bidx=faiss_ip(B_vec)
    oracle=[]
    for i in range(len(probes)):
        _,I=bidx.search(Q_vec[i:i+1],min(K,len(B))); oracle.append(set(("B",int(j)) for j in I[0] if j>=0))
    def dom_to_nA(d): return min(len(A),round(len(B)*d/(1-d)))
    DOMS=[0.50,0.80,0.90,0.95,0.98,len(A)/(len(A)+len(B))]
    rows=[]
    for d in DOMS:
        nA=dom_to_nA(d); accs=[]
        for s in SEEDS:
            random.seed(s); idxA=random.sample(range(len(A)),nA) if nA<len(A) else list(range(len(A)))
            keys=[A[i]["_k"] for i in idxA]+[d2["_k"] for d2 in B]
            vecs=np.vstack([A_vec[idxA],B_vec])
            gidx=faiss_ip(vecs); F=8*K
            ovs=[]
            for i in range(len(probes)):
                _,I=gidx.search(Q_vec[i:i+1],min(F,len(keys)))
                got=[keys[int(j)] for j in I[0] if j>=0 and keys[int(j)][0]=="B"][:K]
                orc=oracle[i]
                if orc: ovs.append(len(set(got)&orc)/len(orc))
            accs.append(sum(ovs)/len(ovs))
        rows.append({"dominance":round(nA/(nA+len(B)),4),"nA":nA,
                     "dense_postfilter8x_overlap_at_k":round(sum(accs)/len(accs),4)})
    rep={"n_probes":len(probes),"k":K,"seeds":SEEDS,"embedding_model":"paraphrase-multilingual-MiniLM-L12-v2",
         "dense_per_user_overlap_at_k":1.0,
         "rows":rows,
         "note":"MEASURED, and it refutes a naive 'dense does not starve' claim: dense post-filter (8x over-fetch) overlap@5 ALSO collapses with dominance (1.0 -> 0.32 at the real 98.5%), in fact below the sparse path's 0.46 (which used a larger 18x over-fetch). Crowding -- the global top-N being committed BEFORE the tenant post-filter -- affects BOTH sparse and dense post-filtering. Cosine's corpus-composition-independence removes only the sparse-only IDF statistics-capture term (+0.14, exp_p1b), NOT crowding. The genuine asymmetry is structural: dense ANN can push the predicate INTO traversal (Filtered-DiskANN/ACORN) to eliminate crowding, whereas a shared inverted index has no analogue short of a per-tenant sub-index. The per-tenant sub-index restores overlap to 1.0 on BOTH sides."}
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(rep,open(OUT,"w"),ensure_ascii=False,indent=2)
    print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
