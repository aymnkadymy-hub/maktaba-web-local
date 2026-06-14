#!/usr/bin/env python3
"""
Paper 1 — added rigor: (A) decompose the starvation loss into CROWDING vs
STATISTICS-CAPTURE (shared IDF), and (B) 95% bootstrap CIs on the headline.

Decomposition (all on shared-vocabulary minority probes, real corpus):
  overlap_full_dominance_F90   : shared pool + shared IDF, deployed over-fetch  (the measured loss)
  overlap_global_Finf          : shared IDF, B-pool (global index, retrieve ALL, filter to B, top-k)
                                 -> removes CROWDING only (no cutoff); keeps shared-IDF reweighting
  overlap_per_user (=1.0)      : tenant IDF + B-pool
  CROWDING effect  = overlap_global_Finf  - overlap_F90
  STAT-CAPTURE eff = 1.0 (per-user) - overlap_global_Finf   (IDF reweighting changes B's internal order)
Deterministic; bootstrap uses a fixed seed.
"""
import json, os, re, random
import bm25s
CORP=os.environ.get("MAKTABA_CORPUS", os.path.expanduser("~/maktaba-web-local/bm25_cache/corpus.json"))
OUT=os.path.join(os.path.dirname(__file__),"results","exp_p1b_ablation.json")
K=5; DF_MIN=30; SEED=0
def _n(t):
    t=re.sub(r'[ً-ٰٟ]','',t);t=re.sub(r'[أإآٱ]','ا',t);t=re.sub(r'ى','ي',t);t=re.sub(r'ة','ه',t);t=re.sub(r'ـ','',t);return t.lower()
STOP=set(['في','من','الي','علي','عن','مع','بين','حتي','لكن','او','ان','اذا','لو','بل','ثم','هو','هي','هم','هن','انا','نحن','انت','هذا','هذه','ذلك','تلك','الذي','التي','الذين','كان','كانت','يكون','تكون','لا','ما','لم','لن','قد','كل','بعض','جميع','غير','وفي','وعلي','ومن','وهو','وهي','وان'])
AR_STOP=list(STOP)
def ctoks(t): return [x for x in re.split(r'[^\w؀-ۿ]+',_n(t)) if len(x)>1 and x not in STOP]
def build(docs):
    nm=[_n(d["c"]) for d in docs]; pr=[(d,n) for d,n in zip(docs,nm) if n.strip()]; kept=[p[0] for p in pr]
    idx=bm25s.BM25(); idx.index(bm25s.tokenize([p[1] for p in pr],stopwords=AR_STOP,show_progress=False),show_progress=False)
    return idx,kept
def search(idx,kept,q,topn):
    qt=bm25s.tokenize([_n(q)],stopwords=AR_STOP,show_progress=False); topn=min(topn,len(kept))
    r,s=idx.retrieve(qt,k=topn,show_progress=False); return [kept[int(i)] for i,sc in zip(r[0],s[0]) if float(sc)>0]
def main():
    data=json.load(open(CORP))
    by={}
    for e in data:
        m=e.get("m",{}); md=m.get("metadata",m) if isinstance(m,dict) else {}; u=md.get("user_id")
        by.setdefault(u,[]).append({"c":e.get("c",""),"m":md})
    rk=sorted([(u,len(v)) for u,v in by.items() if u],key=lambda x:-x[1]); A,B=by[rk[0][0]],by[rk[1][0]]
    for i,d in enumerate(A): d["_k"]=("A",i)
    for i,d in enumerate(B): d["_k"]=("B",i)
    import collections; Adf=collections.Counter()
    for d in A:
        for t in set(ctoks(d["c"])): Adf[t]+=1
    probes=[]
    for d in B:
        sh=[t for t in ctoks(d["c"]) if Adf.get(t,0)>=DF_MIN]
        sh=sorted(dict.fromkeys(sh),key=lambda t:-Adf[t])[:8]
        if len(sh)>=3: probes.append((" ".join(sh),d["_k"]))
    seen=set(); probes=[(q,k) for q,k in probes if not (q in seen or seen.add(q))]
    # per-user oracle top-k
    bidx,bkept=build(B); oracle={}
    for q,tgt in probes: oracle[q]=set(d["_k"] for d in search(bidx,bkept,q,K))
    # global index over full corpus
    gidx,gkept=build(A+B)
    def overlap_for(q, mode, F):
        if mode=="global":
            docs=[d["_k"] for d in search(gidx,gkept,q,F) if d["_k"][0]=="B"][:K]
        orc=oracle[q]; return (len(set(docs)&orc)/len(orc)) if orc else None
    perq=[]
    for q,tgt in probes:
        ov_F90=overlap_for(q,"global",18*K)            # shared pool + shared IDF (deployed)
        ov_Finf=overlap_for(q,"global",len(gkept))     # B-pool + shared IDF (no crowding)
        perq.append((ov_F90,ov_Finf))
    import statistics as st
    n=len(probes)
    mean=lambda xs:sum(xs)/len(xs)
    ovF90=mean([a for a,_ in perq]); ovFinf=mean([b for _,b in perq])
    rep={"n_probes":n,"k":K,
         "overlap_deployed_F90":round(ovF90,4),
         "overlap_global_Finf_no_crowding":round(ovFinf,4),
         "overlap_per_user":1.0,
         "crowding_effect":round(ovFinf-ovF90,4),
         "statistics_capture_effect":round(1.0-ovFinf,4),
         "interpretation":"loss decomposes: removing the over-fetch cutoff (crowding) recovers `crowding_effect`; the residual gap to 1.0 is shared-IDF statistics capture"}
    # bootstrap 95% CI on overlap_deployed_F90
    random.seed(SEED); vals=[a for a,_ in perq]; boots=[]
    for _ in range(2000):
        s=[vals[random.randrange(n)] for _ in range(n)]; boots.append(sum(s)/n)
    boots.sort(); rep["overlap_deployed_F90_ci95"]=[round(boots[int(0.025*len(boots))],4),round(boots[int(0.975*len(boots))],4)]
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(rep,open(OUT,"w"),ensure_ascii=False,indent=2)
    print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
