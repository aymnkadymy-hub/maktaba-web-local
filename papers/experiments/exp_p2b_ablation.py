#!/usr/bin/env python3
"""Paper 2 — glossary-size ablation: how cross-script recall scales with how much
of the static glossary is present. Deterministic; uses the project's own glossary
and bm25s on the real corpus. Subsamples the AR->EN map to fractions of its size
(deterministic prefix of length-sorted keys) and measures Arabic->English book-level
recall@10 on the labelled AR->EN query set."""
import json, os, re, sys
sys.path.insert(0,os.environ.get("MAKTABA_ROOT", os.path.expanduser("~/maktaba-web-local")))
import bm25s, backend.rag.glossary as G
CORP=os.environ.get("MAKTABA_CORPUS", os.path.expanduser("~/maktaba-web-local/bm25_cache/corpus.json"))
OUT=os.path.join(os.path.dirname(__file__),"results","exp_p2b_ablation.json")
TENANT="3CLDpsvv7RhtClg5iqhr-g"; K=10
def _n(t):
    t=re.sub(r'[ً-ٰٟ]','',t);t=re.sub(r'[أإآٱ]','ا',t);t=re.sub(r'ى','ي',t);t=re.sub(r'ة','ه',t);t=re.sub(r'ـ','',t);return t.lower()
AR_STOP=['في','من','الي','علي','عن','مع','بين','حتي','لكن','او','ان','اذا','لو','بل','ثم','هو','هي','هم','هن','انا','نحن','انت','هذا','هذه','ذلك','تلك','الذي','التي','الذين','كان','كانت','يكون','تكون','لا','ما','لم','لن','قد','كل','بعض','جميع','غير','وفي','وعلي','ومن','وهو','وهي','وان']
def build(docs):
    nm=[_n(d["c"]) for d in docs]; pr=[(d,n) for d,n in zip(docs,nm) if n.strip()]; kept=[p[0] for p in pr]
    idx=bm25s.BM25(); idx.index(bm25s.tokenize([p[1] for p in pr],stopwords=AR_STOP,show_progress=False),show_progress=False); return idx,kept
def search(idx,kept,q,topn):
    qt=bm25s.tokenize([_n(q)],stopwords=AR_STOP,show_progress=False); topn=min(topn,len(kept))
    r,s=idx.retrieve(qt,k=topn,show_progress=False); return [kept[int(i)] for i,sc in zip(r[0],s[0]) if float(sc)>0]
AR2EN=[("الشبكة العصبية التوليدية","generative-deep-learning"),("فرط التخصيص في الشبكة العصبية","generative-deep-learning"),
 ("خوارزمية البحث","AI_Search_Algorithms"),("الحوافز الاقتصادية","Freakonomics"),
 ("النشر العلمي ومعامل التأثير","100 Q&A About Scientific Publishing"),("التحكيم والسرقة الادبية","100 Q&A About Scientific Publishing")]
def main():
    data=json.load(open(CORP))
    A=[{"c":e.get("c",""),"m":(e.get("m",{}).get("metadata",e.get("m",{})) if isinstance(e.get("m"),dict) else {})} for e in data]
    A=[d for d in A if d["m"].get("user_id")==TENANT]
    idx,kept=build(A)
    full=dict(G._AR_EN)                              # save full glossary
    keys=sorted(full.keys(), key=lambda k:(-len(k),k))  # deterministic order (long keys first)
    def recall_at(frac):
        n=int(round(frac*len(keys))); sub={k:full[k] for k in keys[:n]}
        G._AR_EN.clear(); G._AR_EN.update(sub)
        hit=0
        for q,tgt in AR2EN:
            ext=G.cross_lingual_terms(_n(q)); qq=(q+" "+ext).strip() if ext else q
            bks=[d["m"].get("book_title","") for d in search(idx,kept,qq,K)]
            hit+=int(any(tgt.lower() in (b or "").lower() for b in bks))
        return n, round(hit/len(AR2EN),4)
    rows=[]
    for f in [0.0,0.25,0.5,0.75,1.0]:
        n,r=recall_at(f); rows.append({"glossary_fraction":f,"entries":n,"ar_to_en_recall_at_10":r})
    G._AR_EN.clear(); G._AR_EN.update(full)          # restore
    rep={"n_queries":len(AR2EN),"k":K,"glossary_full_size":len(full),"rows":rows,
         "note":"AR->EN cross-script book-level recall@10 vs fraction of the static glossary present (length-sorted prefix). 0% = no expansion (lexical cross-script floor)."}
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(rep,open(OUT,"w"),ensure_ascii=False,indent=2)
    print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
