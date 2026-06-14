#!/usr/bin/env python3
"""
Paper 2 — dense multilingual-encoder baselines for cross-script retrieval, to
answer "why not just use a multilingual encoder?". On the SAME real corpus and
the SAME labelled cross-script query set, we measure book-level recall@10 for:
  - the SMALL deployed on-device encoder: paraphrase-multilingual-MiniLM-L12-v2 (~118M)
  - a HEAVY multilingual encoder:        LaBSE (~471M, ~1.8GB)
and report per-query embedding latency + model size, so the glossary (BM25 + 48µs,
no model) can be positioned on the cost/accuracy frontier. Deterministic (eval mode).
Compare to the BM25 numbers in exp_p2_glossary.json (no-glossary 0.0 AR->EN; +glossary 1.0).
"""
import json, os, time, statistics
import numpy as np
CORP=os.environ.get("MAKTABA_CORPUS", os.path.expanduser("~/maktaba-web-local/bm25_cache/corpus.json"))
OUT=os.path.join(os.path.dirname(__file__),"results","exp_p2c_labse.json")
TENANT="3CLDpsvv7RhtClg5iqhr-g"; K=10
AR2EN=[("الشبكة العصبية التوليدية","generative-deep-learning"),("فرط التخصيص في الشبكة العصبية","generative-deep-learning"),
 ("خوارزمية البحث","AI_Search_Algorithms"),("الحوافز الاقتصادية","Freakonomics"),
 ("النشر العلمي ومعامل التأثير","100 Q&A About Scientific Publishing"),("التحكيم والسرقة الادبية","100 Q&A About Scientific Publishing")]
EN2AR=[("partial fractions","الكسور الجزئيه"),("programming variable and function","اساسيات البرمجة"),
 ("loop and array in programming","اساسيات البرمجة")]
def main():
    from sentence_transformers import SentenceTransformer
    import faiss
    data=json.load(open(CORP))
    A=[{"c":e.get("c",""),"m":(e.get("m",{}).get("metadata",e.get("m",{})) if isinstance(e.get("m"),dict) else {})} for e in data]
    A=[d for d in A if d["m"].get("user_id")==TENANT and (d["c"] or "").strip()]
    titles=[d["m"].get("book_title","") for d in A]
    rep={"corpus":{"tenant":TENANT,"n_chunks":len(A)},"k":K,
         "bm25_reference":{"AR_to_EN_no_glossary":0.0,"AR_to_EN_with_glossary":1.0,"source":"exp_p2_glossary.json"},
         "models":{}}
    MODELS=[("paraphrase-multilingual-MiniLM-L12-v2","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2","small (~118M, deployed on-device encoder)"),
            ("LaBSE","sentence-transformers/LaBSE","heavy (~471M, ~1.8GB)")]
    for short,name,desc in MODELS:
        print(f"loading {short}...")
        model=SentenceTransformer(name, device="cpu")
        t0=time.perf_counter()
        cvec=np.asarray(model.encode([d["c"] for d in A], normalize_embeddings=True, batch_size=64, show_progress_bar=False),dtype="float32")
        embed_corpus_s=time.perf_counter()-t0
        idx=faiss.IndexFlatIP(cvec.shape[1]); idx.add(cvec)
        def recall(qs):
            hit=0; lat=[]
            for q,tgt in qs:
                t1=time.perf_counter(); qv=np.asarray(model.encode([q],normalize_embeddings=True),dtype="float32"); lat.append((time.perf_counter()-t1)*1000)
                _,I=idx.search(qv,min(K,len(A)))
                bks=[titles[int(j)] for j in I[0] if j>=0]
                hit+=int(any(tgt.lower() in (b or "").lower() for b in bks))
            return round(hit/len(qs),4), round(statistics.median(lat),2)
        ar,ar_lat=recall(AR2EN); en,en_lat=recall(EN2AR)
        rep["models"][short]={"desc":desc,"dim":int(cvec.shape[1]),
            "AR_to_EN_dense_recall_at_10":ar,"EN_to_AR_dense_recall_at_10":en,
            "query_embed_ms_median":round((ar_lat+en_lat)/2,2),
            "corpus_embed_seconds":round(embed_corpus_s,1)}
        print(f"  {short}: AR->EN dense={ar}, EN->AR dense={en}, q-embed {(ar_lat+en_lat)/2:.1f}ms, corpus-embed {embed_corpus_s:.0f}s")
    rep["note"]=("Dense multilingual encoders cross scripts via a shared embedding space. The comparison is "
        "cost/accuracy, not a SOTA race: BM25+glossary reaches AR->EN 1.0 at ~48us/query with no model, "
        "complementing the small deployed encoder; LaBSE is a 1.8GB model. We report whatever recall each achieves.")
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(rep,open(OUT,"w"),ensure_ascii=False,indent=2)
    print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
