#!/usr/bin/env python3
"""v3-CLEAN: the apples-to-apples matched-pipeline test the reviewers asked for.

The v2 confound was: new Arabic books were crude-chunked (pdftotext + a quick packer)
while English came from the production corpus. v3-clean removes the confound by chunking
the new Arabic books through the SAME production pipeline the corpus used:
    fitz page text  ->  smart_normalize (Arabic normalizer: strips tatweel/diacritics,
    unifies alef/ya, etc.)  ->  RecursiveCharacterTextSplitter(chunk_size=500,
    chunk_overlap=100, production separators).
English remains the production corpus chunks (already produced by that same pipeline),
so the comparison is now matched. Run with a venv that has fitz + langchain + ST:
    python exp_arabic_bias_v3_clean.py
Set MAKTABA_APP to the maktaba-web-local checkout (for bm25_cache/corpus.json);
place the Arabic source books in experiments/_ar_books_local/ (gitignored — copyrighted).
"""
import os, json, glob, importlib.util
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import numpy as np
from scipy import stats as sps

HERE=os.path.dirname(os.path.abspath(__file__))
APP=os.environ.get("MAKTABA_APP", os.path.expanduser("~/maktaba-web-local"))
CORPUS=os.path.join(APP,"bm25_cache","corpus.json")           # backup corpus (63 AR / 3760 EN)
SEED=0; np.random.seed(SEED)

# production Arabic normalizer (load the module file directly to avoid app-wide imports)
_spec=importlib.util.spec_from_file_location("arabic_normalizer",
    os.path.join(APP,"backend","utils","arabic_normalizer.py"))
_an=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_an)
smart_normalize=_an.smart_normalize
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
SPLITTER=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,
    separators=["\n\n","\n",".","؟","!","،","؛"," ",""])

def is_ar(s,th=0.25):
    a=sum(1 for c in s if '؀'<=c<='ۿ'); l=sum(1 for c in s if c.isalpha() or '؀'<=c<='ۿ')
    return l>0 and a/max(1,l)>th

def chunk_pdf_production(path):
    """Replicates backend/core/ingestion.py chunking for one PDF -> list[str]."""
    doc=fitz.open(path); pages=[]
    for i in range(len(doc)):
        try: text=smart_normalize(doc[i].get_text("text"))
        except Exception: continue
        if text and len(text)>20: pages.append(text)
    doc.close()
    chunks=[]
    for pg in pages:
        for part in SPLITTER.split_text(pg):
            if part.strip(): chunks.append(part.strip())
    return [c for c in chunks if len(c)>=60 and is_ar(c)]

def chunk_text_production(path):
    """For a text/markdown source: same normalize+splitter, no fitz."""
    raw=smart_normalize(open(path,encoding="utf8",errors="ignore").read())
    return [c.strip() for c in SPLITTER.split_text(raw) if len(c.strip())>=60 and is_ar(c)]

# new Arabic books — chunked through the PRODUCTION pipeline.
# Place the source books in experiments/_ar_books_local/ (gitignored — copyrighted; only
# the derived *_v3_clean.json stats are released). PDFs and .md/.txt are both picked up.
_BOOKS=os.path.join(HERE,"_ar_books_local")
AR_PDFS=sorted(glob.glob(os.path.join(_BOOKS,"*.pdf")))
AR_TXT=sorted(glob.glob(os.path.join(_BOOKS,"*.md"))+glob.glob(os.path.join(_BOOKS,"*.txt")))

def desc(a):
    a=np.asarray(a,float)
    return dict(n=int(a.size),mean=round(float(a.mean()),3),median=round(float(np.median(a)),3),
                std=round(float(a.std(ddof=1)),3),min=round(float(a.min()),3),max=round(float(a.max()),3))
def sd(x): return float(np.std(x,ddof=1))
def boot(en,ar,fn,B=10000):
    en=np.asarray(en,float); ar=np.asarray(ar,float); rng=np.random.default_rng(SEED); v=[]
    for _ in range(B): v.append(fn(rng.choice(en,en.size,True),rng.choice(ar,ar.size,True)))
    return round(float(np.median(v)),3),round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)
def tests(en,ar):
    en=np.asarray(en,float); ar=np.asarray(ar,float)
    lev=sps.levene(en,ar,center="median"); flk=sps.fligner(en,ar)
    welch=sps.ttest_ind(en,ar,equal_var=False); mwu=sps.mannwhitneyu(en,ar,alternative="two-sided")
    rm,rl,rh=boot(en,ar,lambda e,a:sd(e)/max(1e-9,sd(a)))
    dm,dl,dh=boot(en,ar,lambda e,a:float(e.mean()-a.mean()))
    return {"std_ratio_EN_over_AR":round(sd(en)/max(1e-9,sd(ar)),3),
        "bootstrap_std_ratio":{"median":rm,"ci95":[rl,rh],"arabic_narrower_sig":bool(rl>1.0),"arabic_wider_sig":bool(rh<1.0)},
        "levene_p":float(f"{lev.pvalue:.3e}"),"fligner_p":float(f"{flk.pvalue:.3e}"),
        "mean_gap_EN_minus_AR":round(float(en.mean()-ar.mean()),3),"bootstrap_mean_gap_ci":[dl,dh],
        "welch_p":float(f"{welch.pvalue:.3e}"),"arabic_lower_sig":bool(welch.pvalue<0.05 and en.mean()>ar.mean()),
        "mannwhitney_p":float(f"{mwu.pvalue:.3e}")}

def main():
    # Arabic: 63 production corpus + new books via production pipeline
    rows=json.load(open(CORPUS))
    AR=[("live_corpus", (x.get("c","")or"").strip()) for x in rows
        if len((x.get("c","")or"").strip())>=60 and is_ar((x.get("c","")or"").strip())]
    EN=[("english_corpus",(x.get("c","")or"").strip()) for x in rows
        if len((x.get("c","")or"").strip())>=60 and not is_ar((x.get("c","")or"").strip())]
    prov={"live_corpus":len(AR)}
    for p in AR_PDFS:
        if os.path.exists(p):
            cs=chunk_pdf_production(p); b=os.path.splitext(os.path.basename(p))[0]
            AR+=[(b,c) for c in cs]; prov[b]=len(cs)
    for p in AR_TXT:
        if os.path.exists(p):
            cs=chunk_text_production(p); b=os.path.splitext(os.path.basename(p))[0]
            AR+=[(b,c) for c in cs]; prov[b]=len(cs)
    print("CLEAN Arabic provenance (production-chunked):",prov,"total",len(AR))
    # balance English
    step=max(1,len(EN)//len(AR)); ENb=[EN[i] for i in range(0,len(EN),step)][:len(AR)]
    print("English balanced:",len(ENb))

    from sentence_transformers import CrossEncoder, SentenceTransformer
    ce=CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    mini=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    labse=SentenceTransformer("sentence-transformers/LaBSE")
    head=lambda t,n=12:" ".join(t.split()[:n])
    def ce_p(P):return [float(x) for x in ce.predict(P,show_progress_bar=False)]
    def cos(m,P):
        q=m.encode([a for a,_ in P],normalize_embeddings=True,show_progress_bar=False)
        p=m.encode([b for _,b in P],normalize_embeddings=True,show_progress_bar=False)
        return [float((q[i]*p[i]).sum()) for i in range(len(P))]
    def pairs(ch):
        bk=[b for b,_ in ch]; ip=[[head(c),c[:250]] for _,c in ch]; op=[]
        for i,(b,c) in enumerate(ch):
            j=(i+max(1,len(ch)//3))%len(ch); k=0
            while bk[j]==b and k<len(ch): j=(j+1)%len(ch); k+=1
            op.append([head(c),ch[j][1][:250]])
        return ip,op
    ai,_=pairs(AR); ei,_=pairs(ENb)
    R={"pipeline":"production (fitz + smart_normalize + RecursiveCharacterTextSplitter 500/100)",
       "provenance":prov,"n_arabic":len(AR),"n_english_balanced":len(ENb),"models":{}}
    for name,fn in [("cross_encoder",lambda P:ce_p(P)),("cosine_miniLM",lambda P:cos(mini,P)),("cosine_LaBSE",lambda P:cos(labse,P))]:
        a=fn(ai); e=fn(ei); R["models"][name]={"arabic":desc(a),"english":desc(e),"tests":tests(e,a)}
        T=R["models"][name]["tests"]
        print(f"[{name}] AR std {desc(a)['std']} EN std {desc(e)['std']} | EN/AR={T['std_ratio_EN_over_AR']} CI{T['bootstrap_std_ratio']['ci95']} "
              f"Lev p={T['levene_p']} Flig p={T['fligner_p']} | gap(EN-AR)={T['mean_gap_EN_minus_AR']} Welch p={T['welch_p']} ar_lower={T['arabic_lower_sig']}")
    json.dump(R,open(os.path.join(HERE,"results","exp_arabic_bias_v3_clean.json"),"w"),ensure_ascii=False,indent=1)
    print("saved v3_clean")
if __name__=="__main__": main()
