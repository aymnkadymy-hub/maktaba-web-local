#!/usr/bin/env python3
"""
Arabic algorithmic-bias measurement — v2 (EXPANDED + STATISTICALLY TESTED).

Improvements over v1 (addressing reviewer feedback):
  * EXPANDED Arabic sample: 4 additional real Arabic educational books are chunked and
    added to the 63 live-corpus Arabic chunks, taking the Arabic pool from 63 into the
    hundreds (no fabricated data; copyrighted texts kept local, only stats released).
  * BALANCED CONTROL: English is down-sampled to the SAME n as Arabic, so any width
    difference cannot be a small-Arabic-sample artifact.
  * MULTI-MODEL: the score-compression measurement is repeated on THREE frozen scorers
    (cross-encoder logit, paraphrase-multilingual-MiniLM cosine, LaBSE cosine) to show
    the effect is not specific to one model.
  * SIGNIFICANCE TESTS: Levene + Fligner-Killeen (equality-of-variance == the compression
    claim), Welch t-test and Mann-Whitney U (location == the "not lower" claim), and
    bootstrap 95% CIs on the std-ratio rho = sd_EN/sd_AR and the mean-gap d_mu.
  * QUERY-VARIANT ROBUSTNESS: the self-match probe is repeated with head-12, head-6, and
    mid-span pseudo-queries to show compression is not an artifact of one probe shape.

Deterministic. CPU. Offline (cached HF models). Reads the live corpus + the 4 local books.
"""
import os, re, json, statistics as st, math
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import numpy as np
from scipy import stats as sps

SEED=0
np.random.seed(SEED)
HERE=os.path.dirname(os.path.abspath(__file__))
CORPUS=os.path.expanduser("~/Desktop/maktaba-web-local/bm25_cache/corpus.json")
BOOKS=os.path.join(HERE,"_ar_books_local")
os.makedirs(os.path.join(HERE,"results"),exist_ok=True)

BIDI=dict.fromkeys(map(ord,"‪‫‬‭‮⁦⁧⁨⁩‎‏ـ"),None)
def clean(s): return re.sub(r"[ \t]+"," ",s.translate(BIDI)).strip()
def is_ar(s,th=0.25):
    ar=sum(1 for c in s if '؀'<=c<='ۿ'); let=sum(1 for c in s if c.isalpha() or '؀'<=c<='ۿ')
    return let>0 and ar/max(1,let)>th

def chunk_text(txt, target=400, lo=120, hi=620):
    txt=clean(txt.replace("\r"," "))
    # split on sentence-ish boundaries / newlines, then greedily pack to ~target chars
    parts=re.split(r"(?<=[\.\!\?؟،\n])\s+", txt)
    out=[]; buf=""
    for p in parts:
        p=p.strip()
        if not p: continue
        if len(buf)+len(p)+1<=hi:
            buf=(buf+" "+p).strip()
        else:
            if buf: out.append(buf); buf=p
            else: out.append(p[:hi]); buf=""
        if len(buf)>=target:
            out.append(buf); buf=""
    if buf: out.append(buf)
    return [c for c in out if lo<=len(c)<=hi and is_ar(c)]

def load_expanded_arabic():
    chunks=[]; prov={}
    # 1) the 4 new local books (each a distinct "book" for cross-book pairing)
    for fn in sorted(os.listdir(BOOKS)):
        if not fn.endswith(".txt"): continue
        book=fn[:-4]
        cs=chunk_text(open(os.path.join(BOOKS,fn),encoding="utf8",errors="ignore").read())
        for c in cs: chunks.append((book,c))
        prov[book]=len(cs)
    # 2) the live-corpus Arabic chunks (their own book bucket)
    rows=json.load(open(CORPUS)); n_live=0
    for x in rows:
        t=clean((x.get("c","")or""))
        if len(t)<60 or not is_ar(t): continue
        chunks.append(("live_corpus_arabic", t)); n_live+=1
    prov["live_corpus_arabic"]=n_live
    return chunks, prov

def load_english(limit=1200):
    rows=json.load(open(CORPUS)); en=[]
    for x in rows:
        t=clean((x.get("c","")or""))
        if len(t)<60 or is_ar(t): continue
        en.append(("english_corpus",t))
    step=max(1,len(en)//limit)
    return [en[i] for i in range(0,len(en),step)][:limit], len(en)

def desc(a):
    a=np.asarray(a,float)
    return dict(n=int(a.size),mean=round(float(a.mean()),3),median=round(float(np.median(a)),3),
                std=round(float(a.std(ddof=1)),3),min=round(float(a.min()),3),max=round(float(a.max()),3))

def boot_ratio(en,ar,fn,B=10000):
    en=np.asarray(en,float); ar=np.asarray(ar,float); rng=np.random.default_rng(SEED); vals=[]
    for _ in range(B):
        e=rng.choice(en,en.size,replace=True); a=rng.choice(ar,ar.size,replace=True)
        vals.append(fn(e,a))
    lo,hi=np.percentile(vals,[2.5,97.5])
    return round(float(np.median(vals)),3),round(float(lo),3),round(float(hi),3)

def sd(x): return float(np.std(x,ddof=1))
def compress_tests(en_in,ar_in):
    """all tests on the RELEVANT (self-match) bands. en/ar are score arrays."""
    en=np.asarray(en_in,float); ar=np.asarray(ar_in,float)
    lev=sps.levene(en,ar,center="median")          # Brown-Forsythe (robust equality of variance)
    flk=sps.fligner(en,ar)                          # Fligner-Killeen (non-parametric scale)
    welch=sps.ttest_ind(en,ar,equal_var=False)     # means (the "Arabic lower?" test)
    mwu=sps.mannwhitneyu(en,ar,alternative="two-sided")
    rho_med,rho_lo,rho_hi=boot_ratio(en,ar,lambda e,a: sd(e)/max(1e-9,sd(a)))
    dmu_med,dmu_lo,dmu_hi=boot_ratio(en,ar,lambda e,a: float(e.mean()-a.mean()))
    # Cohen's d for the mean gap
    nsd=math.sqrt(((en.size-1)*en.var(ddof=1)+(ar.size-1)*ar.var(ddof=1))/(en.size+ar.size-2))
    d=float((en.mean()-ar.mean())/max(1e-9,nsd))
    return {
      "std_ratio_EN_over_AR": round(sd(en)/max(1e-9,sd(ar)),3),
      "bootstrap_std_ratio": {"median":rho_med,"ci95":[rho_lo,rho_hi],"significant_compression":bool(rho_lo>1.0)},
      "levene_BF": {"W":round(float(lev.statistic),3),"p":float(f"{lev.pvalue:.3e}"),"variances_differ":bool(lev.pvalue<0.05)},
      "fligner_killeen": {"stat":round(float(flk.statistic),3),"p":float(f"{flk.pvalue:.3e}"),"scale_differs":bool(flk.pvalue<0.05)},
      "mean_gap_EN_minus_AR": round(float(en.mean()-ar.mean()),3),
      "bootstrap_mean_gap": {"median":dmu_med,"ci95":[dmu_lo,dmu_hi]},
      "welch_t_means": {"t":round(float(welch.statistic),3),"p":float(f"{welch.pvalue:.3e}"),"arabic_lower":bool(welch.pvalue<0.05 and en.mean()>ar.mean())},
      "mann_whitney_u": {"U":round(float(mwu.statistic),1),"p":float(f"{mwu.pvalue:.3e}")},
      "cohens_d_means": round(d,3),
    }

def main():
    AR,prov=load_expanded_arabic()
    EN,n_en_total=load_english()
    print(f"Arabic pool: {len(AR)} chunks  provenance={prov}")
    print(f"English pool: {len(EN)} sampled of {n_en_total}")

    from sentence_transformers import CrossEncoder, SentenceTransformer
    ce=CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    mini=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    labse=SentenceTransformer("sentence-transformers/LaBSE")

    def head(t,n): return " ".join(t.split()[:n])
    def mid(t,n=12):
        w=t.split(); s=max(0,(len(w)-n)//2); return " ".join(w[s:s+n])
    def ce_pred(pairs): return [float(x) for x in ce.predict(pairs,show_progress_bar=False)]
    def cos_pred(model,pairs):
        q=model.encode([a for a,_ in pairs],normalize_embeddings=True,show_progress_bar=False)
        p=model.encode([b for _,b in pairs],normalize_embeddings=True,show_progress_bar=False)
        return [float((q[i]*p[i]).sum()) for i in range(len(pairs))]

    # deterministic cross-book partner: next chunk from a DIFFERENT book bucket
    def make_pairs(chunks, qfn):
        books=[b for b,_ in chunks]
        inp=[[qfn(c), c[:250]] for _,c in chunks]
        outp=[]
        for i,(b,c) in enumerate(chunks):
            j=(i+ max(1,len(chunks)//3)) % len(chunks)
            k=0
            while books[j]==b and k<len(chunks): j=(j+1)%len(chunks); k+=1
            outp.append([qfn(c), chunks[j][1][:250]])
        return inp,outp

    # balance English to Arabic n (deterministic even sample)
    nbal=len(AR)
    step=max(1,len(EN)//nbal); EN_bal=[EN[i] for i in range(0,len(EN),step)][:nbal]
    print(f"balanced English: {len(EN_bal)} (matched to Arabic n={nbal})")

    R={"provenance":prov,"n_arabic":len(AR),"n_english_pool":len(EN),"n_english_balanced":len(EN_bal),"seed":SEED}

    # ---- primary: head-12 self-match, 3 models, balanced ----
    qfn=lambda t: head(t,12)
    ar_in_p,ar_out_p=make_pairs(AR,qfn)
    en_in_p,en_out_p=make_pairs(EN_bal,qfn)
    models={"cross_encoder":(ce_pred,None),"cosine_miniLM":(cos_pred,mini),"cosine_LaBSE":(cos_pred,labse)}
    R["models"]={}
    for name,(fn,mdl) in models.items():
        sc=(lambda P: fn(P)) if mdl is None else (lambda P: fn(mdl,P))
        ar_in=sc(ar_in_p); en_in=sc(en_in_p); ar_out=sc(ar_out_p); en_out=sc(en_out_p)
        R["models"][name]={
          "self_match_relevant":{"arabic":desc(ar_in),"english":desc(en_in)},
          "cross_doc_irrelevant":{"arabic":desc(ar_out),"english":desc(en_out)},
          "compression_tests":compress_tests(en_in,ar_in),
        }
        t=R["models"][name]["compression_tests"]
        print(f"\n[{name}]  sd_EN/sd_AR={t['std_ratio_EN_over_AR']}  boot95={t['bootstrap_std_ratio']['ci95']}"
              f"  Levene p={t['levene_BF']['p']}  Fligner p={t['fligner_killeen']['p']}")
        print(f"           mean_gap(EN-AR)={t['mean_gap_EN_minus_AR']}  Welch p={t['welch_t_means']['p']}  (arabic_lower={t['welch_t_means']['arabic_lower']})")

    # ---- query-variant robustness (cross-encoder, balanced) ----
    R["query_variants_cross_encoder"]={}
    for vname,vfn in [("head12",lambda t:head(t,12)),("head6",lambda t:head(t,6)),("mid12",lambda t:mid(t,12))]:
        ai,_=make_pairs(AR,vfn); ei,_=make_pairs(EN_bal,vfn)
        a=ce_pred(ai); e=ce_pred(ei)
        ct=compress_tests(e,a)
        R["query_variants_cross_encoder"][vname]={"arabic_std":desc(a)["std"],"english_std":desc(e)["std"],
            "std_ratio_EN_over_AR":ct["std_ratio_EN_over_AR"],"levene_p":ct["levene_BF"]["p"],
            "boot_ci95":ct["bootstrap_std_ratio"]["ci95"],"mean_gap_EN_minus_AR":ct["mean_gap_EN_minus_AR"]}
        print(f"  variant {vname}: sd_ratio={ct['std_ratio_EN_over_AR']} (CI {ct['bootstrap_std_ratio']['ci95']})")

    # ---- tokenization fertility (expanded Arabic vs English), reranker tokenizer ----
    tok=ce.tokenizer
    def fert(chunks):
        tw=sum(len(c.split()) for _,c in chunks); tt=sum(len(tok.tokenize(c)) for _,c in chunks); return tt/max(1,tw)
    fa=fert(AR); fe=fert(EN_bal)
    R["tokenization_fertility"]={"arabic_tokens_per_word":round(fa,3),"english_tokens_per_word":round(fe,3),"arabic_over_english":round(fa/fe,2)}
    print(f"\nfertility: AR {fa:.3f} vs EN {fe:.3f} = {fa/fe:.2f}x")

    out=os.path.join(HERE,"results","exp_arabic_bias_v2.json")
    json.dump(R,open(out,"w"),ensure_ascii=False,indent=1)
    print("\nsaved ->",out)

if __name__=="__main__":
    main()
