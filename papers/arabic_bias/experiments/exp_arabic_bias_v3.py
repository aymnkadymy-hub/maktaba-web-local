#!/usr/bin/env python3
"""v3: same as v2 but with ONE objective, language-neutral prose-quality filter applied
IDENTICALLY to Arabic and English, to remove the crude-PDF-extraction confound that
inflated some Arabic books' variance in v2. Reports per-book clean counts + 3-model stats."""
import os,json
os.environ.setdefault("HF_HUB_OFFLINE","1");os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import numpy as np
import exp_arabic_bias_v2 as V2
from exp_arabic_bias_v2 import desc, compress_tests, clean, is_ar

def quality_ok(c):
    L=len(c); 
    if L<150 or L>640: return False
    letters=sum(1 for ch in c if ch.isalpha() or '؀'<=ch<='ۿ')
    if letters/L < 0.62: return False                      # extraction noise / punctuation-heavy
    digits=sum(1 for ch in c if ch.isdigit())
    if digits/L > 0.18: return False
    w=c.split()
    if not (18<=len(w)<=120): return False
    if max(len(x) for x in w) > 28: return False           # merged-garbage token
    return True

def main():
    AR,prov=V2.load_expanded_arabic()
    AR=[(b,c) for b,c in AR if quality_ok(c)]
    from collections import Counter
    provc=Counter(b for b,_ in AR)
    EN,ntot=V2.load_english(limit=2000)
    EN=[(b,c) for b,c in EN if quality_ok(c)]
    print("clean Arabic per book:",dict(provc),"total",len(AR))
    print("clean English:",len(EN))

    from sentence_transformers import CrossEncoder, SentenceTransformer
    ce=CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    mini=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    labse=SentenceTransformer("sentence-transformers/LaBSE")
    head=lambda t,n=12:" ".join(t.split()[:n])
    def ce_pred(P):return [float(x) for x in ce.predict(P,show_progress_bar=False)]
    def cos(m,P):
        q=m.encode([a for a,_ in P],normalize_embeddings=True,show_progress_bar=False)
        p=m.encode([b for _,b in P],normalize_embeddings=True,show_progress_bar=False)
        return [float((q[i]*p[i]).sum()) for i in range(len(P))]
    def pairs(ch):
        bk=[b for b,_ in ch]; inp=[[head(c),c[:250]] for _,c in ch]; out=[]
        for i,(b,c) in enumerate(ch):
            j=(i+max(1,len(ch)//3))%len(ch); k=0
            while bk[j]==b and k<len(ch): j=(j+1)%len(ch); k+=1
            out.append([head(c),ch[j][1][:250]])
        return inp,out
    n=len(AR); step=max(1,len(EN)//n); ENb=[EN[i] for i in range(0,len(EN),step)][:n]
    ai,_=pairs(AR); ei,_=pairs(ENb)
    R={"clean_provenance":dict(provc),"n_arabic":len(AR),"n_english_balanced":len(ENb),"filter":"len150-640,letterfrac>0.62,digits<0.18,18-120w,maxword<=28","models":{}}
    for name,fn in [("cross_encoder",lambda P:ce_pred(P)),("cosine_miniLM",lambda P:cos(mini,P)),("cosine_LaBSE",lambda P:cos(labse,P))]:
        a=fn(ai); e=fn(ei)
        R["models"][name]={"arabic":desc(a),"english":desc(e),"tests":compress_tests(e,a)}
        t=R["models"][name]["tests"]
        print(f"[{name}] AR std {desc(a)['std']} vs EN std {desc(e)['std']} | EN/AR={t['std_ratio_EN_over_AR']} CI{t['bootstrap_std_ratio']['ci95']} "
              f"Levene p={t['levene_BF']['p']} | meangap(EN-AR)={t['mean_gap_EN_minus_AR']} Welch p={t['welch_t_means']['p']} ar_lower={t['welch_t_means']['arabic_lower']}")
    json.dump(R,open(os.path.join(V2.HERE,"results","exp_arabic_bias_v3.json"),"w"),ensure_ascii=False,indent=1)
    print("saved v3")
if __name__=="__main__": main()
