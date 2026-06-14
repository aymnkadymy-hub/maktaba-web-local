#!/usr/bin/env python3
"""Figures for the EXPANDED, multi-model, statistically-tested results (exp_arabic_bias_v2.json)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"figure.dpi":150,"savefig.bbox":"tight"})
AR_C="#b2182b"; EN_C="#2166ac"; ACC="#1a9850"; GREY="#666666"
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
R=json.load(open(os.path.join(HERE,"experiments/results/exp_arabic_bias_v2.json")))
def save(fig,n): fig.savefig(os.path.join(FIG,n+".pdf")); fig.savefig(os.path.join(FIG,n+".png"),dpi=150); plt.close(fig); print("wrote",n)
def stars(p):
    return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<5e-2 else "n.s."

MODELS=[("cross_encoder","Cross-encoder\nlogit"),("cosine_miniLM","MiniLM\ncosine"),("cosine_LaBSE","LaBSE\ncosine")]
nA=R["n_arabic"]; nB=R["n_english_balanced"]

# ---- FIG 9: multi-model self-match SD (AR vs EN) + ratio + significance ----
fig,ax=plt.subplots(figsize=(7.2,4.2))
x=np.arange(len(MODELS)); w=0.36
arsd=[R["models"][k]["self_match_relevant"]["arabic"]["std"] for k,_ in MODELS]
ensd=[R["models"][k]["self_match_relevant"]["english"]["std"] for k,_ in MODELS]
# normalize each model to its own English SD so the 3 very different scales are comparable
arn=[a/e for a,e in zip(arsd,ensd)]; enn=[1.0]*len(MODELS)
b1=ax.bar(x-w/2,enn,w,color=EN_C,edgecolor="black",lw=.5,label=f"English (balanced n={nB})")
b2=ax.bar(x+w/2,arn,w,color=AR_C,edgecolor="black",lw=.5,label=f"Arabic (n={nA})")
for i,(k,_) in enumerate(MODELS):
    t=R["models"][k]["compression_tests"]; rho=t["std_ratio_EN_over_AR"]; p=t["levene_BF"]["p"]
    ax.text(i,max(1.0,arn[i])+0.05,f"EN/AR = {rho}×\n{stars(p)}",ha="center",fontsize=7.6,color="#222")
    ax.text(i-w/2,enn[i]+0.01,f"{ensd[i]:g}",ha="center",fontsize=6.5,va="bottom",color="white")
    ax.text(i+w/2,arn[i]+0.01,f"{arsd[i]:g}",ha="center",fontsize=6.5,va="bottom")
ax.set_xticks(x); ax.set_xticklabels([m[1] for m in MODELS]); ax.set_ylim(0,1.85)
ax.set_ylabel("relevant-band SD\n(normalized to English = 1.0)")
ax.legend(fontsize=7.6,loc="upper right",frameon=True,framealpha=0.9); ax.grid(axis="y",ls=":",alpha=.4)
ax.set_title("Figure 9. The relevant-band spread differs SIGNIFICANTLY by language on all three frozen scorers (Levene shown;\n"
             "*** p<0.001) — but the DIRECTION is model-dependent: Arabic narrower on LaBSE, wider on cross-encoder/MiniLM\n"
             "(expanded mixed-quality sample; superseded by the clean test, Fig 12). Raw SDs labelled on bars.",fontsize=7.5,pad=10)
fig.subplots_adjust(top=0.80)
save(fig,"fig9_multimodel")

# ---- FIG 10: bootstrap 95% CI forest plot on sd_EN/sd_AR (models + query variants) ----
fig,ax=plt.subplots(figsize=(7.2,4.2))
rows=[]
for k,lab in MODELS:
    t=R["models"][k]["compression_tests"]["bootstrap_std_ratio"]
    rows.append((lab.replace("\n"," "),t["median"],t["ci95"][0],t["ci95"][1],AR_C))
for v,vd in R["query_variants_cross_encoder"].items():
    rows.append((f"CE query={v}",vd["std_ratio_EN_over_AR"],vd["boot_ci95"][0],vd["boot_ci95"][1],GREY))
rows=rows[::-1]
y=np.arange(len(rows))
for i,(lab,m,lo,hi,c) in enumerate(rows):
    ax.plot([lo,hi],[i,i],color=c,lw=2.0); ax.plot(m,i,"o",color=c,ms=6)
    ax.text(hi+0.05,i,f"{m:g} [{lo:g}, {hi:g}]",va="center",fontsize=7.2,color="#222")
ax.axvline(1.0,color="black",ls="--",lw=1.2)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=8)
ax.set_ylim(-0.6,len(rows)-0.4)
ax.set_xlabel("bootstrap 95% CI of  sd(English) / sd(Arabic)   [10,000 resamples];   dashed = parity")
ax.set_xlim(0.3,max(2.0,max(r[3] for r in rows)+0.85)); ax.grid(axis="x",ls=":",alpha=.4)
ax.set_title("Figure 10. Expanded sample: CIs exclude parity on every model (spread differs significantly), but the\nDIRECTION is model-dependent — ratio >1 = Arabic narrower (LaBSE); <1 = Arabic wider (cross-encoder/MiniLM).\nNOTE: superseded by the clean matched-pipeline test (Fig 12); here Arabic books were crude-chunked.",fontsize=7.8,pad=10)
fig.subplots_adjust(top=0.82,left=0.20)
save(fig,"fig10_bootstrap_ci")

# ---- FIG 11: per-Arabic-book diagnostic (the extraction-quality confound), cross-encoder ----
# measured per-book cross-encoder self-match std (diagnostic run)
books=[("digital_logic\n(clean)",0.308,"clean"),("live_corpus\n(clean,v1)",0.347,"clean"),
       ("powerpoint\n(clean)",0.529,"clean"),("computer_skills\n(noisy PDF)",1.097,"noisy"),
       ("curriculum\n(noisy PDF)",1.297,"noisy")]
EN_REF=0.623
fig,ax=plt.subplots(figsize=(7.2,4.0))
xs=np.arange(len(books))
cols=[ACC if t=="clean" else AR_C for _,_,t in books]
ax.bar(xs,[v for _,v,_ in books],color=cols,edgecolor="black",lw=.5)
for i,(_,v,_) in enumerate(books): ax.text(i,v+0.02,f"{v:g}",ha="center",fontsize=8)
ax.axhline(EN_REF,color=EN_C,lw=1.6,ls="--"); ax.text(len(books)-0.5,EN_REF+0.03,f"English (balanced) = {EN_REF}",color=EN_C,fontsize=7.5,ha="right")
ax.set_xticks(xs); ax.set_xticklabels([b for b,_,_ in books],fontsize=7.4)
ax.set_ylabel("cross-encoder self-match SD"); ax.set_ylim(0,1.75); ax.grid(axis="y",ls=":",alpha=.4)
ax.legend(handles=[plt.Rectangle((0,0),1,1,fc=ACC),plt.Rectangle((0,0),1,1,fc=AR_C)],
          labels=["cleanly-extracted Arabic (tighter than EN)","noisy raw-PDF extraction (inflates SD)"],
          fontsize=7.2,loc="upper left",frameon=True,framealpha=0.9)
ax.set_title("Figure 11. Why the magnitude is fragile: cleanly-chunked Arabic books are tighter than English, but two\n"
             "noisily-extracted raw-PDF books inflate the band — a chunk-quality confound, not a language reversal.",fontsize=7.8,pad=10)
fig.subplots_adjust(top=0.83)
save(fig,"fig11_perbook")
print("V2 FIGURES DONE")
