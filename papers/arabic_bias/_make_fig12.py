#!/usr/bin/env python3
"""Fig 12: the definitive clean matched-pipeline test (v3) — the compression DIRECTION is unstable."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10,"figure.dpi":150,"savefig.bbox":"tight"})
AR_C="#b2182b"; EN_C="#2166ac"; GREY="#666666"
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
v3=json.load(open(os.path.join(HERE,"experiments/results/exp_arabic_bias_v3_clean.json")))

fig,axs=plt.subplots(1,2,figsize=(7.4,3.5))

# Panel A: the cross-encoder Arabic band FLIPS from narrower (v1, small homogeneous) to wider (v3, large diverse)
ax=axs[0]
labels=["v1: n=63\n(1 homogeneous book)","v3-clean: n=390\n(5 diverse books)"]
ar=[0.348,1.109]; en=[0.909,0.623]
x=np.arange(2); w=0.36
ax.bar(x-w/2,ar,w,color=AR_C,edgecolor="black",lw=.5,label="Arabic")
ax.bar(x+w/2,en,w,color=EN_C,edgecolor="black",lw=.5,label="English")
for i,(a,e) in enumerate(zip(ar,en)):
    ax.text(i-w/2,a+0.02,f"{a}",ha="center",fontsize=7.5); ax.text(i+w/2,e+0.02,f"{e}",ha="center",fontsize=7.5)
ax.text(0,0.62,"Arabic\nNARROWER",ha="center",fontsize=6.6,color=AR_C)
ax.text(1,1.30,"Arabic WIDER\n(compression refuted)",ha="center",fontsize=6.6,color=AR_C)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=7.4); ax.set_ylim(0,1.75)
ax.set_ylabel("cross-encoder self-match SD")
ax.legend(fontsize=7.2,frameon=False,loc="upper center",ncol=2,bbox_to_anchor=(0.5,1.02))
ax.set_title("(a) The direction flips with corpus size/diversity",fontsize=8.6,pad=14); ax.grid(axis="y",ls=":",alpha=.4)

# Panel B: v3-clean 3-model bootstrap CI forest (definitive, matched pipeline)
ax=axs[1]
M=[("Cross-encoder","cross_encoder"),("MiniLM","cosine_miniLM"),("LaBSE","cosine_LaBSE")]
rows=[]
for lab,k in M:
    t=v3["models"][k]["tests"]["bootstrap_std_ratio"]; rows.append((lab,t["median"],t["ci95"][0],t["ci95"][1]))
rows=rows[::-1]
for i,(lab,m,lo,hi) in enumerate(rows):
    c=AR_C if hi<1.0 else (GREY if (lo<1.0<hi) else EN_C)
    ax.plot([lo,hi],[i,i],color=c,lw=2.2); ax.plot(m,i,"o",color=c,ms=6)
    ax.text(hi+0.04,i,f"{m:g} [{lo:g},{hi:g}]",va="center",fontsize=7.2)
ax.axvline(1.0,color="black",ls="--",lw=1.2); ax.text(1.0,2.42,"parity",fontsize=7,ha="center",color="#333")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows],fontsize=8)
ax.set_ylim(-0.7,2.6); ax.set_xlim(0.3,1.75)
ax.set_xlabel("sd(EN)/sd(AR), bootstrap 95% CI\n<1 = Arabic WIDER;  ≈1 = no difference",fontsize=8)
ax.set_title("(b) Clean matched-pipeline (n=390): no compression",fontsize=8.6); ax.grid(axis="x",ls=":",alpha=.4)

fig.suptitle("Figure 12. The definitive test: on a clean, matched-pipeline, larger sample the \"Arabic compression\" direction does NOT hold —\n"
             "Arabic is significantly WIDER on the cross-encoder/MiniLM and indistinguishable on LaBSE. The geometry gap is real but its\n"
             "direction is unstable (corpus-, genre-, and model-dependent) — which is exactly why the cutoff must be calibrated per tenant.",fontsize=7.5,y=1.02)
fig.subplots_adjust(top=0.80,wspace=0.32,bottom=0.20)
fig.savefig(os.path.join(FIG,"fig12_definitive.pdf")); fig.savefig(os.path.join(FIG,"fig12_definitive.png"),dpi=150)
print("wrote fig12_definitive")
