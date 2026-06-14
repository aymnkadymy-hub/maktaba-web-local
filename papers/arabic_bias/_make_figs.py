import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams.update({
    "font.family":"DejaVu Serif","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"figure.dpi":150,"savefig.bbox":"tight","axes.grid":False,
})
AR_C="#b2182b"; EN_C="#2166ac"; ACC="#1a9850"; GREY="#666666"
HERE=os.path.dirname(os.path.abspath(__file__))
FIG=os.path.join(HERE,"figures")
def J(p): return json.load(open(os.path.join(HERE,p)))
bias=J("experiments/results/exp_arabic_bias.json")
cut =J("../../paper_experiments/results/exp1_cutoffs.json")
star=J("../experiments/results/exp_p1_starvation.json")
dens=J("../experiments/results/exp_p1c_dense.json")

def save(fig,name):
    fig.savefig(os.path.join(FIG,name+".pdf")); fig.savefig(os.path.join(FIG,name+".png"),dpi=150); plt.close(fig)
    print("wrote",name)

# ---------- FIG 1: pipeline + where bias enters ----------
fig,ax=plt.subplots(figsize=(7.2,3.5)); ax.axis("off"); ax.set_xlim(0,15); ax.set_ylim(0,7)
def box(x,y,w,h,t,fc="#eef2f7",ec="#34495e"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8.6)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,lw=1.1,color="#34495e"))
box(0.2,3.0,1.7,1.0,"Query\n(AR or EN)",fc="#fff4e6")
box(2.3,3.0,1.7,1.0,"Normalize\n+ tokenize")
box(4.4,4.3,2.2,1.0,"Sparse BM25\n(per-tenant)")
box(4.4,1.7,2.2,1.0,"Dense FAISS\n(HNSW)")
box(7.0,3.0,1.6,1.0,"RRF\nmerge")
box(9.0,3.0,2.0,1.0,"Cross-encoder\nrerank")
box(11.4,3.0,1.9,1.0,"Per-tenant\nrelevance gate")
box(13.7,3.0,1.1,1.0,"Answer",fc="#e8f6ec")
arrow(1.9,3.5,2.3,3.5); arrow(4.0,3.7,4.4,4.7); arrow(4.0,3.3,4.4,2.3)
arrow(6.6,4.7,7.0,3.8); arrow(6.6,2.3,7.0,3.2); arrow(8.6,3.5,9.0,3.5)
arrow(11.0,3.5,11.4,3.5); arrow(13.3,3.5,13.7,3.5)
def tag(x,y,n,t):
    ax.add_patch(plt.Circle((x,y),0.22,fc=AR_C,ec="none",zorder=5))
    ax.text(x,y,n,ha="center",va="center",color="white",fontsize=8,fontweight="bold",zorder=6)
    ax.text(x,y-0.55,t,ha="center",va="top",fontsize=7.0,color=AR_C)
tag(3.15,2.7,"1","tokenization\nfragmentation 1.27×")
tag(5.5,1.5,"4","cross-script\nlexical zero")
tag(10.0,2.7,"2","score\ncompression ~2.6×")
tag(12.35,2.7,"3","mis-placed\nglobal cutoff")
ax.text(5.5,5.6,"⊛  minority-tenant starvation (per-tenant sub-index)  →  overlap@5 0.46→1.0",
        ha="center",fontsize=7.2,color=GREY,style="italic")
ax.set_title("Figure 1. The offline multilingual RAG pipeline and the four points where Arabic is disadvantaged",fontsize=9.3)
save(fig,"fig1_pipeline")

# ---------- FIG 2: score compression (headline) ----------
def band(ax,xpos,st,color,label):
    lo,hi=st["min"],st["max"]; m=st["mean"]; sd=st["std"]
    ax.plot([xpos,xpos],[lo,hi],color=color,lw=1.3,zorder=2)
    ax.add_patch(Rectangle((xpos-0.16,m-sd),0.32,2*sd,fc=color,ec="none",alpha=0.30,zorder=1))
    ax.plot([xpos-0.16,xpos+0.16],[m,m],color=color,lw=2.2,zorder=3)
    ax.plot([xpos-0.12,xpos+0.12],[st["median"],st["median"]],color=color,lw=1.0,ls="--",zorder=3)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.7))
ce=bias["cross_encoder_selfmatch"]; cos=bias["cosine_selfmatch"]
band(axs[0],1,ce["arabic"],AR_C,"Arabic"); band(axs[0],2,ce["english"],EN_C,"English")
axs[0].set_xticks([1,2]); axs[0].set_xticklabels(["Arabic\nstd 0.348","English\nstd 0.909"])
axs[0].set_title("Cross-encoder logit",fontsize=10); axs[0].set_ylabel("self-match (relevant) score")
band(axs[1],1,cos["arabic"],AR_C,"Arabic"); band(axs[1],2,cos["english"],EN_C,"English")
axs[1].set_xticks([1,2]); axs[1].set_xticklabels(["Arabic\nstd 0.062","English\nstd 0.128"])
axs[1].set_title("Cosine similarity",fontsize=10)
for a in axs: a.set_xlim(0.4,2.6); a.grid(axis="y",ls=":",alpha=0.4)
axs[0].text(1.5,axs[0].get_ylim()[0]+0.1,"EN band ≈2.6× wider",ha="center",va="bottom",fontsize=7.6,color=GREY)
axs[1].text(1.5,axs[1].get_ylim()[0]+0.01,"EN band ≈2.1× wider",ha="center",va="bottom",fontsize=7.6,color=GREY)
fig.subplots_adjust(wspace=0.28)
fig.suptitle("Figure 2. Score compression: Arabic relevant scores occupy a much narrower band — not a lower one\n"
             "(box = mean ± SD, whisker = min–max, dashed = median; Arabic mean is equal-or-higher)",fontsize=9.0,y=1.03)
save(fig,"fig2_compression")

# ---------- FIG 3: separation + mis-placed global cutoff (cross-encoder) ----------
fig,ax=plt.subplots(figsize=(7.2,3.4))
ce_in=bias["cross_encoder_selfmatch"]; ce_out=bias["cross_encoder_crossdoc"]
lanes=[("English",2.0,ce_in["english"],ce_out["english"],EN_C),
       ("Arabic", 0.0,ce_in["arabic"], ce_out["arabic"], AR_C)]
for name,y,inb,outb,c in lanes:
    ax.add_patch(Rectangle((outb["mean"]-outb["std"],y-0.22),2*outb["std"],0.44,fc=GREY,alpha=0.35,ec="none"))
    ax.add_patch(Rectangle((inb["mean"]-inb["std"], y-0.22),2*inb["std"], 0.44,fc=c,alpha=0.50,ec="none"))
    ax.text(outb["mean"],y+0.40,"irrelevant band",ha="center",fontsize=7.2,color=GREY)
    ax.text(inb["mean"], y+0.40,"relevant band", ha="center",fontsize=7.2,color=c)
    ax.text(-11.4,y,name,ha="right",va="center",fontsize=9.5,color=c,fontweight="bold")
# single global default line
ax.axvline(-5.0,color="black",ls="--",lw=1.4)
ax.text(-6.6,3.15,"global default −5.0\n(one line for everyone)",ha="center",fontsize=7.4)
# per-tenant cutoff RANGE as a shaded band (declutters individual labels)
ax.add_patch(Rectangle((-3.29,-1.1),(-1.39)-(-3.29),4.2,fc=ACC,alpha=0.12,ec="none"))
ax.axvline(-3.29,color=EN_C,lw=1.0,alpha=0.7); ax.axvline(-1.39,color=AR_C,lw=1.4,alpha=0.9)
ax.annotate("Arabic tenant\nstrictest (−1.39)",xy=(-1.39,-0.9),xytext=(2.0,-0.9),fontsize=7.2,color=AR_C,
            va="center",arrowprops=dict(arrowstyle="->",color=AR_C,lw=0.9))
ax.text(-3.29,-1.35,"English (−3.29)",ha="center",fontsize=7.0,color=EN_C)
ax.text(-2.34,2.2,"per-tenant\ngate cutoffs",ha="center",fontsize=7.0,color="#127a40")
ax.set_xlim(-11.6,12); ax.set_ylim(-1.7,3.6); ax.set_yticks([])
ax.set_xlabel("cross-encoder relevance logit"); ax.grid(axis="x",ls=":",alpha=0.3)
ax.set_title("Figure 3. One global cutoff cannot fit both: Arabic's relevant and irrelevant bands sit closer together,\nso the gate places a stricter, geometry-matched cutoff per tenant (Arabic tenants strictest)",fontsize=8.7)
save(fig,"fig3_cutoff")

# ---------- FIG 4: per-tenant calibrated cutoffs ----------
ten=["T1\nEN ML","T2\nEN econ","T3\nAR+math","T4\nAR+EN","T5\nEN STEM"]
ce_t=[-2.93,-2.75,-1.59,-1.39,-3.29]; cos_t=[0.382,0.329,0.409,0.302,0.325]
isar=[False,False,True,True,False]
fig,axs=plt.subplots(1,2,figsize=(7.4,4.0))
cols=[AR_C if a else EN_C for a in isar]
# cross-encoder (negative cutoffs; bars hang down from 0; strictest = least negative)
axs[0].bar(range(5),ce_t,color=cols,edgecolor="black",lw=0.5)
axs[0].set_ylim(-4.0,0.5); axs[0].set_title("Cross-encoder cutoff  τ_t",fontsize=9.5,pad=8); axs[0].set_ylabel("cutoff (logit)")
for i,v in enumerate(ce_t): axs[0].text(i,v-0.12,f"{v:g}",ha="center",va="top",fontsize=7.2)
# cosine
axs[1].bar(range(5),cos_t,color=cols,edgecolor="black",lw=0.5)
axs[1].set_ylim(0,0.50); axs[1].set_title("Cosine cutoff  τ_t",fontsize=9.5,pad=8); axs[1].set_ylabel("cutoff (cosine)")
for i,v in enumerate(cos_t): axs[1].text(i,v+0.008,f"{v:g}",ha="center",va="bottom",fontsize=7.2)
for ax in axs:
    ax.set_xticks(range(5)); ax.set_xticklabels(ten,fontsize=7.4); ax.grid(axis="y",ls=":",alpha=0.4)
fig.legend(handles=[plt.Rectangle((0,0),1,1,fc=AR_C),plt.Rectangle((0,0),1,1,fc=EN_C)],
           labels=["Arabic-containing tenant","English tenant"],loc="lower center",ncol=2,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Figure 4. Per-tenant calibrated cutoffs — the gate (label-free) makes Arabic-containing tenants the strictest\n(least-negative cross-encoder cutoff; highest cosine cutoff)",fontsize=8.8,y=1.0)
fig.subplots_adjust(bottom=0.20,top=0.84,wspace=0.28)
save(fig,"fig4_pertenant")

# ---------- FIG 5: gate performance ----------
fig,ax=plt.subplots(figsize=(7.2,3.2))
metrics=["Precision","Recall","F1"]; x=np.arange(3); w=0.27
fixed=[0.706,1.000,0.828]; gate=[0.938,1.000,0.968]; orac=[1.0,1.0,1.0]
ax.bar(x-w,fixed,w,label="fixed default −5.0",color=GREY,edgecolor="black",lw=0.5)
ax.bar(x,  gate, w,label="self-calibrated gate (no labels)",color=ACC,edgecolor="black",lw=0.5)
ax.bar(x+w,orac, w,label="label-tuned global oracle",color="#999999",edgecolor="black",lw=0.5,hatch="//")
for i,(a,b,c) in enumerate(zip(fixed,gate,orac)):
    for dx,v in [(-w,a),(0,b),(w,c)]: ax.text(i+dx,v+0.012,f"{v:.3f}",ha="center",fontsize=6.6)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0,1.12); ax.set_ylabel("score")
ax.legend(fontsize=7.4,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.30))
ax.grid(axis="y",ls=":",alpha=0.4)
ax.set_title("Figure 5. The label-free gate lifts precision 0.71→0.94 (F1 0.83→0.97), recall pinned at 1.0;\noff-topic acceptances fall 25→4 — near the oracle it never sees, without beating it",fontsize=8.8)
save(fig,"fig5_gate")

# ---------- FIG 6: cross-script recall ----------
fig,ax=plt.subplots(figsize=(6.6,3.0))
labels=["BM25\n(no glossary)","BM25 + glossary\n(ours, ~48 µs)","Dense MiniLM\n(~16 ms)","Dense LaBSE\n(~22 ms)"]
vals=[0.0,1.0,1.0,1.0]; cols=[GREY,ACC,EN_C,"#7fa8d0"]
b=ax.bar(range(4),vals,color=cols,edgecolor="black",lw=0.5)
for i,v in enumerate(vals): ax.text(i,v+0.03,f"{v:.2f}",ha="center",fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(labels,fontsize=7.8); ax.set_ylim(0,1.18)
ax.set_ylabel("Arabic→English book-level recall@10"); ax.grid(axis="y",ls=":",alpha=0.4)
ax.set_title("Figure 6. Cross-script recall: the glossary lifts the structural lexical floor 0.00→1.00 —\nbut a dense encoder already reaches 1.00, so the glossary's net win is cost, not recall (EN→AR is 1.00 throughout)",fontsize=8.4)
save(fig,"fig6_crossscript")

# ---------- FIG 7: tenant starvation ----------
fig,ax=plt.subplots(figsize=(7.0,3.4))
sv=star["expA_recall_vs_dominance"]["by_probeset"]["shared_common_vocab"]
dv=star["expA_recall_vs_dominance"]["by_probeset"]["distinct_vocab"]
dn=dens["rows"]
dom=[r["dominance"] for r in sv]
ax.plot(dom,[r["oracle_overlap_at_k"] for r in sv],"-o",color=AR_C,lw=1.6,ms=4,label="sparse, shared vocab (minority, often AR)")
ax.plot([r["dominance"] for r in dn],[r["dense_postfilter8x_overlap_at_k"] for r in dn],"-s",color="#7b3294",lw=1.4,ms=4,label="dense FAISS post-filter (starves worse)")
ax.plot(dom,[r["oracle_overlap_at_k"] for r in dv],"-^",color=GREY,lw=1.2,ms=4,label="distinct-vocab control (no starvation)")
ax.axhline(1.0,color=ACC,lw=1.6,ls="--",label="per-tenant sub-index (restored)")
ax.axvline(0.9847,color="black",lw=0.8,ls=":"); ax.text(0.9847,0.13,"real skew\n98.5%",fontsize=6.6,ha="right")
ax.set_xlabel("dominant-tenant share of the shared index"); ax.set_ylabel("oracle-overlap@5 (minority tenant)")
ax.set_ylim(0,1.08); ax.legend(fontsize=7.2,loc="lower left",frameon=True)
ax.grid(ls=":",alpha=0.4)
ax.set_title("Figure 7. Minority-tenant starvation: sparse overlap collapses 0.91→0.46 as one tenant dominates;\ndense starves worse (→0.32); a per-tenant sub-index restores exact retrieval (1.0)",fontsize=8.6)
save(fig,"fig7_starvation")

# ---------- FIG 8: tokenization fertility ----------
fig,ax=plt.subplots(figsize=(4.6,3.0))
f=bias["tokenization_fertility"]
ax.bar([0,1],[f["english_tokens_per_word"],f["arabic_tokens_per_word"]],color=[EN_C,AR_C],edgecolor="black",lw=0.5,width=0.55)
ax.set_xticks([0,1]); ax.set_xticklabels(["English\n1.536","Arabic\n1.946"])
ax.text(0.5,1.97,"1.27× more\nfragmented",ha="center",fontsize=8,color=AR_C)
ax.set_ylabel("subword tokens per word"); ax.set_ylim(0,2.3); ax.grid(axis="y",ls=":",alpha=0.4)
ax.set_title("Figure 8. Tokenization fertility\n(reranker's own tokenizer, our corpus)",fontsize=9)
save(fig,"fig8_fertility")
print("ALL FIGURES DONE")
