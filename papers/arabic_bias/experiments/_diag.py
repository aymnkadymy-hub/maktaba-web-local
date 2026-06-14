import os,re,json
os.environ.setdefault("HF_HUB_OFFLINE","1");os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np
from exp_arabic_bias_v2 import load_expanded_arabic, clean, is_ar
AR,prov=load_expanded_arabic()
from sentence_transformers import CrossEncoder
ce=CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
head=lambda t,n=12:" ".join(t.split()[:n])
# per-book self-match std (cross-encoder) — is the wide Arabic band from clean live corpus or new books?
from collections import defaultdict
byb=defaultdict(list)
for b,c in AR: byb[b].append(c)
print(f"{'book':<28} {'n':>4} {'mean':>7} {'std':>6} {'min':>7}")
for b,cs in byb.items():
    sc=[float(x) for x in ce.predict([[head(c),c[:250]] for c in cs],show_progress_bar=False)]
    a=np.array(sc); print(f"{b:<28} {len(cs):>4} {a.mean():>7.3f} {a.std(ddof=1):>6.3f} {a.min():>7.3f}")
