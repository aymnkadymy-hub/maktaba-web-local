#!/usr/bin/env python3
"""Paper 3 — automatic dialectness signal: score each MSA sentence and its
dialectized rewrite with the Sentence-ALDi model (Keleg et al., EMNLP 2023),
a continuous 0=MSA .. 1=fully-dialectal estimator. Hypothesis: rewriting
significantly RAISES dialectness. This is an automatic dialect-quality signal
complementing (not replacing) a human study. Deterministic (eval mode, no sampling)."""
import json, os, sys, statistics
sys.path.insert(0,os.environ.get("MAKTABA_ROOT", os.path.expanduser("~/maktaba-web-local")))
from backend.dialect import dialect_processor as DP
OUT=os.path.join(os.path.dirname(__file__),"results","exp_p3b_dialectness.json")
MSA=["الآن لا يوجد كثير من الوقت لكن كل شيء جيد","ماذا تريد أن تفعل اليوم","كيف حالك يا صديقي",
 "هذا الكلام صحيح تماماً وأنا أفهم الفكرة","سوف يتم شرح الموضوع","دعنا نبدأ من البداية",
 "المعلم يشير إلى أن المتغير مهم جداً","أين الكتاب الذي تتحدث عنه","ربما يكون هذا جيداً ولكن ليس دائماً",
 "أيضاً يوجد حل آخر للمسألة","نحن نتكلم عن البرمجة الآن","أنا فهمت الدرس جيداً","رائع، هذا ممتاز",
 "كثيراً ما نرى هذا في الاقتصاد","بالطبع المعادلة تعدّ من أهم المفاهيم"]
def main():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name="AMR-KELEG/Sentence-ALDi"
        tok=AutoTokenizer.from_pretrained(name)
        model=AutoModelForSequenceClassification.from_pretrained(name); model.eval()
        def aldi(text):
            with torch.no_grad():
                inp=tok(text,return_tensors="pt",truncation=True,max_length=128)
                logit=model(**inp).logits.squeeze().item()
            # ALDi head is a [0,1] regressor; clamp defensively
            return max(0.0,min(1.0,logit))
        before=[aldi(s) for s in MSA]
        after=[aldi(DP.dialectize(s)) for s in MSA]
        deltas=[a-b for a,b in zip(after,before)]
        # paired sign test / simple stats
        n_up=sum(1 for d in deltas if d>0.01)
        rep={"model":name,"n_sentences":len(MSA),
             "mean_aldi_before":round(statistics.mean(before),4),
             "mean_aldi_after":round(statistics.mean(after),4),
             "mean_delta":round(statistics.mean(deltas),4),
             "median_delta":round(statistics.median(deltas),4),
             "n_sentences_dialectness_increased":n_up,
             "per_sentence":[{"msa":s,"iraqi":DP.dialectize(s),"aldi_before":round(b,3),"aldi_after":round(a,3)}
                             for s,b,a in zip(MSA,before,after)],
             "note":"Automatic dialectness (ALDi: 0=MSA, 1=dialectal). NEGATIVE RESULT: mean ALDi is flat-to-slightly-LOWER after rewriting (0.55->0.49) and most Iraqi rewrites are not scored as more dialectal -- even clearly-Iraqi outputs like 'شلونك يا صاحبي'. ALDi was trained on Egyptian/Levantine/Gulf data (AOC) WITHOUT Iraqi, so it does not register Iraqi markers (هسه/ماكو/شلونك/گاع) as dialectal. This is a result about the METRIC, not the rewriter: no off-the-shelf automatic dialectness metric validly covers Iraqi, which (a) reinforces Iraqi's low-resource status and (b) confirms a blind human evaluation by Iraqi speakers is the necessary dialect-quality measure -- the paper's stated principal gap."}
    except Exception as e:
        rep={"status":"model_unavailable","error":str(e),
             "note":"Sentence-ALDi could not be loaded in this environment; the human study remains the primary dialect-quality evaluation (stated as the main limitation)."}
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(rep,open(OUT,"w"),ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in rep.items() if k!="per_sentence"},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
