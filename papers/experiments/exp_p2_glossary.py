#!/usr/bin/env python3
"""
Paper 2 — Bilingual glossary + continual term learning for offline cross-script CLIR.

Deterministic experiment on the REAL maktaba-web-local corpus (tenant A, which
owns both English and Arabic books). Uses the project's OWN glossary
(backend/rag/glossary.cross_lingual_terms) and Arabic normaliser, and the same
bm25s lexical engine the system uses.

Measures book-level cross-script retrieval recall@k:
  (1) WITHOUT glossary expansion  — a query in language X vs a book in language Y
  (2) WITH glossary expansion     — other-language equivalents appended (no LLM)
and a CONTINUAL-LEARNING demo: held-out terms absent from the static glossary,
recall before vs. after the term pair is learned (as learn_term() would persist).

Run: ../../maktaba-web-local/.venv/bin/python exp_p2_glossary.py
"""
import json, os, re, sys
sys.path.insert(0, os.environ.get("MAKTABA_ROOT", os.path.expanduser("~/maktaba-web-local")))
import bm25s
import backend.rag.glossary as G

CORPUS = os.environ.get("MAKTABA_CORPUS", os.path.expanduser("~/maktaba-web-local/bm25_cache/corpus.json"))
OUT    = os.path.join(os.path.dirname(__file__), "results", "exp_p2_glossary.json")
TENANT = "3CLDpsvv7RhtClg5iqhr-g"
K = 10

def _normalize(text):
    text = re.sub(r'[ً-ٰٟ]', '', text); text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ى','ي',text); text = re.sub(r'ة','ه',text); text = re.sub(r'ـ','',text)
    return text.lower()
_AR_STOP=['في','من','الي','علي','عن','مع','بين','حتي','لكن','او','ان','اذا','لو','بل','ثم',
 'هو','هي','هم','هن','انا','نحن','انت','هذا','هذه','ذلك','تلك','الذي','التي','الذين','كان',
 'كانت','يكون','تكون','لا','ما','لم','لن','قد','كل','بعض','جميع','غير','وفي','وعلي','ومن','وهو','وهي','وان']

def build(docs):
    normed=[_normalize(d["c"]) for d in docs]
    pairs=[(d,n) for d,n in zip(docs,normed) if n.strip()]
    kept=[p[0] for p in pairs]
    idx=bm25s.BM25(); idx.index(bm25s.tokenize([p[1] for p in pairs],stopwords=_AR_STOP,show_progress=False),show_progress=False)
    return idx,kept
def search(idx,kept,q,topn):
    qt=bm25s.tokenize([_normalize(q)],stopwords=_AR_STOP,show_progress=False)
    topn=min(topn,len(kept)); res,sc=idx.retrieve(qt,k=topn,show_progress=False)
    return [kept[int(i)] for i,s in zip(res[0],sc[0]) if float(s)>0]
def books_in(docs): return [d["m"].get("book_title","") for d in docs]

def main():
    data=json.load(open(CORPUS))
    A=[{"c":e.get("c",""),"m":(e.get("m",{}).get("metadata",e.get("m",{})) if isinstance(e.get("m"),dict) else {})} for e in data]
    A=[d for d in A if d["m"].get("user_id")==TENANT]
    idx,kept=build(A)

    def expand(q):
        ext=G.cross_lingual_terms(_normalize(q))
        return (q+" "+ext).strip() if ext else q
    def hit(q, target_substr, use_glossary):
        qq=expand(q) if use_glossary else q
        bks=books_in(search(idx,kept,qq,K))
        return any(target_substr.lower() in (b or "").lower() for b in bks), G.cross_lingual_terms(_normalize(q))

    # ── labelled cross-script queries: (query, lang, target_book_substr) ───────
    AR2EN=[  # Arabic query, answer in an English book
        ("الشبكة العصبية التوليدية","generative-deep-learning"),
        ("فرط التخصيص في الشبكة العصبية","generative-deep-learning"),
        ("خوارزمية البحث","AI_Search_Algorithms"),
        ("الحوافز الاقتصادية","Freakonomics"),
        ("النشر العلمي ومعامل التأثير","100 Q&A About Scientific Publishing"),
        ("التحكيم والسرقة الادبية","100 Q&A About Scientific Publishing"),
    ]
    EN2AR=[  # English query, answer in an Arabic book
        ("partial fractions","الكسور الجزئيه"),
        ("programming variable and function","اساسيات البرمجة"),
        ("loop and array in programming","اساسيات البرمجة"),
    ]
    report={"corpus":{"tenant":TENANT,"n_chunks":len(A),
                      "books":sorted(set(books_in(A)))},
            "config":{"k":K,"glossary_ar_en":len(G._AR_EN),"glossary_en_ar":len(G._EN_AR)},
            "static_glossary":{}}
    for name,qs in [("AR_to_EN",AR2EN),("EN_to_AR",EN2AR)]:
        rows=[]; w_off=w_on=0
        for q,tgt in qs:
            off,_=hit(q,tgt,False); on,ext=hit(q,tgt,True)
            w_off+=int(off); w_on+=int(on)
            rows.append({"query":q,"target":tgt,"hit_without":off,"hit_with":on,"expansion":ext})
        report["static_glossary"][name]={"n":len(qs),
            "recall_without_glossary":round(w_off/len(qs),4),
            "recall_with_glossary":round(w_on/len(qs),4),"rows":rows}
    allq=AR2EN+EN2AR
    off=sum(int(hit(q,t,False)[0]) for q,t in allq); on=sum(int(hit(q,t,True)[0]) for q,t in allq)
    report["static_glossary"]["overall"]={"n":len(allq),
        "recall_without_glossary":round(off/len(allq),4),
        "recall_with_glossary":round(on/len(allq),4)}

    # ── continual-learning demo: held-out terms ABSENT from the static glossary ─
    # single-term Arabic queries whose only useful expansion is the held-out term.
    HELDOUT=[("الانتروبيا","entropy","generative-deep-learning"),
             ("الكامن","latent","generative-deep-learning"),
             ("اللوجستي","logistic regression","generative-deep-learning")]
    cl={"absent_confirmed":[], "before":[], "after":[]}
    G._LEARNED.clear()  # ensure clean slate (in-memory; not persisted)
    for ar,en,tgt in HELDOUT:
        absent = (_normalize(ar) not in G._AR_EN) and (_normalize(ar) not in G._LEARNED)
        cl["absent_confirmed"].append({"term":ar,"absent_from_static":absent})
        b,_=hit(ar,tgt,True); cl["before"].append({"term":ar,"hit":b})
    before_recall=sum(int(x["hit"]) for x in cl["before"])/len(HELDOUT)
    # learn the pairs EXACTLY as the deployed _translate_query→learn_term path does:
    # a prefix-stripped key and the HEAD WORD of the translation (hybrid_retriever.py:105-106),
    # written in-memory (not persisted, to keep the released lexicon unmodified).
    for ar,en,tgt in HELDOUT:
        G._LEARNED[G._strip_ar_prefix(_normalize(ar))] = en.split()[0].lower()
    for ar,en,tgt in HELDOUT:
        b,ext=hit(ar,tgt,True); cl["after"].append({"term":ar,"hit":b,"expansion":ext})
    after_recall=sum(int(x["hit"]) for x in cl["after"])/len(HELDOUT)
    G._LEARNED.clear()
    report["continual_learning"]={"n_heldout":len(HELDOUT),
        "recall_before_learning":round(before_recall,4),
        "recall_after_learning":round(after_recall,4),"detail":cl}

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(report,open(OUT,"w"),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
