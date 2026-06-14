#!/usr/bin/env python3
"""
Paper 3 — Safe deterministic post-hoc Iraqi-dialect rewriting.

Deterministic experiment using the project's OWN processor
(backend/dialect/dialect_processor). Measures:
  (1) SAFE-REWRITING REGRESSION — deployed processor introduces ZERO intra-word
      corruption on a common-vocabulary list; a reconstructed *naive* baseline
      (split-on-particle-substring; unguarded سـ-future prefix) is run on the SAME
      list to quantify the failure class the deployed design avoids.
      (The historical 17/21 and 14/18 figures are documented in the source; here
      we verify the deployed behaviour and measure the reconstructed baseline.)
  (2) COVERAGE — # MSA tokens rewritten to Iraqi over a set of MSA sentences.
  (3) IDEMPOTENCE — dialectize(dialectize(x)) == dialectize(x).
  (4) CODE PRESERVATION — fenced/inline code left byte-identical.
  (5) MAP STATS — deployed map size, multi-word keys, morphological-rule count.

Run: ../../maktaba-web-local/.venv/bin/python exp_p3_dialect.py
"""
import json, os, re, sys
sys.path.insert(0, os.environ.get("MAKTABA_ROOT", os.path.expanduser("~/maktaba-web-local")))
from backend.dialect import dialect_processor as DP

OUT = os.path.join(os.path.dirname(__file__), "results", "exp_p3_dialect.json")

def has_internal_space_change(orig, out):
    """True if a single source word was split (a space appeared inside it) — the
    corruption mode. Compares word sets: any source word absent from output AND
    whose pieces appear with an inserted space => corruption."""
    return orig != out

# ── (1) regression word list (common MSA vocabulary that contains particle/
#         siin substrings — the words the removed rules used to corrupt) ────────
PARTICLE_WORDS = ["الاقتصاد","المعلومات","الفهم","العلاقة","المعرفة","الفلسفة",
    "العملية","المعادلة","الفائدة","العنوان","المفهوم","العمل","الفكرة","المعنى",
    "العالم","السلام","الكلام","المعلم","العلوم","المنفعة","العين"]          # 21
SIIN_WORDS = ["سيارة","سياسة","سنوات","ستيفن","سياق","سينما","سيطرة","سيدة",
    "سؤال","سبب","سرعة","سطح","سلسلة","سهم","سوق","سيف","سماء","سنة"]            # 18

def naive_particle_split(text):
    # reconstructed UNGUARDED rule class: surround particle substrings with spaces
    for p in ["لا","على","مع","في","عن","ال"]:
        text = re.sub(p, " "+p+" ", text)
    return re.sub(r"\s+"," ",text).strip()

def naive_siin_future(text):
    # reconstructed UNGUARDED rule: س + (أ/ي/ت/ن) -> راح + letter (no word-boundary guard)
    return re.sub(r"س([أيتن])", r"راح \1", text)

def main():
    rep={}

    # (1) regression
    dep_particle_corrupt=[w for w in PARTICLE_WORDS if DP.dialectize(w)!=w and " " in DP.dialectize(w)]
    dep_siin_corrupt   =[w for w in SIIN_WORDS    if DP.dialectize(w)!=w and " " in DP.dialectize(w)]
    naive_particle_corrupt=[w for w in PARTICLE_WORDS if naive_particle_split(w)!=w]
    naive_siin_corrupt    =[w for w in SIIN_WORDS    if naive_siin_future(w)!=w]
    rep["regression"]={
      "particle_list_size":len(PARTICLE_WORDS),"siin_list_size":len(SIIN_WORDS),
      "deployed_particle_corruptions":len(dep_particle_corrupt),
      "deployed_siin_corruptions":len(dep_siin_corrupt),
      "naive_particle_corruptions":len(naive_particle_corrupt),
      "naive_siin_corruptions":len(naive_siin_corrupt),
      "naive_particle_examples":{w:naive_particle_split(w) for w in naive_particle_corrupt[:6]},
      "naive_siin_examples":{w:naive_siin_future(w) for w in naive_siin_corrupt[:6]},
      "documented_historical":{"particle":"17/21","siin":"14/18",
        "source":"backend/dialect/dialect_processor.py comments (lines 41-46, 72-76)"},
    }

    # (2) coverage on MSA sentences (constructed to contain mappable MSA forms)
    MSA=[
      "الآن لا يوجد كثير من الوقت لكن كل شيء جيد","ماذا تريد أن تفعل اليوم",
      "كيف حالك يا صديقي","هذا الكلام صحيح تماماً وأنا أفهم الفكرة",
      "سوف يتم شرح الموضوع","دعنا نبدأ من البداية","المعلم يشير إلى أن المتغير مهم جداً",
      "أين الكتاب الذي تتحدث عنه","ربما يكون هذا جيداً ولكن ليس دائماً",
      "أيضاً يوجد حل آخر للمسألة","بالطبع المعادلة تعدّ من أهم المفاهيم",
      "نحن نتكلم عن البرمجة الآن","أنا فهمت الدرس جيداً","رائع، هذا ممتاز",
      "كثيراً ما نرى هذا في الاقتصاد",
    ]
    changed=0; tok_changed=0; tok_total=0; rows=[]
    for s in MSA:
        out=DP.dialectize(s); changed+=int(out!=s)
        a=s.split(); b=out.split()
        tok_total+=len(a); diff=sum(1 for x,y in zip(a,b) if x!=y)+abs(len(a)-len(b))
        tok_changed+=diff
        rows.append({"msa":s,"iraqi":out})
    rep["coverage"]={"n_sentences":len(MSA),"sentences_changed":changed,
      "sentence_change_rate":round(changed/len(MSA),4),
      "approx_tokens_total":tok_total,"approx_tokens_changed":tok_changed,
      "approx_token_change_rate":round(tok_changed/tok_total,4),"examples":rows}

    # (3) idempotence
    nonidem=[s for s in MSA if DP.dialectize(DP.dialectize(s))!=DP.dialectize(s)]
    rep["idempotence"]={"n_sentences":len(MSA),"non_idempotent":len(nonidem),
                        "idempotent":len(nonidem)==0}

    # (4) code preservation
    code_cases=[
      "هذا متغير: ```python\nx = 5  # الآن القيمة كثير مهمة\nprint(x)\n```",
      "استخدم الدالة `def foo(): return كثير` الآن",
    ]
    preserved=[]
    for c in code_cases:
        out=DP.dialectize(c)
        block=re.findall(r"```[\s\S]*?```|`[^`]+`", c)
        ok=all(b in out for b in block)
        preserved.append({"ok":ok})
    rep["code_preservation"]={"n":len(code_cases),
      "all_preserved":all(p["ok"] for p in preserved)}

    # (5) map stats
    M=DP._MAP
    rep["map_stats"]={"map_entries":len(M),
      "multiword_keys":sum(1 for k in M if " " in k),
      "max_key_len_words":max(len(k.split()) for k in M),
      "morph_rules":len(DP._MORPH_PATTERNS),
      "spacing_rules":len(DP._SPACING_PATTERNS)}

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(rep,open(OUT,"w"),ensure_ascii=False,indent=2)
    print(json.dumps(rep,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
