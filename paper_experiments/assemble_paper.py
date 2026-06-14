#!/usr/bin/env python3
"""Assemble the drafted sections + verified references into one paper_draft.md."""
import json, os, re

OUT_DIR = os.path.expanduser("~/Desktop/maktaba_paper")
os.makedirs(OUT_DIR, exist_ok=True)
WF = "/tmp/claude-1000/-home-aymen/09d57a76-482f-4456-acfa-eb49ba1e0ead/tasks/woqy6ifas.output"
SRC = os.path.expanduser("~/Desktop/maktaba_paper_sources/verified_sources.json")

wf = json.load(open(WF))
drafts = wf["result"]["drafts"]
# paper-order: abstract-intro, related, problem-method, system, evaluation, discussion
order = ["Abstract + 1. Introduction", "2. Related Work",
         "3. Problem Formulation + 4. Method", "5. System and Implementation",
         "6. Evaluation", "7. Discussion, Limitations, Threats to Validity, Ethics & 8. Conclusion"]
by = {d["section"]: d["markdown"] for d in drafts}
body = "\n\n".join(by[s] for s in order if s in by)

# ---- References (alphabetical by first author) from verified sources ----
src = json.load(open(SRC))
seen, refs = set(), []
for topic in src["result"]["discovered"]:
    for s in topic["sources"]:
        k = re.sub(r'[^a-z0-9]', '', s["title"].lower())[:50]
        if k in seen: continue
        seen.add(k); refs.append(s)
def sortkey(s): return s["authors"].split(",")[0].lower()
refs.sort(key=sortkey)
reflines = ["\n\n---\n\n## References\n"]
for s in refs:
    doi = f" https://doi.org/{s['doi']}" if s.get("doi") else (f" {s['url']}" if s.get("url") else "")
    reflines.append(f"- {s['authors']} ({s['year']}). {s['title']}. *{s['venue']}*.{doi}")
body += "\n".join(reflines) + "\n"

open(os.path.join(OUT_DIR, "paper_draft.md"), "w").write(body)
print(f"assembled paper_draft.md: {len(body)} chars, {len(refs)} references")
print(f"-> {OUT_DIR}/paper_draft.md")
