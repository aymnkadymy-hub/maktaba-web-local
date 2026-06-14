#!/usr/bin/env python3
"""
Build a VERIFIED bibliography for the maktaba paper from the source-verification
workflow output: dedupe -> BibTeX + a human-readable references.md, and download
every open-access PDF into ~/Desktop/maktaba_paper_sources/pdfs/.

Only sources that were web-verified (real title/authors/venue) are included.
Nothing is invented here.
"""
import json, os, re, sys, urllib.request, ssl

SRC_JSON = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/claude-1000/-home-aymen/09d57a76-482f-4456-acfa-eb49ba1e0ead/tasks/wjjo86wji.output"
OUT = os.path.expanduser("~/Desktop/maktaba_paper_sources")
PDFS = os.path.join(OUT, "pdfs")
os.makedirs(PDFS, exist_ok=True)

data = json.load(open(SRC_JSON))
sources = []
seen = set()
for topic in data["result"]["discovered"]:
    for s in topic["sources"]:
        key = re.sub(r'[^a-z0-9]', '', s["title"].lower())[:50]
        if key in seen:
            continue
        seen.add(key)
        sources.append(s)
print(f"{len(sources)} unique verified sources")

def bibkey(s):
    a = re.sub(r'[^A-Za-z]', '', s["authors"].split(",")[0]) or "anon"
    w = re.sub(r'[^A-Za-z]', '', s["title"].split()[0].lower())
    return f"{a.lower()}{s['year']}{w}"

def venue_type(v):
    v = v.lower()
    if "arxiv" in v: return "misc"
    if any(k in v for k in ["journal","transactions","foundations","trends","pnas","information retrieval","proceedings of the national"]):
        return "article"
    return "inproceedings"

# ---- BibTeX ----
bib = []
keys = {}
for s in sources:
    k = bibkey(s);
    while k in keys: k += "x"
    keys[id(s)] = k
    typ = venue_type(s["venue"])
    fields = [f'  title = {{{s["title"]}}}',
              f'  author = {{{s["authors"]}}}',
              f'  year = {{{s["year"]}}}']
    vf = "journal" if typ == "article" else ("booktitle" if typ == "inproceedings" else "howpublished")
    fields.append(f'  {vf} = {{{s["venue"]}}}')
    if s.get("doi"):
        fields.append(f'  doi = {{{s["doi"]}}}')
    url = s.get("url") or s.get("openAccessPdf")
    if url:
        fields.append(f'  url = {{{url}}}')
    bib.append(f"@{typ}{{{k},\n" + ",\n".join(fields) + "\n}")
open(os.path.join(OUT, "references.bib"), "w").write("\n\n".join(bib) + "\n")
print(f"wrote references.bib ({len(bib)} entries)")

# ---- human-readable references.md ----
md = ["# Verified references — maktaba self-calibrating per-tenant RAG relevance gate",
      f"\n*{len(sources)} sources, every one web-verified (real title/authors/venue). "
      "Open-access PDFs are mirrored in `pdfs/`.*\n"]
for i, s in enumerate(sorted(sources, key=lambda x: x["authors"]), 1):
    doi = f" https://doi.org/{s['doi']}" if s.get("doi") else ""
    url = s.get("url", "")
    md.append(f"{i}. {s['authors']} ({s['year']}). **{s['title']}**. *{s['venue']}*.{doi or (' ' + url)}\n"
              f"   - relevance: {s['relevance']}\n"
              f"   - verified: {s['verifiedFrom'][:160]}")
open(os.path.join(OUT, "references.md"), "w").write("\n".join(md) + "\n")
print("wrote references.md")

# ---- download open-access PDFs ----
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
ok = fail = skip = 0
manifest = []
for s in sources:
    pdf = s.get("openAccessPdf", "").strip()
    k = keys[id(s)]
    if not pdf:
        skip += 1; manifest.append({"key": k, "pdf": None, "status": "no-open-access"}); continue
    dest = os.path.join(PDFS, f"{k}.pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        ok += 1; manifest.append({"key": k, "pdf": pdf, "status": "cached"}); continue
    try:
        req = urllib.request.Request(pdf, headers={"User-Agent": "Mozilla/5.0 (research; bib builder)"})
        with urllib.request.urlopen(req, timeout=45, context=ctx) as r:
            blob = r.read()
        if blob[:4] == b"%PDF" or len(blob) > 30000:
            open(dest, "wb").write(blob); ok += 1
            manifest.append({"key": k, "pdf": pdf, "status": "downloaded", "bytes": len(blob)})
            print(f"  OK   {k}  ({len(blob)//1024} KB)")
        else:
            fail += 1; manifest.append({"key": k, "pdf": pdf, "status": "not-a-pdf"})
            print(f"  ??   {k}  (not a pdf)")
    except Exception as e:
        fail += 1; manifest.append({"key": k, "pdf": pdf, "status": f"error:{type(e).__name__}"})
        print(f"  FAIL {k}  ({type(e).__name__})")

json.dump(manifest, open(os.path.join(OUT, "download_manifest.json"), "w"), indent=2)
print(f"\nPDFs: {ok} downloaded/cached, {fail} failed, {skip} paywalled (no open PDF)")
print(f"folder -> {OUT}")
