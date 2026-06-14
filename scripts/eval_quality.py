#!/usr/bin/env python3
"""
Automated quality-evaluation harness for the live chatbot.

Turns "does it feel right?" into measurable numbers. Runs a fixed set of labelled
queries against a running server and scores three dimensions per query:

  1. ROUTING   — did it answer from books vs general knowledge as expected?
  2. BOOK      — when a specific book is expected, did the cited source match?
  3. LANGUAGE  — is the answer in the same language as the question?

Usage:
    .venv/bin/python scripts/eval_quality.py
    BASE_URL=http://localhost:8000 USER=... PASS=... .venv/bin/python scripts/eval_quality.py

Exit code 0 if the overall score meets PASS_THRESHOLD (default 0.80), else 1 —
so it can gate a deploy / detect regressions.
"""
import json
import os
import re
import sys
import urllib.request

BASE = os.getenv("BASE_URL", "http://localhost:8000")
USER = os.getenv("EVAL_USER", "demo_user")          # set EVAL_USER to your local account
PASS = os.getenv("EVAL_PASS", "")                    # set EVAL_PASS in the environment; never hard-code a real password
THRESHOLD = float(os.getenv("PASS_THRESHOLD", "0.80"))
TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "180"))

# ── Labelled test set ─────────────────────────────────────────────────────────
# books=True  → answer should be grounded in a book (from_books true)
# books=False → general knowledge or small talk (from_books false)
# book        → substring the cited source must contain (None = any/none)
# lang        → expected answer language, == query language
CASES = [
    # English book questions
    {"q": "What is a generative adversarial network?", "books": True,  "book": "generative",            "lang": "en"},
    {"q": "Explain breadth-first search.",             "books": True,  "book": "AI_Search",             "lang": "en"},
    {"q": "What is the main argument of Freakonomics?","books": True,  "book": "Freakonomics",          "lang": "en"},
    {"q": "How do I choose a journal to publish in?",  "books": True,  "book": "Publishing",            "lang": "en"},
    # Arabic book questions
    {"q": "شنو هي المتغيرات بالبرمجة؟",                 "books": True,  "book": "اساسيات",               "lang": "ar"},
    {"q": "اشرح خوارزمية A star للبحث",                 "books": True,  "book": "AI_Search",             "lang": "ar"},
    # Cross-lingual (Arabic question → English-content book)
    {"q": "شنو هو الذكاء التوليدي والشبكات التوليدية؟", "books": True,  "book": None,                    "lang": "ar"},
    # General knowledge (no book) — must NOT cite a book
    {"q": "What is the capital of France?",            "books": False, "book": None,                    "lang": "en"},
    {"q": "شنو عاصمة اليابان؟",                          "books": False, "book": None,                    "lang": "ar"},
    {"q": "How much is 15 multiplied by 4?",           "books": False, "book": None,                    "lang": "en"},
    # Small talk — must NOT cite a book
    {"q": "السلام عليكم، شلونك اليوم؟",                  "books": False, "book": None,                    "lang": "ar"},
    {"q": "لا شكرا، خلص",                                "books": False, "book": None,                    "lang": "ar"},
]


def _http(path, data=None, cookie=None):
    url = BASE + path
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST" if data is not None else "GET")
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def login():
    resp = _http("/auth/login", {"username": USER, "password": PASS})
    sc = resp.headers.get("Set-Cookie", "")
    m = re.search(r"(maktaba_token=[^;]+)", sc)
    if not m:
        sys.exit("✗ login failed — no cookie returned")
    return m.group(1)


def ask(cookie, q):
    """Return (answer_text, from_books, source)."""
    resp = _http("/chat/stream", {"message": q, "session_id": "eval"}, cookie)
    final, toks, meta = "", "", {}
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        p = line[6:]
        if p == "[DONE]":
            break
        try:
            d = json.loads(p)
        except Exception:
            continue
        if "replace" in d:
            final = d["replace"]
        elif "token" in d:
            toks += d["token"]
        if "meta" in d:
            meta = d["meta"]
    ans = final or toks
    return ans, bool(meta.get("from_books")), (meta.get("source") or "")


def lang_of(text):
    ar = len(re.findall(r"[؀-ۿ]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    return "ar" if ar > la else "en"


def main():
    cookie = login()
    rows, routing_ok, book_ok, book_tot, lang_ok = [], 0, 0, 0, 0
    for c in CASES:
        try:
            ans, from_books, src = ask(cookie, c["q"])
        except Exception as e:
            rows.append((c["q"], "ERR", str(e)[:30], "", ""))
            continue
        r_ok = (from_books == c["books"])
        routing_ok += r_ok
        b_ok = True
        if c["books"] and c["book"]:
            book_tot += 1
            b_ok = c["book"].lower() in src.lower()
            book_ok += b_ok
        l_ok = (lang_of(ans) == c["lang"]) if ans.strip() else False
        lang_ok += l_ok
        flags = ("R" if r_ok else "r") + ("B" if b_ok else "b") + ("L" if l_ok else "l")
        rows.append((c["q"][:34], flags, f"books={from_books}", src[:22], c["lang"]))

    n = len(CASES)
    print(f"\n{'query':36} {'flags':6} {'routed':14} {'source':24} lang")
    print("-" * 92)
    for q, flags, routed, src, lang in rows:
        print(f"{q:36} {flags:6} {routed:14} {src:24} {lang}")
    print("-" * 92)
    routing = routing_ok / n
    book = (book_ok / book_tot) if book_tot else 1.0
    lang = lang_ok / n
    overall = (routing + book + lang) / 3
    print(f"Routing (books vs general): {routing_ok}/{n}  = {routing:.0%}")
    print(f"Book selection (when expected): {book_ok}/{book_tot}  = {book:.0%}")
    print(f"Language match: {lang_ok}/{n}  = {lang:.0%}")
    print(f"OVERALL: {overall:.0%}   (threshold {THRESHOLD:.0%})")
    print("Legend: R/B/L = routing/book/language correct (lowercase = failed)")
    sys.exit(0 if overall >= THRESHOLD else 1)


if __name__ == "__main__":
    main()
