# The Speaking Library, explained for everyone

*A plain-language companion to the paper "Self-Calibrating Per-Tenant Relevance Gating with a Conditional Cross-Lingual Fallback for Offline Multilingual RAG."*
**Author: Ayman Kazim Yousef — AlSafwa University, Karbala, Iraq.**

---

## 1. What is this about, in one sentence?

I built a private app that lets you **chat with your own books** — it reads the PDFs you give it and answers your questions from them — and this paper is about one small but important piece of that app: **teaching it to know when it does *not* know.**

## 2. The everyday problem

Imagine a very eager library assistant. You ask a question, and the assistant *always* hands you some book — even when **no** book in the library is actually about your question. So if your library is full of programming books and you ask "what's the capital of France?", a bad assistant grabs a programming book anyway and reads you a confident, wrong answer, claiming it came from that book.

AI "chat-with-your-documents" systems (called **RAG**) have exactly this failure. After they search your documents, they must decide: *"Is anything I found actually relevant? Or should I admit the library doesn't cover this and answer from general knowledge instead?"* Getting this wrong is a top cause of **made-up ("hallucinated") answers** and **wrong source citations**.

## 3. How that decision is normally made — and why it's fragile

The system gives every passage a **relevance score**, and uses a single **cut-off number**: if the best passage scores above the line, "use it"; below the line, "ignore it." Almost everyone sets this line **by hand, once, the same for everybody.**

That one-size-fits-all line is fragile, because the scores **shift**:
- with the **topic** of your library,
- with the **language** (the AI scores Arabic passages lower than English ones, even when they're equally relevant), and
- with the **person** — every user owns a *different* private library.

So one fixed line is too strict for some users (it hides good answers) and too loose for others (it lets in off-topic junk).

## 4. The idea in this paper

Instead of one hand-set line for everyone, the system **figures out its own line, automatically, for each user's library** — using **no labels, no feedback, no training, and without anything leaving your device.**

The trick is simple and intuitive. From your own books the system quietly builds two little piles of scores:
- a **"definitely relevant" pile** — it takes the first words of a passage and scores them against *that same passage* (obviously a match), and
- a **"definitely unrelated" pile** — it scores those same words against a passage from a *different book* (obviously not a match).

Now it can *see* where "relevant" scores sit and where "unrelated" scores sit, and it places the cut-off **in the gap between the two piles.** No human, no labels — the library calibrates itself.

A nice bonus falls out for free: because Arabic passages score lower overall, the "piles" for an Arabic library sit lower, so the system automatically picks a **different, fairer line for Arabic** — without being told which language it is.

There's also a **bonus for multilingual users**: if you ask in Arabic and your relevant book is in English, the system notices the in-language search found nothing good and **only then** translates your question once and tries again — so it never wastes time translating when it doesn't need to.

## 5. Does it actually work? (honest version)

I ran real tests on my own library — **10 books, ~3,700 passages, English and Arabic** — on an ordinary computer. The result:

- The self-calibrating line scored **0.97** on a "did it accept/reject correctly?" measure, versus **0.83** for the old hand-set line. In plain terms, the old line wrongly accepted **25** off-topic passages; the new one accepted only **4** — cutting the junk that causes made-up answers by about **five-sixths**, without losing any good answers.
- It found a *different* right line for each of five different libraries — proving one global line can't fit everyone.

**Where I'm honest about the limits** (this is in the paper, stated plainly):
- This is a **small test on one computer**, not a giant benchmark.
- My test questions were **snippets taken from the books themselves**, which is easier than real human questions — so the real-world numbers will be lower.
- I measured the *decision* (accept/reject), **not** the final written answers — so "fewer made-up answers" is a well-reasoned expectation, not yet a measured fact.
- A line "tuned with a labeled answer key" would actually score even higher — **but** you can never get that answer key for someone's *private* library, which is the whole point: my method needs **no answer key at all.**

I'd rather tell you these limits up front than have a reviewer find them — and that honesty is what makes the work trustworthy.

## 6. Why it matters

For a private, on-device study app — the kind a student in Karbala or anywhere can run on a modest laptop with no internet and no cloud bills — this means: **better answers, fewer confident mistakes, fair treatment of Arabic, and complete privacy** (nothing about your books ever leaves your machine). The whole system and the experiment scripts are released so anyone can check the numbers themselves.
