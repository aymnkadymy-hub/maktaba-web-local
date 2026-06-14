# maktaba-web-local — The Speaking Library (المكتبة الناطقة)

[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![Paper: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-blue.svg)](paper/)

A **fully offline, on-device, privacy-preserving, multilingual (Arabic + English) personal digital library** with retrieval-augmented question answering. You give it your own PDFs; it answers your questions **from your books** — and, crucially, it knows **when it does *not* know** and declines to invent an answer. Everything runs locally: no cloud, no telemetry, nothing leaves the machine.

> **Research artifact.** This repository accompanies the paper *Self-Calibrating Per-Tenant Relevance Gating with a Conditional Cross-Lingual Fallback for Offline Multilingual RAG* (in [`paper/`](paper/)). Its central contribution is an **unsupervised, label-free, per-tenant self-calibrating relevance gate**: the system derives, for each user's private library, the score cut-off that decides whether any retrieved passage is relevant enough to ground an answer — with no labels, no feedback, no training, and no data leaving the device. A plain-language explanation for non-experts is in [`paper/PLAIN_LANGUAGE_for_everyone.md`](paper/PLAIN_LANGUAGE_for_everyone.md).

---

## What is the contribution?

A retrieval-augmented (RAG) system must decide, after retrieval and reranking, *"is any passage actually relevant, or should I answer from general knowledge instead of forcing an off-topic book into the prompt?"* That decision is a **score cut-off**, and almost everyone hand-tunes a single global constant — which is brittle, because reranker scores shift with the corpus, the language (multilingual rerankers score Arabic lower than English), and the tenant (each user owns a different library).

This system **calibrates the cut-off per tenant, automatically, label-free**, from the relevance scorer's own score geometry: it scores a passage's leading words against *that same passage* (a guaranteed-relevant anchor) and against a passage from a *different book* (an irrelevant anchor), and places the threshold in the gap between the two bands. See [`backend/rag/relevance_gate.py`](backend/rag/relevance_gate.py).

On a real 10-book, 3,739-chunk Arabic+English library, the label-free gate reaches **F1 0.968** versus **0.828** for the deployable hand-tuned default — raising precision from **0.71 to 0.94** while keeping recall at 1.0 (full results, with honest scope and limitations, in the paper).

## Architecture (pipeline)

```
PDF ingest → chunk → HYBRID retrieval [ BM25 (per-user sub-index) + FAISS-HNSW dense ]
           → Reciprocal Rank Fusion → cross-encoder rerank → SELF-CALIBRATING PER-TENANT GATE
           → local LLM (Ollama / llama.cpp) → Iraqi-dialect post-processing → answer
```

Also includes: RAPTOR tree retrieval, HyDE, strict multi-tenant per-user isolation, study tools (MCQ quiz, SM-2 flashcards, notes), and security engineering (bcrypt + SHA-256-hashed tokens + HttpOnly cookies, CSP/CSRF, rate limiting). FastAPI backend (~11k LoC Python) + optional native C++ chunker + Vanilla-JS SPA frontend.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # configure LLM provider (defaults to local Ollama)
python -m uvicorn backend.server_backend:app --host 0.0.0.0 --port 8000
# open http://localhost:8000  · API docs at /docs
```

The default configuration is **fully offline** (embedded Qdrant + local Ollama). No API key is required.

## Reproduce the paper's experiments

The experiment scripts that produced every number in the paper are in [`paper_experiments/`](paper_experiments/) and are **deterministic** (no RNG):

```bash
python paper_experiments/exp_calibration.py     # Exp1: per-tenant cut-offs (cross-encoder + cosine)
python paper_experiments/exp2_gate_accuracy.py  # Exp2: labelled gate accuracy vs baselines
python paper_experiments/exp4_ablation.py       # Exp4: alpha ablation
```

> The evaluation corpus is **private user-owned books and is not redistributed**; the scripts reproduce the methodology on any comparable local library. Because the calibration is label-free and deterministic, pointing it at the same corpus yields the same thresholds.

## Honest project status

Research / working prototype, reduced to practice and released for reproduction. The paper is explicit about scope: the evaluation is a small, single-machine micro-benchmark on a 10-book, English-dominated corpus; the "relevant" test queries are passage-derived spans (easier than natural questions); and the downstream benefit (fewer hallucinated answers) is argued from precision, not yet measured on generated text. See the paper's *Limitations and Threats to Validity*.

## Repository layout

```
backend/            FastAPI backend: api/ auth/ core/ database/ dialect/ llm/ memory/ rag/ utils/
  rag/relevance_gate.py   ← the self-calibrating per-tenant gate (the paper's core)
  rag/hybrid_retriever.py ← BM25 + FAISS-HNSW + RRF + rerank
static/             Vanilla-JS SPA frontend
native_engine/      optional C++ chunker (pybind11) — source only
paper_experiments/  deterministic scripts + numeric results behind the flagship paper
paper/              the flagship paper (md + pdf), plain-language version, verified references
papers/             the 3 companion papers (md+pdf+bib+figures), a complete plain-language
                    explainer, an algorithms catalog, and their deterministic experiments/
scripts/, tests/    helper scripts and the test suite
```

## Companion papers (the maktaba research series)

Alongside the flagship paper, three focused companion papers (an honest, disclosed split of the same project, each with a distinct contribution; they cross-cite) are in [`papers/`](papers/), each with its PDF, sources, figures, and deterministic experiments under [`papers/experiments/`](papers/experiments/):

- **P1 — [Sparse-Retrieval Tenant Starvation](papers/bm25_tenant_starvation/)**: a lexical post-filtering failure mode in shared-index multi-tenant retrieval (one tenant's passages crowd the global top-N before the per-tenant filter), measured on the live skew, with a per-tenant BM25 sub-index remedy. We also measure that dense post-filtering starves too.
- **P2 — [A Self-Extending Bilingual Glossary](papers/glossary_continual_xlingual/)**: continual, label-free cross-script term acquisition (no model weight updates) that repairs the lexical half of the hybrid retriever for Arabic↔English, with measured dense (MiniLM/LaBSE) baselines.
- **P3 — [Safe Deterministic Post-Hoc Dialect Rewriting](papers/iraqi_dialect_rag/)**: boundary-guarded MSA→Iraqi rewriting that is provably unable to corrupt ordinary words, evaluated for zero corruption vs. an unguarded baseline.

Also in `papers/`: a complete **plain-language explainer** of the whole system and an **algorithms catalog** (with an honest novelty ledger). Every reported number is regenerated by the scripts in `papers/experiments/`; the private evaluation corpus is not redistributed.

## External resources (not redistributed here)

The Iraqi-dialect post-processor is built from third-party Arabic dialect datasets (IADD, IA2D) which are **not** included in this repository for size and licensing reasons; clone them separately into `backend/dialect/` if you need to rebuild the dialect map.

## Citation

If you use this work, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button reads it), or the flagship paper directly (see [`paper/references.bib`](paper/references.bib) for its bibliography):

```
Ayman Kazim Yousef (2026). Self-Calibrating Per-Tenant Relevance Gating with a
Conditional Cross-Lingual Fallback for Offline Multilingual RAG.
```

For a specific companion paper (P1/P2/P3), cite that paper from [`papers/`](papers/).

**Author:** Ayman Kazim Yousef — Department of Artificial Intelligence Engineering, AlSafwa University, Karbala, Iraq · ORCID [0009-0006-7409-9367](https://orcid.org/0009-0006-7409-9367).

## License

Code is released under the **MIT** licence ([`LICENSE`](LICENSE)); the paper, figures, and prose documentation under **CC BY 4.0**.
