This is a synthesis task — I have four per-area analyses and need to consolidate them into one materials brief. No file reading needed; everything is in the provided context. Let me produce the brief directly.

# MATERIALS BRIEF — "ALGORITHMIC BIAS AGAINST ARABIC"
### (Maktaba project; scientific version + plain-language version)

---

## 1. THE HONEST THESIS (grounded in real measured numbers)

**The bias against Arabic in multilingual retrieval is SCORE COMPRESSION (reduced discriminability), NOT lower scores.** The earlier "Arabic scores lower / tau=0.52" framing is refuted by our own data and must be dropped entirely.

The defensible, measured claim:

> A shared multilingual reranker/embedder is **less discriminative for Arabic than for English**. It crams Arabic relevant-pair scores into a narrow band (cross-encoder self-match std AR 0.348 vs EN 0.909, ~2.6x tighter; cosine std AR 0.062 vs EN 0.128, ~2.1x tighter), driven in part by a 1.27x subword fragmentation penalty (AR 1.95 vs EN 1.54 tokens/word). Arabic relevant pairs do **not** score lower — they score *slightly higher* (CE 10.53 vs 10.45; cosine 0.805 vs 0.754) — but the relevant-vs-irrelevant separation shrinks because the dynamic range collapses. A global cutoff calibrated on English's wide geometry therefore sits in the wrong place for Arabic.

**The mitigation is an EMERGENT engineering property, not retraining.** Three frozen-model mechanisms fight this bias:
1. A **self-calibrating per-tenant relevance gate** that reads each tenant's own score geometry and automatically hands Arabic-heavy tenants a stricter, geometry-matched cutoff — with no language-specific code, no labels, no training.
2. A **continual self-extending cross-script glossary** that lifts the structural Arabic→English lexical floor from 0 to 1.0 and learns held-out Arabic terms one-shot, with zero weight updates.
3. A **per-tenant BM25 sub-index** that prevents dominant-tenant starvation of minority (often Arabic) libraries.

The honest one-liner: **Arabic fairness is achieved as an engineering property of frozen, on-device models — not as a model-training result.**

---

## 2. CONSOLIDATED REAL NUMBERS (deduplicated)

### A. The core bias — measured (papers/arabic_bias, exp_arabic_bias.json; n=40/language; reranker's own tokenizer & production models)
| Quantity | Arabic | English | Note |
|---|---|---|---|
| Tokenization fertility (tokens/word) | **1.946** | **1.536** | 1.27x more fragmented for Arabic |
| Cross-encoder self-match **std** | **0.348** | **0.909** | EN ~2.6x wider → AR less discriminative |
| Cosine self-match **std** | **0.062** | **0.128** | EN ~2.1x wider |
| Cross-encoder self-match mean | 10.534 | 10.448 | EN−AR = −0.087 (**Arabic slightly HIGHER**) |
| Cosine self-match mean | 0.805 | 0.754 | EN−AR = −0.051 (**Arabic higher**) |
| Cross-encoder self-match min / max | 9.485 / 10.891 | **5.214** / 10.857 | EN has the long left tail; AR bunched |
| Cosine self-match range | 0.657–0.913 | 0.432–1.000 | AR compressed |
| Cross-doc (irrelevant) CE mean (std) | −6.69 (1.179) | −7.673 (2.261) | AR irrelevant band higher+tighter → smaller separation |
| Cross-doc cosine mean (std) | 0.130 (0.144) | 0.147 (0.097) | (cosine irrelevant std is *smaller* for EN — see §5) |

### B. The gate (flagship paper_final.md; 3,739 chunks / 10 books / 5 tenants; 60 IN + 60 OUT probe)
- Self-calibrated label-free: **F1 0.968** (acc 0.967, prec 0.938, rec 1.000)
- Fixed −5.0 default: **F1 0.828** (acc 0.792, prec 0.706, rec 1.000)
- **Precision 0.71 → 0.94** at recall pinned to 1.0
- Off-topic acceptances: fixed default accepts **25**, gate accepts **4** (removes 21/25)
- Oracles (need labels, undeployable): global-fixed +3.04 → F1 1.000; per-tenant → F1 0.992 (gate is "near-oracle")
- Alpha ablation: F1 = 0.736 / 0.845 / **0.968** / 0.992 / 0.992 / 0.992 at α = 0.00 / 0.10 / **0.25** / 0.50 / 0.75 / 1.00; recall=1.0 throughout; α≥0.75 plateau is a clamp-to-+2.0 artifact (all 5 tenants clamped) — not true insensitivity

### C. Per-tenant cutoffs — CANONICAL run (use this set consistently; paper_final Exp1 / exp1_cutoffs.json)
- **Cross-encoder tau_t:** T1 EN ML/AI **−2.93**, T2 EN econ **−2.75**, T3 Arabic+math **−1.59**, T4 Mixed AR+EN **−1.39**, T5 STEM **−3.29**. Span [−3.29, −1.39] ≈ 1.9 logits. **The two Arabic-containing tenants (T3, T4) are the two strictest.**
- **Cosine tau_t:** T1 0.382, T2 0.329, T3 (pure-Arabic) **0.409 (highest of all)**, T4 0.302 (lowest), T5 0.325.
- **Critical cross-check:** Arabic tenants' in-domain `in_lo` are NOT lower — T3 10.615, T4 10.584 vs T1 10.596, T2 10.521. **The higher Arabic cutoff comes from a tighter/compressed band, not from higher means and not from lower relevant scores.**
- Tenant sizes: T1=1535, T2=2010, T3=177, T4=2733, T5=45 (two orders of magnitude).
- *(There is a second run in PATENT_DISCLOSURE — AR −1.79/−1.27 vs EN −3.31/−2.75; cosine AR 0.424 vs EN 0.382/0.329 — same direction, different absolutes. Cite the canonical run; note run-to-run variation in a footnote. Do NOT mix the two.)*

### D. Cross-script glossary (papers/glossary; real 3725-chunk bilingual tenant; book-level recall@10)
- **Arabic→English: 0.00 → 1.00** with glossary (n=6) — the Arabic lexical floor is **exactly 0 by construction**
- English→Arabic: **1.00 → 1.00** (n=3) already bridged — Arabic technical books embed English terms verbatim (asymmetry, reported not averaged)
- Overall 0.333 → 1.000
- Continual one-shot acquisition (n=3 held-out, verified absent): al-intrubiya→entropy, al-kamin→latent, al-lujisti→logistic; 0.0 → 1.0 each; deterministic/model-free thereafter; **no weight updates**
- Glossary-size ablation (AR→EN recall vs fraction): 0% → 0.0; 25%/77 → 0.833; 50%/154 → 0.833; 75%/231 → 1.0; 100%/308 → 1.0. Deployed: **308 AR→EN + 280 EN→AR pairs**
- Cost: glossary ~**48 µs/query** (median 48.12, p95 53.57 over 10k calls), no model — vs dense MiniLM 16.19 ms/query + 92.1 s corpus embed; LaBSE 21.91 ms/query + 453.1 s

### E. Tenant starvation & per-tenant sub-index (papers/bm25_tenant_starvation; real 98.5%/1.5% skew, 3725 vs 58 chunks)
- Shared-vocab minority oracle-overlap@5: 0.50→0.913, 0.80→0.833, 0.90→0.707, 0.95→0.631, 0.98→0.483, **0.9847(real)→0.461**; mean yield 4.72→2.11 of 5
- Distinct-vocab control: flat **0.968 → 0.914** (isolates vocabulary overlap, NOT tenant size, as the cause)
- Loss decomposition at real skew: deployed 0.461 (95% CI [0.35, 0.578]) → 0.856 (crowding removed) → 1.0 (per-tenant). **Crowding effect 0.394; shared-IDF statistics-capture 0.144 (sparse-only)**
- **Dense ALSO starves (measured):** FAISS post-filter (8x) overlap@5 collapses 1.0 → **0.322** at real skew, *below* sparse 0.461 — refutes "dense is immune." Per-tenant index restores both to 1.0
- Over-fetch: m=1 recall 0.889 / 11.1% zero-yield; m=3 0.944/0.056; m≥6 1.0/0.0; deployed 18x. Target global rank median 1, p90 13, max 25
- Per-tenant build/mem/query: 58 chunks 2.8 ms / 0.3 MB / 0.037 ms; 500 chunks 23.6 ms / 2.3 MB; 3725 chunks 170 ms / 11.4 MB. Cache bounded 200 tenants
- Live skew also observed at 2697/58/33 across 3 users (97/2/1) — same motivation

### F. Iraqi dialect rewriter (papers/iraqi_dialect_rag)
- Safety (intra-word corruption): deployed boundary-guarded **0/21 + 0/18**; naive baseline 21/21 + 10/18; removed historical rules 17/21 + 14/18
- Coverage: 15/15 sentences (100%), ~54% tokens changed (44/82). Map 321 entries (129 multi-word, max 3-word keys), 22 morph rules, 2 spacing rules. Idempotent, code-preserving. Cost 0.148 ms/sentence
- **ALDi negative result (Sentence-ALDi):** dialectness 0.5528 → 0.4898 (delta −0.063), only 2/15 rose — ALDi (Egyptian/Levantine/Gulf AOC, no Iraqi) doesn't register هسه/ماكو/شلون/گاع. A **metric-coverage gap, not a rewriter failure**
- Iraqi low-resource scale: IADD has **216 Iraqi / 135,804 total (~0.16%)**; AraDiCE omits Iraqi; AL-QASIDA's 8 varieties exclude Iraqi; MADAR = 25 cities + MSA

### G. Corpus / system constants
- Flagship corpus: 3,739 chunks / 10 books / 5 themed tenants, EN-dominated, CPU. Starvation corpus: 3,838 entries / 13 books / 2 tenants (3725/58/55-null)
- Live corpus language skew: 3,775 EN-dominant chunks vs **62 AR-dominant** (the one clearly Arabic book = "اخيرة عربي", 89% AR, 45 chunks) — Arabic data volume is modest (see §5)
- Models: reranker **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** (~52 MB, 100 langs, MS MARCO); embedder **paraphrase-multilingual-MiniLM-L12-v2** (384-dim); dialect detector **UBC-NLP/MARBERT**
- Gate constants: S=8 pseudo-queries, 12-word head, 18-word probe spans, α=0.25, out_hi=Q75(S_out), in_lo=Q25(S_in), gap test in_lo>out_hi, clamp CE [−8,+2] / cosine [0,0.95], guard K≥2 & |C_t|≥6, cost 2S=16 scorer calls, cache key (tenant, scorer, corpus-state)
- Pipeline: BM25 (bm25s) + FAISS-HNSW dense, RRF (k=60), cross-encoder rerank, then gate; strict user_id isolation; fully offline; local LLM (qwen2.5)

---

## 3. THE THREE MITIGATIONS — how each fights Arabic bias

### Mitigation 1 — Self-calibrating per-tenant relevance gate → fights SCORE COMPRESSION
**Algorithm:** For tenant t, take S=8 deterministic pseudo-queries (first 12 words of evenly-sampled chunks). Build S_in = self-match scores M(q_i, c_i) (guaranteed-relevant upper anchor) and S_out = cross-book scores M(q_i, c'_i) (irrelevant lower anchor). Set out_hi=Q75(S_out), in_lo=Q25(S_in). If in_lo > out_hi, place tau_t = out_hi + α·(in_lo − out_hi), α=0.25; else safe default. Clamp, cache per (tenant, scorer, corpus-state), accept iff s* ≥ tau_t. Cost 2S=16 scorer calls per (re)calibration.

**Why it fights Arabic bias (the load-bearing argument, Sec 4.3):** Both anchors are drawn from the *same corpus* and scored by the *same model M*, so any per-language behaviour enters S_in and S_out identically and is absorbed into where the gap sits. The gate **never inspects language**. Because Arabic's score band is *compressed* (tighter std), the same percentile-gap geometry lands the cutoff *higher* — so Arabic-heavy tenants automatically receive a stricter, geometry-matched bar (pure-Arabic T3 strictest in both score spaces; Arabic-containing tenants strictest in cross-encoder). A single global English-tuned cutoff would systematically mis-gate Arabic; the gate corrects this **without retraining**. Scorer-agnostic: runs identically on the unbounded cross-encoder logit OR the bounded bi-encoder cosine (no torch needed — fairness-of-access on low-end devices). Paired with a **conditional cross-lingual fallback**: translate once (cached, timeout-bounded) only on a calibrated same-language miss.

### Mitigation 2 — Continual self-extending cross-script glossary → fights the STRUCTURAL LEXICAL FLOOR
**Algorithm:** expand(q) = q ∪ {D(w) : w ∈ terms(q)} — additive cross-script expansion (recall monotone, no model at query time). Multi-word keys matched before components; Arabic clitic-prefix stripping (peels ال/و/ب/ل/ف/ك/لل). Continual acquisition: a same-language **retrieval miss is the supervision signal** — on a short single-term miss, the one-off LLM translation is captured by learn_term(strip_prefix(head(q)) → head(translation)) into bounded local JSON (≤5000 pairs; guards: no curated overwrite, key≥3 chars, translation≤60 chars). The model fires at most once per term, ever.

**Why it fights Arabic bias:** An Arabic query can *never* lexically match an English passage — BM25 = 0 by construction (the absolute floor of the bias, AR→EN recall structurally zero). The glossary lifts this to 1.0 at ~48 µs with no model and **no weight update**, and recovers held-out Arabic terms (entropy/latent/logistic) one-shot. It is **continual learning of the retrieval LAYER, not weights** — auditable, revertible, no catastrophic forgetting (explicit contrast to CREAM, which updates weights).

### Mitigation 3 — Per-tenant BM25 sub-index → fights MINORITY-TENANT STARVATION
**Algorithm:** Filter corpus to the tenant *before* BM25 scoring (IDF/doc-freqs become tenant-local). Build per-tenant bm25s index lazily on first query, RAM-cache keyed by (identity(C), length(C)) signature for auto-invalidation, bound 200 tenants with eviction, F=k (no over-fetch needed), fallback to shared+post-filter on build failure.

**Why it fights Arabic bias:** In a bilingual library the small/new (often Arabic) tenant that *shares academic vocabulary* with a 97%-dominant tenant gets its relevant chunks outranked off the page (overlap collapses to 0.46). The per-tenant index restores exact retrieval to 1.0 — a **retrieval-availability fairness precondition** for everything downstream. *(Honesty caveat — see §5: the mechanism is anti-minority-sharing-vocabulary, not intrinsically anti-Arabic, and dense starves worse than sparse.)*

*(Optional fourth supporting layer for the dialect angle: safe boundary-guarded MSA→Iraqi rewriting serves the lowest-resource Arabic dialect — every rule wrapped in Arabic-letter negative look-behind/look-ahead `(?<![ء-ي])...(?![ء-ي])` so it can only fire on whole tokens. 0/21+0/18 corruption vs naive 21+10. This is dialect-level fairness one layer below retrieval, but its quality is NOT validated — see §5.)*

---

## 4. CONSOLIDATED CITABLE SOURCES (deduplicated)

**Status note:** 49 of these are web-verified with DOIs/arXiv in verified_sources.json; all appear in the four references.bib files.

### Score-distribution thresholding & calibration (the gate's lineage)
- Otsu (1979) — Threshold Selection from Gray-Level Histograms — IEEE TSMC 9(1):62-66 — DOI 10.1109/TSMC.1979.4310076
- Manmatha, Rath & Feng (2001) — Modeling Score Distributions for Combining Search Engine Outputs — SIGIR — DOI 10.1145/383952.384005
- Arampatzis, Kamps & Robertson (2009) — Where to Stop Reading a Ranked List — SIGIR — DOI 10.1145/1571941.1572031
- Arampatzis & Robertson (2011) — Modeling Score Distributions in IR — Inf. Retrieval 14(1):26-46 — DOI 10.1007/s10791-010-9145-5
- Penha & Hauff (2021) — Calibration and Uncertainty of Neural L2R for Conversational Search — EACL — DOI 10.18653/v1/2021.eacl-main.12 *(use ONLY for "rerankers imperfectly calibrated," NOT for the Arabic direction)*

### Adaptive / training-free gating (closest prior art)
- Wang, Wei & Ling (2025) — TARG: Retrieval as a Decision, Training-Free Adaptive Gating — arXiv:2511.09803 (closest label-free gating prior art; gates *whether to retrieve*, not post-retrieval per-tenant cutoff)
- Yan et al. (2024) — CRAG — arXiv:2401.15884
- Asai et al. (2024) — Self-RAG — ICLR — arXiv:2310.11511
- Jeong et al. (2024) — Adaptive-RAG — NAACL — DOI 10.18653/v1/2024.naacl-long.389
- AutoRAG-HP (2024) — arXiv:2406.19251 (bandit, needs reward — contrast: gate is reward-free)
- Zhou & Chen (2025) — Optimizing Retrieval for RAG via RL — arXiv:2510.24652

### RAG fairness (the bias framing — pre-verified, on-narrative)
- Hu et al. (2024) — No Free Lunch: RAG Undermines Fairness in LLMs — arXiv:2410.07589
- Zhang et al. (2025) — The Other Side of the Coin: Fairness in RAG (FairFilter) — arXiv:2504.12323
- da Silva de Oliveira et al. (2025) — Fairness Testing in RAG — arXiv:2509.26584
- Ruparel & Patel (2025) — Caching at Scale: Fairness in Multi-tenant RAG — SN Computer Science — DOI 10.1007/s42979-025-04467-3
- Singh & Joachims (2018) — Fairness of Exposure in Rankings — KDD; Biega, Gummadi & Weikum (2018) — Equity of Attention — SIGIR *(deliberate exposure-fairness, contrast with unintended starvation)*
- *(Lead, uncited:* Kim & Diaz — Towards Fair RAG — arXiv:2409.11598)

### Multi-tenant / filtered-ANN (starvation, dense-side cure)
- Jin et al. (2024) — Curator: Efficient Indexing for Multi-Tenant Vector DBs — arXiv:2401.07119
- Gollapudi et al. (2023) — Filtered-DiskANN — WWW — DOI 10.1145/3543507.3583552
- Patel et al. (2024) — ACORN — SIGMOD/PACMMOD — DOI 10.1145/3654923

### CLIR / cross-script
- Adeyemi et al. (2024) — Zero-Shot Cross-Lingual Reranking with LLMs for Low-Resource Languages — ACL Short — DOI 10.18653/v1/2024.acl-short.59 (in-language strongest)
- Macmillan-Scott, Goworek & Özyiğit (2025) — Generative Query Expansion with Multilingual LLMs for CLIR — arXiv:2511.19325 (cross-script especially hard)
- Pirkola (1998) — Dictionary-Based CLIR — SIGIR — DOI 10.1145/290941.290957; Pirkola et al. (2001) — Inf. Retrieval 4(3-4):209-230
- Lin et al. (2023) — Neural Ranking/Reranking Baselines for CLIR — arXiv:2304.01019
- Nie (2010) — Cross-Language Information Retrieval — Morgan & Claypool — DOI 10.2200/S00266ED1V01Y201005HLT008
- Feng et al. (2022) — LaBSE — ACL — DOI 10.18653/v1/2022.acl-long.62
- Zhang et al. (2023) — MIRACL (18 languages) — TACL — DOI 10.1162/tacl_a_00595
- *(Leads, uncited:* ArabicaQA/AraDPR arXiv:2403.17848; XOR-TyDi arXiv:2010.11856; NeuCLIRBench arXiv:2511.14758)

### Continual retrieval (contrast: they update weights, we update JSON)
- Son et al. (2026) — CREAM: Continual Retrieval on Dynamic Streaming Corpora — KDD'26 — arXiv:2601.02708

### Arabic & Iraqi-dialect NLP
- Antoun, Baly & Hajj (2020) — AraBERT — OSACT4/LREC — aclanthology 2020.osact-1.2 — arXiv:2003.00104
- Obeid et al. (2020) — CAMeL Tools — LREC — aclanthology 2020.lrec-1.868
- Abdul-Mageed, Elmadany & Nagoudi (2021) — ARBERT & MARBERT — ACL-IJCNLP — DOI 10.18653/v1/2021.acl-long.551
- Abdul-Mageed et al. (2021/2022/2023) — NADI Shared Tasks (incl. Iraqi) — WANLP — arXiv:2103.08466, 2210.09582, 2310.16117
- Mousi et al. (2024) — AraDiCE (EXCLUDES Iraqi) — arXiv:2409.11404
- Robinson et al. (2024) — AL-QASIDA (MSA-reversion; 8 varieties, not Iraqi) — arXiv:2412.04193
- Keleg, Goldwater & Magdy (2023) — ALDi (trained on AOC, no Iraqi) — EMNLP — arXiv:2310.13747
- Salloum & Habash (2012) — Elissa: Dialectal→MSA MT (inverse direction, incl. Iraqi) — COLING Demos
- Bouamor et al. (2018) — MADAR Corpus and Lexicon (25 cities + MSA) — LREC
- Habash (2010) — Introduction to Arabic NLP — Morgan & Claypool — ISBN 978-1-59829-795-9
- Zahir (2021) — IADD (216 Iraqi / 135,804) — Data in Brief 40:107777 — DOI 10.1016/j.dib.2021.107777
- Al-Jawad et al. (2022) — CIAD/IA2D Iraqi Twitter Corpus (1,673) — ITMO J. 22(2):308-316 — DOI 10.17586/2226-1494-2022-22-2-308-316
- Woodhead & Beene eds. (1967) — A Dictionary of Iraqi Arabic — Georgetown UP
- Bechiri & Lanasri (2026) — DziriBOT: RAG for Algerian Arabic Dialect — arXiv:2602.02270 (closest dialect-RAG comparator)

### Core RAG / retrieval / on-device infrastructure
- Lewis et al. (2020) — RAG — NeurIPS — arXiv:2005.11401; Guu et al. (2020) — REALM — ICML — arXiv:2002.08909; Izacard & Grave (2021) — FiD — EACL — DOI 10.18653/v1/2021.eacl-main.74; Gao et al. (2023) — RAG Survey — arXiv:2312.10997
- Robertson & Zaragoza (2009) — BM25 and Beyond — DOI 10.1561/1500000019; Lù (2024) — BM25S — arXiv:2407.03618
- Karpukhin et al. (2020) — DPR — EMNLP — DOI 10.18653/v1/2020.emnlp-main.550; Khattab & Zaharia (2020) — ColBERT — SIGIR — DOI 10.1145/3397271.3401075; Izacard et al. (2022) — Contriever — TMLR — arXiv:2112.09118
- Nogueira & Cho (2019) — monoBERT — arXiv:1901.04085; Bonifacio et al. (2021) — mMARCO (the reranker's training data) — arXiv:2108.13897; Wang et al. (2020) — MiniLM — NeurIPS — arXiv:2002.10957
- Malkov & Yashunin (2020) — HNSW — IEEE TPAMI — DOI 10.1109/TPAMI.2018.2889473; Johnson, Douze & Jégou (2021) — FAISS — IEEE TBD — DOI 10.1109/TBDATA.2019.2921572
- Cormack, Clarke & Büttcher (2009) — Reciprocal Rank Fusion — SIGIR — DOI 10.1145/1571941.1572114
- Gao et al. (2023) — HyDE — ACL — DOI 10.18653/v1/2023.acl-long.99; Sarthi et al. (2024) — RAPTOR — ICLR — arXiv:2401.18059
- On-device: Wang & Chau (2024) — MeMemo — SIGIR — DOI 10.1145/3626772.3657662; Park, Lee & Kim (2025) — MobileRAG — arXiv:2507.01079; Seemakhupt et al. (2024) — EdgeRAG — arXiv:2412.21023

### Model artifacts (citable)
- cross-encoder/mmarco-mMiniLMv2-L12-H384-v1; paraphrase-multilingual-MiniLM-L12-v2 (Reimers & Gurevych); UBC-NLP/MARBERT

### Author's own prior work
- Yousef (2026) — flagship "Self-Calibrating Per-Tenant Relevance Gating…" — Zenodo concept DOI 10.5281/zenodo.20688577; ORCID 0009-0006-7409-9367; AlSafwa University, Karbala, Iraq
- Yousef (2026) — P1 Tenant Starvation; P2 Bilingual Glossary; P3 Iraqi Dialect Rewriting
- PATENT_DISCLOSURE.md (author's own technical doc — **must be reworded before citing**, see §5)

### Patents (prior-art landscape, NOT label-free per-tenant calibration)
- US 8,924,378 (Cramer/Surf Canyon); US 11,314,794 (Shen/ITRI); US 12,561,314 (Brenner & Seyler/Goldman Sachs, RAG optimization)

### Datasets
- IADD (135K texts; 216 Iraqi); IA2D/CIAD (1,673 Iraqi tweets); Georgetown Dictionary of Iraqi Arabic

---

## 5. CONTRADICTIONS TO HANDLE HONESTLY

1. **"Arabic scores lower" is FALSE and must be purged.** Our data shows Arabic relevant means slightly *higher* (CE 10.53 vs 10.45; cosine 0.805 vs 0.754). The flagship abstract/intro, the plain-language file, PATENT_DISCLOSURE.md (lines 37, 170), INVENTOR_BRIEF.md, and relevance_gate.py docstring (line 12) all still assert the refuted claim — **reword all of them to score COMPRESSION** before citing. No "tau=0.52" appears in paper_final.md or the plain-language file (it was an older draft); cutoffs are negative logits (−1.39…−3.29) and cosines 0.302…0.409.

2. **Fix the causal claim, keep the outcome.** Source docs say Arabic gets a higher cutoff *because Arabic scores lower*. Correct mechanism: Arabic's band is *compressed/tighter*, so the percentile-gap geometry lands higher. Re-derive the higher-Arabic-cutoff result from std-compression, not a raw-level offset. The flagship's Sec 4.3 "M assigns Arabic HIGHER raw scores" framing is a *tension* to clean up, not a hard error.

3. **Compensation is PARTIAL and SCORER-DEPENDENT — not a universal law.** Only pure-Arabic T3 is strictest in *both* score spaces. Mixed AR+EN T4 is cross-encoder-strictest (−1.39) but cosine-*loosest* (0.302). State the honest claim: T3 high in both; mixed rank-flips. Do not sell "Arabic = always strictest."

4. **Cutoff numbers vary by run.** Canonical (paper) AR −1.59/−1.39 vs EN −2.75/−2.93/−3.29; disclosure run AR −1.79/−1.27 vs EN −3.31/−2.75; single-tenant examples −3.24 and −2.34. Same direction, different absolutes. **Pick the canonical run; footnote run-to-run variation.**

5. **Glossary's net benefit over dense-alone is NIL at book-level.** Both MiniLM and LaBSE already cross scripts at recall 1.0 unaided. So "dense multilingual retrieval fails Arabic" is FALSE at book-level. The glossary's value is **cost (~48 µs, no model), the lexical-half fusion contribution, interpretability, and the continual loop** — not beating dense retrieval. Scope dense-failure claims to *compression/discriminability*, never to cross-script recall.

6. **Cross-script disadvantage is one-directional.** EN→AR is already 1.0 *without* glossary (Arabic technical books embed English terms verbatim). The burden falls on the AR→EN direction only — report the asymmetry, don't average it away.

7. **Tenant starvation is NOT intrinsically anti-Arabic.** It is driven by *vocabulary overlap*, not language (distinct-vocab minority stays flat 0.91–0.97). And **dense starves worse than sparse** (0.322 < 0.461) — so dense does not "fix" the minority-tenant problem; only the per-tenant sub-index (or in-traversal filtered-ANN) does. Frame starvation as a fairness *precondition* harming minority-vocab-sharing (often Arabic) tenants, not as an Arabic-bias mechanism per se.

8. **The F1 0.968 headline is NOT Arabic-specific.** It's a 60IN/60OUT probe pooled across all 5 tenants on a small, English-dominated 10-book corpus; recall pins to 1.0 because IN queries are easy passage-derived self-matches with the source left in the pool. **The Arabic-fairness angle rides on Exp1 (cutoff geometry), not on the F1 number.**

9. **The ALDi result is NEGATIVE — do not read it as rewriter failure.** 0.55→0.49 reflects ALDi's *lack of Iraqi coverage*, not poor rewriting. The dialect paper claims **safety/coverage/stability only — NOT dialect-quality improvement.** No validated automatic metric currently shows the Iraqi rewriter improves dialectness. Do not overclaim.

10. **Scope caveats to carry everywhere.** All results are single-CPU-machine micro-benchmarks with tiny n (40/40 bias; 6/3/3 glossary; 15 sentences dialect; 18 starvation probes). Live Arabic data volume is modest (62 AR-dominant chunks; the one genuinely-Arabic book = 45 chunks; several "Arabic" books are Arabic-titled but English-bodied). These are **direction/mechanism results, not population estimates.** The self-match anchor is a near-identical query/passage pair (upper bound on relevance), a documented deliberate design choice, not a real query distribution — flag it before a reviewer does.

11. **Cosine irrelevant-band caveat.** Cosine cross-document std is actually *smaller* for English (AR 0.144 vs EN 0.097). The compression story is clean for the **relevant/self-match** distribution; state the compression claim specifically about the self-match band, not uniformly "Arabic always tighter."

---

## 6. SUGGESTED SECTION OUTLINE

### Scientific version
1. **Abstract** — bias = score compression (not lower scores); 1.27x tokenization penalty; three frozen-model engineering mitigations; emergent fairness without retraining.
2. **Introduction** — multilingual RAG for an offline Arabic/Iraqi education platform; the fairness question; thesis = engineering property, not training.
3. **Related Work** — RAG fairness (Hu/Zhang/da Silva); score-distribution thresholding (Otsu/Manmatha/Arampatzis); training-free gating (TARG/CRAG/Self-RAG); CLIR & cross-script (Adeyemi/Macmillan-Scott/Pirkola); multi-tenant/filtered-ANN (Curator/Filtered-DiskANN/ACORN); Arabic & Iraqi NLP (AraBERT/CAMeL/NADI/AraDiCE/AL-QASIDA/ALDi/DziriBOT).
4. **Measuring the bias** — methodology (self-match & cross-doc anchors, reranker's own tokenizer, deterministic sampling, n=40); results = compression std table + tokenization fertility + "Arabic is NOT lower" + shrunken separation. **Lead with the refutation of "Arabic scores lower."**
5. **Mitigation 1: self-calibrating per-tenant gate** — Algorithm 1; Sec 4.3 absorption argument; Exp1 per-tenant cutoffs (canonical run, both score spaces); in_lo cross-check; gate F1 (clearly scoped as pooled/non-Arabic-specific); alpha ablation w/ clamp caveat.
6. **Mitigation 2: continual cross-script glossary** — additive expansion; continual one-shot acquisition; AR→EN 0→1; size ablation; dense baselines + honest "net benefit nil at book-level / value is cost+fusion+continual."
7. **Mitigation 3: per-tenant BM25 sub-index** — starvation diagnosis; vocabulary-overlap negative control; crowding/statistics-capture decomposition; dense-also-starves; restoration to 1.0. Honest "not intrinsically anti-Arabic."
8. **Iraqi dialect layer (optional)** — boundary-guarded safe rewriting; corruption table; ALDi negative result as resource-gap evidence; safety/coverage claims only.
9. **Discussion** — fairness as emergent engineering property; partiality & scorer-dependence; on-device/label-free significance for Karbala students.
10. **Limitations & Threats to Validity** — every item in §5: micro-benchmark scope, tiny n, self-match anchor, partial/scorer-dependent compensation, one-directional cross-script, F1 not Arabic-specific, ALDi negative, modest Arabic data volume.
11. **Conclusion.**

### Plain-language version (template: PLAIN_LANGUAGE_for_everyone.md — first-person, single-author, everyday analogies, explicit honesty section)
1. **The problem in one sentence** — "The AI is not unfair to Arabic by scoring it lower — it's unfair by being *fuzzier* about Arabic, squeezing all its judgments into a narrow range so it can't tell good from bad as clearly."
2. **Why this happens** — subword fragmentation analogy (Arabic chopped into 1.27x more pieces); "the ruler has fewer marks for Arabic."
3. **Three fixes, no retraining** — (a) the gate that "sets its own bar per library by watching its own scores" → Arabic libraries automatically get a stricter, fairer bar; (b) the growing bilingual dictionary that lets an Arabic question find the right English book (and *learns new words once, forever*); (c) the per-library shelf so a small Arabic library isn't buried under a giant one.
4. **The Iraqi-dialect layer** — serving the most overlooked Arabic dialect; "even the tools meant to measure Iraqi don't know Iraqi."
5. **Where I'm honest about the limits** — small tests on one laptop; the fix is partial and depends on the scorer; modern dense search already crosses scripts (the dictionary's value is speed/transparency/learning); I do NOT claim the dialect rewriter produces *better* Iraqi, only *safe* Iraqi; I do NOT claim "Arabic always scores lower."
6. **Why it matters** — a student in Karbala on a modest offline laptop gets fairer Arabic retrieval as a property of careful engineering, free and on-device.

---

**Single most load-bearing asset:** `papers/arabic_bias/experiments/results/exp_arabic_bias.json` (+ `exp_arabic_bias.py`) — the reproducible source of every corrected number and the deterministic methodology to cite verbatim. **Single most important honesty action:** purge "Arabic scores lower"/tau=0.52 everywhere (flagship abstract, plain-language file, PATENT_DISCLOSURE.md, INVENTOR_BRIEF.md, relevance_gate.py docstring) and replace with the score-compression framing, while keeping the still-valid higher-Arabic-cutoff result.