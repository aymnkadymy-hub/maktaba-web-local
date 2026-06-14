# Algorithmic Bias Against Arabic: A Measurement, and Three Frozen-Model Mitigations, in an Offline Multilingual Retrieval System

**Ayman Kazim Yousef**

Department of Artificial Intelligence Engineering, AlSafwa University, Karbala, Iraq

ORCID: 0009-0006-7409-9367 · ai25009@student.alsafwa.edu.iq

*Companion to the Maktaba (al-Maktaba al-Natiqa) offline multilingual RAG series. Experiment scripts and result files are released for reproduction.*

---

## Abstract

A widespread intuition about multilingual retrieval is that systems are unfair to lower-resource languages such as Arabic because they *score Arabic content lower*. We measured this directly on a deployed, fully offline, on-device multilingual digital-library system and found that the intuition is, in its most common form, wrong — and that the truth is more actionable. Using the production reranker's own tokenizer and scorer over our corpus, Arabic relevant query-passage pairs do **not** score lower than English ones; if anything they score *slightly higher* (cross-encoder self-match mean 10.534 for Arabic vs. 10.448 for English; cosine 0.805 vs. 0.754). The actual bias is **score compression**: the shared multilingual scorer crams Arabic relevant-pair scores into a much narrower band than English (cross-encoder self-match standard deviation 0.348 for Arabic vs. 0.909 for English, ≈2.6× tighter; cosine 0.062 vs. 0.128, ≈2.1× tighter; n = 40 per language). Because the irrelevant ("cross-document") band sits closer for Arabic too (cross-encoder −6.69 vs. −7.673), the *separation* between relevant and irrelevant collapses. A global acceptance cutoff tuned on English's wide score geometry is therefore mis-placed for Arabic, even though Arabic is not "scored lower." This compression co-occurs with a measured 1.27× subword fragmentation penalty (Arabic 1.946 vs. English 1.536 tokens per word in the reranker's own tokenizer), consistent with the general tokenization-unfairness literature [4, 36, 38]. An expanded, statistically-tested replication (n = 309 Arabic chunks from five sources, English balanced to the same n, three frozen scorers) confirms that the relevant-band *spread* differs significantly between Arabic and English on **every** model (Brown–Forsythe/Levene and Fligner–Killeen p < 0.01), but shows the *direction and magnitude* are model- and extraction-quality-dependent: the compression direction holds cleanly on LaBSE (std-ratio 1.22, bootstrap 95% CI [1.06, 1.41]) and on cleanly-chunked Arabic, while a noisier raw-PDF expansion widens the cross-encoder band. We therefore frame the robust finding as a **significant, language-dependent score-geometry mismatch** that mis-places a global cutoff — whose **direction is unstable**: in a small single-book Arabic sample it looked like *compression* (Arabic narrower), but on a clean, larger, matched-pipeline sample (n = 390) the deployed cross-encoder makes Arabic significantly *wider and lower*, while LaBSE shows no difference (§7.9). We report full significance tests. The per-tenant gate corrects the mismatch regardless of its direction.

We then show that this specific failure can be corrected **without any model retraining**, as an emergent property of three frozen, on-device mechanisms. (i) A *self-calibrating per-tenant relevance gate* reads each tenant's own score geometry from label-free self-match and cross-document anchors and places a tenant-matched cutoff; because Arabic-heavy tenants exhibit a compressed band, the same percentile-gap rule automatically hands them a stricter, geometry-matched bar (the two Arabic-containing tenants are the two strictest cross-encoder cutoffs; the pure-Arabic tenant is strictest in cosine), with no language-specific code and no labels. On a 60-in/60-out probe the gate reaches F1 0.968 (precision 0.938, recall 1.000) versus 0.828 for a fixed default, removing 21 of 25 off-topic acceptances. (ii) A *continual self-extending cross-script glossary* lifts the structurally zero Arabic→English lexical-match floor from book-level BM25 recall 0.00 to 1.00 and acquires held-out technical terms one-shot with no weight updates, at ≈48 µs per query and no model; we report the honest caveat that a dense multilingual encoder already reaches the same book-level recall, so the glossary's net benefit over dense retrieval is essentially cost and interpretability, not recall. (iii) A *per-tenant BM25 sub-index* prevents a dominant tenant from starving a small minority (often Arabic) tenant that shares academic vocabulary, restoring minority overlap@5 from 0.461 to 1.0; we note that dense retrieval starves *worse* (0.322), so this is a retrieval-availability precondition, not a sparse-specific quirk.

All results are single-machine CPU micro-benchmarks on a small, English-dominated corpus (63 Arabic vs. 3,760 English chunks), with deliberately near-identical self-match anchors and small per-condition samples; they characterize *direction and mechanism*, not population effect sizes. We state explicitly what we do and do not claim, and we release the experiment scripts and result files for reproduction. The contribution we stand behind is conceptual as much as numerical: **Arabic fairness here is achieved as an engineering property of frozen, on-device models — not as a model-training result.**

---

## 1. Introduction

Retrieval-augmented generation (RAG) is now the default architecture for question answering over private document collections [15, 17, 24]. A RAG system that serves a multilingual user base inherits a fairness obligation that goes beyond generation quality: if the *retrieval* stage systematically surfaces worse evidence for one language, every downstream answer for that language is degraded before the language model ever runs. This is the setting of the present paper. We operate a fully offline, on-device multilingual digital-library platform built for students at AlSafwa University in Karbala, Iraq — a deployment where the users read and ask questions in both Arabic and English, where the hardware is a modest CPU laptop with no cloud fallback, and where retraining a multilingual reranker on-device is simply not an option. The practical question we faced was direct: *is our system unfair to Arabic, and if so, can we fix it without retraining the models we cannot retrain?*

The folk answer — and the framing of an earlier internal draft of this very project, which we now retract — is that multilingual scorers are unfair to Arabic because they *assign Arabic lower relevance scores*. When we measured it, this turned out to be false on our corpus. Arabic relevant pairs do not score lower than English relevant pairs; in both our scorers they score slightly higher (Section 7). The real, measurable harm is different and, we will argue, more tractable: the shared multilingual scorer is **less discriminative** for Arabic. On our production corpus it compresses Arabic relevant-pair scores into a band roughly 2.1–2.6× narrower than English's, and compresses the irrelevant band toward that same region, so the gap a threshold must exploit — the separation between "relevant" and "irrelevant" — shrinks for Arabic. A single global acceptance threshold, calibrated (implicitly or explicitly) on English's wide score geometry, is then placed in the wrong location for Arabic content. The bias is real, but it lives in the **geometry of the score distribution**, not in its mean. On our production corpus this harm takes the form of **score compression** — Arabic's relevant scores packed into a band ~2–2.6× narrower than English's. The deeper, model-independent spine we hold to throughout is that the bias is a **significant, language-dependent score-geometry mismatch** — whose *direction is not fixed*. An expanded three-model analysis (§7.8) and a definitive clean, matched-pipeline replication (§7.9) show the geometry difference is statistically significant but **flips with corpus homogeneity, genre, and model**: Arabic looks compressed in a small single-book sample, yet is significantly *wider and lower* on the deployed reranker over a large diverse one. The constant is not a direction but a **requirement**: because the geometry cannot be predicted from the language, the cutoff must be *measured per tenant*, not assumed.

Why does the distinction matter? Because it changes what a fix must do. If Arabic scored lower, one might argue for additive score offsets, language-conditioned reweighting, or — most naturally — fine-tuning the encoder on more Arabic data. But Arabic does not score lower, so none of those address the actual defect; and in our deployment we cannot fine-tune anything regardless. Compression, by contrast, is a *per-distribution* problem with a *per-distribution* cure: if the system can read the score geometry of each collection and place its decision boundary to match that geometry, then a compressed Arabic band is handled correctly for the same reason an expanded English band is — not because the system was told which language it is looking at, but because it calibrated to what it saw. This is the conceptual move we make. We treat fairness not as something to be trained into a model, but as a property that can **emerge** from a frozen model when the surrounding retrieval machinery is built to be distribution-aware. We co-locate this scoring analysis with a second, structural source of Arabic disadvantage — cross-script lexical mismatch, where an Arabic query can never lexically match an English passage and BM25 recall is zero by construction — and a third — minority-tenant starvation, where a small Arabic library that shares vocabulary with a large dominant one is ranked off the page. Each has a frozen-model remedy, and we present all three.

We also take pains to be honest about scope, because the temptation to oversell an Arabic-fairness result is real and the data do not support an oversell. Our corpus is small and English-dominated; the Arabic sample is 63 chunks; our anchors are deliberately near-identical query-passage pairs that upper-bound relevance rather than sampling a real query stream; everything runs on one CPU. These are direction-and-mechanism findings, not population estimates, and we say so wherever a number appears. We mirror the candid "what we do and do not claim" posture of our prior work on this platform.

### 1.1 Terminology

We fix the vocabulary used throughout the paper.

- **Tokenization fertility.** The mean number of subword tokens a tokenizer emits per whitespace word. A fertility of 1.946 for Arabic versus 1.536 for English (measured with the reranker's *own* tokenizer on our corpus) means Arabic words are fragmented into 1.27× as many pieces. Higher fertility spends more of a fixed context budget and more of the model's positional/representational capacity on the same semantic content, the unfairness studied generally by Petrov et al. [36], Ahia et al. [4], and Rust et al. [38]. We use the term strictly for this *tokenization-stage* property and do not conflate it with downstream scoring behavior.

- **Score geometry.** The shape — location, spread, and tail structure — of the distribution of scores a fixed scorer assigns to a defined population of pairs (e.g., relevant pairs within one collection). Two distributions can share a mean yet differ sharply in geometry.

- **Score compression.** The specific geometric distortion at the heart of this paper: for a given scorer, the relevant-pair score distribution for one language occupies a *narrower* range than for another, reducing the dynamic range available to separate relevant from irrelevant. We quantify it by the standard deviation of the self-match (relevant) score distribution: cross-encoder 0.348 (Arabic) vs. 0.909 (English); cosine 0.062 vs. 0.128. Compression is a statement about *spread*, explicitly not about *level*.

- **Separability (relevant–irrelevant separation).** The usable gap between the relevant and irrelevant score bands of a population. Even when Arabic relevant scores are slightly higher, separability is lower for Arabic because both bands are compressed toward each other (irrelevant cross-encoder mean −6.69 for Arabic vs. −7.673 for English). Low separability is what makes a mis-placed global threshold costly.

- **Per-tenant calibration.** A *tenant* is an isolated per-user (per-library) partition of the corpus; the system enforces strict tenant isolation. Per-tenant calibration is the act of choosing a tenant-specific acceptance threshold from that tenant's own score geometry, rather than applying one global cutoff to all tenants. It is *label-free*: no relevance judgments are required.

- **Cross-script retrieval.** Retrieval where query and passage are written in different scripts (here, Arabic-script query against Latin-script English passage, or vice versa). Lexical (BM25) overlap across scripts is structurally zero absent a bridge, independent of any model quality — a floor distinct from the scoring-compression effect.

- **Frozen-model mitigation.** A mechanism that improves a fairness or quality outcome *without updating any model weights*: the cross-encoder, the bi-encoder, and any language model remain exactly as shipped. All three mitigations in this paper are frozen-model mitigations; they live in the retrieval/decision layer, are auditable, and are reversible.

### 1.2 Contributions

This paper makes the following contributions.

- **A corrected measurement of Arabic retrieval bias.** Using the production reranker's own tokenizer and scorer over our corpus, we show that Arabic relevant pairs are *not* scored lower than English (cross-encoder self-match mean 10.534 vs. 10.448; cosine 0.805 vs. 0.754), and that the real bias is a **significant language-dependent score-geometry mismatch** — on the clean production corpus a ≈2.6× tighter cross-encoder band and ≈2.1× tighter cosine band for Arabic (§7.2), i.e. *score compression* — which collapses relevant–irrelevant separability and mis-places any global threshold. An expanded, three-model, significance-tested replication (§7.8) confirms the spread differs significantly on every model (Levene/Fligner p < 0.01) while showing the magnitude is sample- and model-dependent, so we do not claim a universal compression ratio. We explicitly retract the "Arabic scores lower" framing as refuted by our own data.

- **A frozen-model, label-free per-tenant relevance gate that corrects compression.** We give an algorithm that reads each tenant's score geometry from self-match and cross-document anchors and places a tenant-matched cutoff with no language-specific logic. Because Arabic-heavy tenants present a compressed band, the same rule automatically assigns them stricter, geometry-matched cutoffs (the two Arabic-containing tenants are the strictest cross-encoder cutoffs; the pure-Arabic tenant is strictest in cosine), and we show via the tenants' in-domain anchors that the higher Arabic cutoff arises from a *tighter band, not lower scores*. The gate reaches F1 0.968 (precision 0.938, recall 1.000) versus 0.828 for a fixed default, approaching label-tuned oracles it never sees (Section 7).

- **A continual self-extending cross-script glossary that removes the structural Arabic→English lexical floor.** Additive cross-script query expansion lifts book-level Arabic→English BM25 recall from 0.00 to 1.00 and acquires held-out technical terms one-shot with no weight updates, at ≈48 µs per query and no model at query time. We pair this with the honest finding that a dense multilingual encoder already reaches the same book-level recall, so the glossary's net contribution over dense retrieval is cost, interpretability, and tenant-specific coverage — not recall (Section 7).

- **A per-tenant BM25 sub-index that prevents minority-tenant starvation.** We diagnose a mechanism by which a dominant tenant starves a small, vocabulary-sharing minority (often Arabic) tenant — minority overlap@5 collapsing from 0.913 to 0.461 as dominance grows — and show that a per-tenant sub-index restores it to 1.0. We report that dense retrieval starves *worse* (0.322), establishing this as a retrieval-availability fairness precondition rather than a sparse-only artifact (Section 7).

- **A reusable framing and a released artifact.** We argue, and instantiate, the thesis that *Arabic fairness can be an emergent engineering property of frozen, on-device models rather than a training outcome*, and we release the experiment scripts and result files so every number in this paper can be reproduced.

### 1.3 What we do and do not claim

We are explicit about the boundaries of these results, in the candid style of our prior work on this platform.

**We claim** that, on the corpus and scorers we measured, the bias against Arabic is a **significant, language-dependent score-geometry difference whose direction is not stable**. In our first, small, single-book sample Arabic looked *compressed* (band 2.1–2.6× tighter; equal-or-slightly-higher mean) — but a definitive clean, matched-pipeline replication on 390 diverse Arabic chunks (§7.9) overturns this: the deployed cross-encoder makes Arabic significantly *wider and lower*, MiniLM wider, and LaBSE shows no difference. The robust, significance-tested claim is therefore a **direction-unstable language-dependent geometry mismatch** (neither 'Arabic compressed' nor 'Arabic lower' is a transferable law), and the engineering claim a **significant language-dependent geometry mismatch** (the v1 'compression' reading proving to be a small-single-book-corpus artifact — on a clean, larger, matched-pipeline sample the direction reverses on the deployed reranker, §7.9), not a fixed-direction effect. We claim that this specific defect can be corrected by **frozen-model, label-free mechanisms** that calibrate to each collection's score geometry, and that under such calibration Arabic-heavy collections receive an appropriately stricter, geometry-matched cutoff as an *emergent* consequence of the score geometry rather than any language-specific rule. We claim the reported experimental numbers exactly as measured.

**We do not claim** that Arabic is scored lower — we specifically refute that claim, including in an earlier draft of our own. We do not claim a *trained* model improvement of any kind: no weights are updated anywhere, and no result here should be read as a statement about retraining. We do not claim population-level effect sizes: every result is a single-CPU micro-benchmark on a **small, English-dominated corpus** (63 Arabic vs. 3,760 English chunks; per-condition n on the order of tens), using deliberately near-identical self-match anchors that *upper-bound* relevance rather than sampling a real query distribution — these are direction-and-mechanism findings only. We do not claim to beat label-tuned oracles: the gate approaches but does not exceed oracle thresholds that require relevance labels it never sees. We do not claim that the cross-script glossary beats dense multilingual retrieval on book-level recall — it does not; its advantage is cost, transparency, and tenant-specific terms, and we state this plainly. And we do not claim that minority-tenant starvation is intrinsically anti-Arabic: it is driven by *shared vocabulary under tenant dominance*, harms whichever minority shares vocabulary (often the Arabic library in our setting), and is in fact *worse* for dense retrieval than for sparse. Finally, the tokenization-unfairness papers we cite [4, 36, 38] study many languages and report no Arabic-specific headline figure; we use them as the general evidence base and treat our measured 1.27× fertility as our own corpus-specific result, not as a downstream-scoring claim.


## 2. Related Work

Our contribution sits at the intersection of five threads: multilingual representation and the *curse of multilinguality*; tokenization unfairness and subword fertility; cross-lingual information retrieval and reranker (mis)calibration; fairness in information retrieval and retrieval-augmented generation (RAG); and Arabic natural-language processing. We review each in turn, and in Section 2.6 we state precisely where prior art stops and where our measurement-and-mitigation contribution begins. Two framings recur in this literature and we flag them now, because our results turn on the difference. The first is that bias against a lower-resource language manifests as *lower scores* — relevant content scoring worse, ranked lower, retrieved less often. The second, which our measurements support and prior work has rarely isolated, is that bias can instead manifest as *reduced discriminability* — a compressed score range in which relevant and irrelevant content are scored almost as highly, so a global decision boundary tuned on a higher-resource language is mis-placed. Much of the work below documents the conditions under which one or both arise; none, to our knowledge, separates the two for Arabic on a frozen multilingual reranker and then mitigates the second without retraining.

### 2.1 Multilingual representation and the curse of multilinguality

A single transformer encoder pre-trained on many languages shares one parameter budget and one subword vocabulary across all of them. Conneau et al. [11], in the XLM-R study, named the resulting trade-off the *curse of multilinguality*: for a fixed model capacity, adding languages improves low-resource transfer up to a point and then degrades per-language quality, because each language receives a smaller effective share of capacity and vocabulary. Wu and Dredze [41] examined multilingual BERT layer by layer and found that cross-lingual transfer is real but markedly uneven across languages, with lower-resource and morphologically richer languages systematically under-served by the shared representation. This is the architectural backdrop to our setting: the reranker we deploy is a compact distilled multilingual MiniLM cross-encoder [39] fine-tuned on multilingual MS MARCO [10], and its embedding counterpart is a multilingual paraphrase MiniLM — both are exactly the kind of capacity-shared, vocabulary-shared model the curse of multilinguality describes.

The prior work in this thread establishes that shared multilingual capacity is unequally allocated, and it predicts that a lower-resource language such as Arabic will be represented less faithfully than English. What it does not do is characterize the *shape* of that under-representation at the scoring stage of a retrieval pipeline. The curse-of-multilinguality literature is largely about downstream task accuracy and probing; it does not ask whether the residual cost shows up as a downward shift of relevant scores or as a contraction of the score range. Our measurement (Section 7) answers that question for one frozen model on one corpus: on this corpus the cost appears as *compression* of the relevant-pair score band rather than a downward shift (the magnitude and direction are model- and quality-dependent; §7.8), and we trace the consequence through to a mis-placed global cutoff.

### 2.2 Tokenization unfairness and fertility

A growing body of work locates a concrete, measurable source of multilingual inequity at the very first stage of the pipeline — subword tokenization. Rust et al. [38] showed that a language-dedicated tokenizer substantially outperforms the tokenizer of a multilingual model on that language, and quantified the gap with *fertility*, the average number of subword tokens a tokenizer emits per word: a higher fertility means a language's words are shattered into more, less meaningful pieces under a vocabulary that was not designed for it. Ahia et al. [4] extended this to the economics of inference, showing that languages with higher fertility are tokenized into far more units for the same content, so users of those languages pay more — in tokens, latency, and money — for identical information; they document large cross-language disparities in tokens-per-text. Petrov et al. [36] systematized the phenomenon across many tokenizers and languages, demonstrating that the same sentence can require several times as many tokens in some languages as in others, and framing this *tokenization unfairness* as a structural disadvantage baked into shared vocabularies before any model weight is consulted.

Two honesty points govern how we use this thread. First, these studies survey Arabic *among many languages*; none reports an Arabic-specific headline fertility number for a corpus and tokenizer like ours, so we cite them as the general evidence base for the mechanism and report our own measured figure — a fertility of 1.946 tokens per word for Arabic versus 1.536 for English under the reranker's *own* tokenizer on our corpus, a 1.27x fragmentation penalty — as our result, not theirs. Second, and more importantly for the thesis: this prior work establishes that tokenization is *upstream* and unfair, but it does not connect that upstream fragmentation to a *downstream* scoring geometry. We are careful not to conflate the two. Tokenization fragmentation is one plausible contributor to the score compression we measure at the reranking stage, but the compression is the phenomenon the gate must contend with, and the fertility number is supporting evidence for *why* compression is plausible — not a substitute for measuring it. We do not claim the gate compensates for tokenization unfairness directly; it compensates for the score geometry that emerges downstream.

### 2.3 Cross-lingual IR and reranker (mis)calibration

When query and document are in different languages, cross-lingual information retrieval (CLIR) must bridge the gap [29], either by translating one side or by relying on a shared multilingual representation. Reproducible multi-stage CLIR baselines establish the retrieve-then-rerank recipe across languages [25], and zero-shot cross-lingual reranking with large language models has been shown to be most effective *in-language*, only competitive cross-lingually for the most capable multilingual models [3] — evidence that a cross-lingual step is not free and is best taken deliberately rather than always. On the dense side, learned dual-encoder retrieval [22] and unsupervised contrastive dense encoders [20], indexed for approximate nearest-neighbour search [21, 27] and fused with the lexical signal via Reciprocal Rank Fusion [12], make multilingual dense retrieval practical, and sentence-level multilingual encoders such as LaBSE [14] align scripts well enough that an Arabic query can retrieve an English passage with no explicit translation. The fused candidates are reranked by a cross-encoder that scores a query–passage pair jointly in the monoBERT style [30], the stage at which our measurements are taken.

The calibration sub-thread is the one most directly relevant to our thesis. Neural rerankers are known to be poorly calibrated: Penha and Hauff [35] show that BERT-based learning-to-rank models do not produce reliably comparable scores and argue for uncertainty-aware ranking. We invoke this result for one specific claim only — that a raw cross-encoder logit cannot be compared against a single universal constant to decide relevance — and *not* for any statement about the Arabic direction, which Penha and Hauff do not study. The CLIR literature documents that cross-lingual retrieval is harder and that in-language signals are stronger; the calibration literature documents that reranker scores are not directly thresholdable. Neither, however, measures *how the score distribution differs between languages on the same frozen model*: whether Arabic relevant pairs score lower than English relevant pairs, or whether they occupy a narrower band. That distinction is the empirical gap we fill, and our finding on the clean production corpus — Arabic relevant pairs score *slightly higher* in mean but with roughly 2–2.6x tighter spread than English (§7.2; an expanded three-model test in §7.8 shows the *spread* difference is statistically significant on every model while the direction is model-dependent) — is precisely the kind of distributional fact that the calibration literature predicts could exist but has not, for Arabic, measured.

### 2.4 Fairness in IR and RAG

A recent line of work asks whether retrieval augmentation helps or harms fairness. Hu et al. [19] show that injecting retrieved context can silently *undermine* the fairness of an LLM's outputs, even for users who are deliberately careful — fairness is degraded by the retrieved evidence itself, not by the base model. Zhang et al. [43] propose FairFilter, a post-retrieval mechanism that removes biased retrieved content, and report that the problem is most acute for the small-scale local LLMs that on-device systems like ours run. da Silva de Oliveira et al. [13] use metamorphic testing with demographic perturbations to reveal that small input changes surface latent bias in small-language-model RAG. These works frame fairness as a property of *what content is admitted into the generation context* and propose filtering it. That framing is adjacent to ours — we, too, filter weakly relevant content with a post-retrieval gate — but the fairness object differs: they target social/demographic bias in the *content* of retrieved passages, whereas we study *language* bias in the *scoring geometry* that decides whether a passage is admitted at all.

Two further fairness traditions bound our claims. In ranking, deliberate fairness-of-exposure objectives reallocate attention across items or groups; that line concerns *intended* fairness interventions on a ranked list, whereas the disadvantage we study is an *unintended* artifact of a frozen scorer's geometry, and our mitigation restores discriminability rather than redistributing exposure. And on multi-tenant systems, the concern that a shared index can disadvantage some tenants relative to others is recognized in the systems literature, but typically as a resource- or latency-fairness question rather than a retrieval-quality one. We are explicit about what we do *not* claim here: although our gate prunes low-relevance context — exactly the intervention the RAG-fairness papers advocate — we measure precision and discriminability, not social fairness, and we do not claim a fairness-of-outputs result. Our use of the word *fairness* is confined to *retrieval fairness across languages*: equal ability to separate relevant from irrelevant content regardless of the query language.

### 2.5 Arabic NLP

The Arabic side of the system rests on established Arabic NLP infrastructure. AraBERT [5] provides a canonical Arabic transformer encoder of the kind that underpins the Arabic competence of multilingual rerankers; ARBERT and MARBERT [1] extend this to Modern Standard and dialectal Arabic respectively; and CAMeL Tools [31] offers an open-source, offline-friendly stack for Arabic preprocessing and morphology. Habash's monograph [18] documents the linguistic properties — rich templatic morphology, clitic agglutination, optional diacritization, orthographic variation — that make Arabic words expand into many subword units, connecting this thread back to the fertility mechanism of Section 2.2. On the dialect axis, the NADI shared tasks [2] establish nuanced, country- and province-level Arabic dialect identification including Iraqi, while resources such as the ALDi dialectness metric [23] are built on dialect corpora that under-represent or omit Iraqi entirely — a low-resource-within-low-resource gap that motivates our most cautious claims. The closest comparable *system* is DziriBOT [9], a RAG conversational agent for a low-resource Arabic dialect.

This infrastructure makes Arabic retrieval possible but does not, on its own, diagnose or repair a *cross-language scoring* disadvantage. The Arabic-NLP thread contributes encoders, tokenizers, dialect resources, and one comparable dialect-RAG system; none of these measures the score-compression bias of a frozen multilingual reranker, and none proposes a per-tenant, label-free, on-device mitigation for it. Our work is complementary: we contribute neither a new Arabic model nor a new dialect agent, but a measurement of how an existing frozen multilingual scorer treats Arabic, and three engineering mechanisms that compensate for that treatment without touching the model's weights.

### 2.6 Distinction

Across all five threads, prior work either *documents* a multilingual disadvantage or *removes* it by changing the model. The curse-of-multilinguality and tokenization-fertility literatures [4, 11, 36, 38, 41] establish that shared multilingual capacity and shared vocabularies under-serve lower-resource languages, but they remedy it by re-training: dedicating tokenizer vocabulary, adding capacity, or building language-specific models. The CLIR literature [3, 25, 29] bridges the language gap by *translation* or by relying on a better multilingual encoder. The fairness-in-RAG literature [13, 19, 43] filters biased *content* but does not address language-scoring geometry. The Arabic-NLP literature [1, 5, 31] supplies better Arabic components, again by training. We depart on three points, which together define our contribution.

First, **mechanism, measured.** We do not assume Arabic "scores lower"; we measure the distribution and report a significant *geometry* difference rather than a simple downward shift — on the clean production corpus Arabic relevant pairs score slightly *higher* in mean but in a roughly 2–2.6x *tighter* band (score compression). An expanded, three-model, significance-tested analysis (§7.8) confirms the spread differs significantly on every scorer while showing the magnitude and direction are sample- and model-dependent, so the robust claim is reduced/altered discriminability, not a universal compression ratio. To our knowledge this specific distinction has not been isolated for Arabic on a frozen multilingual reranker. It also corrects a framing in our own earlier drafts, which we retract.

Second, **frozen models, no retraining.** Every mechanism we propose leaves the multilingual scorer's weights untouched. The mitigation is an *emergent engineering property* of a frozen, on-device model rather than a training result: a self-calibrating per-tenant relevance gate that reads each tenant's own score geometry and hands an Arabic-heavy library a stricter, geometry-matched cutoff with no language-specific code; a continual self-extending cross-script glossary that lifts the structural Arabic-to-English lexical floor without any weight update; and a per-tenant BM25 sub-index that prevents a minority (often Arabic) library from being starved by a dominant one. This is the opposite of the retraining and re-tokenization remedies the prior threads prescribe.

Third, **per-tenant, label-free, on-device.** Unlike translation-on-every-query or globally-tuned thresholds, our gate calibrates the accept/reject boundary per tenant from the unlabeled local corpus alone, at cold start, with no labels, no feedback, and no data leaving the device. We are deliberate about the limits of this claim. The compensation is *partial and scorer-dependent* — only the pure-Arabic tenant is strictest in both score spaces, and a mixed Arabic+English tenant rank-flips between the cross-encoder and cosine scorers. The glossary's net benefit over a dense multilingual encoder is, at book level, essentially nil — a dense encoder such as LaBSE [14] already crosses scripts at full recall — so the glossary's value is cost and tenant-specific coverage, not beating dense retrieval, and we say so. And the starvation our sub-index cures is driven by shared *vocabulary*, not by language per se, so it is a fairness *precondition* for minority-vocabulary tenants rather than an Arabic-bias mechanism in itself. The contribution, stated honestly, is *Arabic fairness as an engineering property, not a training result*: a measured account of where the bias actually enters a frozen multilingual retrieval pipeline, and three on-device mitigations that partially compensate for it without retraining.


## 3. Where Bias Enters the Pipeline

Before we can fix a bias we must say precisely *where* it enters and *what shape* it takes. This section walks the offline multilingual RAG pipeline stage by stage and pinpoints the four places at which Arabic is disadvantaged relative to English. The central, and initially counter-intuitive, claim of the paper is established here in qualitative form and made precise in Section 4: the dominant harm to Arabic in this system is **not** that the multilingual scorer assigns Arabic relevant pairs lower scores — measured on our corpus they are if anything slightly *higher* — but that it crams Arabic scores into a much **narrower band**, shrinking the relevant-versus-irrelevant separation. A global decision cutoff calibrated on English's wide score geometry is therefore mis-placed for Arabic. We call this effect **score compression**, and we are deliberate throughout about distinguishing it from a level shift (a uniformly lower score), which our data refute.

The pipeline is the one specified in Section 6 (Experimental Setup): a per-tenant sparse index (Okapi BM25 [37] via bm25s [26]), a dense approximate-nearest-neighbour index (HNSW [27] under FAISS [21] over `paraphrase-multilingual-MiniLM-L12-v2` embeddings), Reciprocal Rank Fusion [12] of the two ranked lists, a multilingual cross-encoder reranker (`mmarco-mMiniLMv2-L12-H384-v1` [10]) trained on multilingual MS MARCO [10], and finally a per-query relevance gate that decides whether any retrieved passage is good enough to ground an answer. Bias can enter at four distinct stages of this pipeline; we take them in pipeline order.

\begin{center}
\includegraphics[width=1.00\linewidth]{figures/fig1_pipeline.pdf}
\end{center}


### 3.1 Stage 1 — Subword tokenization: a fragmentation penalty

Every model in the pipeline first segments text into subword tokens. The reranker, the embedder, and the local generator each carry their own tokenizer, and none of them was built around the morphology and orthography of Arabic. Arabic is templatic and richly inflected: a single orthographic word routinely carries clitic prepositions, conjunctions, the definite article, and pronominal suffixes, and short vowels are usually unwritten. A subword vocabulary trained on a corpus that is overwhelmingly English (or, in the multilingual case, English-dominated) therefore lacks dedicated subwords for many Arabic surface forms and is forced to fall back on shorter, more numerous pieces. The result is that the *same information content* is spread across more tokens in Arabic than in English.

We measure this directly with the **reranker's own tokenizer** on our corpus (Section 6). Arabic text is segmented at a **fertility of 1.946 tokens per word** versus **1.536 for English** — Arabic is fragmented **1.27×** more than English (Definition 1, Section 4.1). This is not an exotic finding: the disadvantage that non-Latin and morphologically rich languages suffer from English-centric tokenizers is by now well documented as a general phenomenon. Petrov et al. [36] show that the same text costs many more tokens in some languages than in others, with direct consequences for cost, latency, and the effective context window; Ahia et al. [4] quantify the resulting economic and modelling inequity across languages; and Rust et al. [38] demonstrate that tokenizer quality, not just model size, materially affects downstream performance on a given language. We stress an honesty point: **these studies survey many languages and do not report an Arabic-specific headline number**; we cite them as the general evidence base for tokenization unfairness and use our **own measured 1.27×** as the result for *this* pipeline on *this* corpus.

The fragmentation penalty is a *first-order* disadvantage in its own right (more tokens means more positions over which a fixed-capacity encoder must spread attention, a shorter effective Arabic context for the same token budget, and more opportunities for a segmentation to split a meaningful unit). For the purposes of this paper, however, its importance is as the **upstream cause of the downstream effect we actually act on**: representing Arabic as a longer, more fragmented token sequence is one plausible mechanism by which the scorer's Arabic outputs end up *less spread out* — compressed — even when their average level is not depressed. We are careful **not** to conflate the tokenization-stage disadvantage with the scoring-stage compression: they are different harms at different stages, and the gate of Section 5 acts on the latter, not the former. We do not claim a causal proof from fertility to compression; we report both as measured facts and note the natural direction of influence.

### 3.2 Stage 2 — Dense and sparse scoring: the curse of multilinguality and score compression

The retrieved candidates are scored twice: once implicitly in the dense/sparse retrieval stage, and once explicitly by the cross-encoder reranker whose score the gate ultimately thresholds. Both the bi-encoder embedder and the cross-encoder are **shared** multilingual models — one set of parameters covers roughly a hundred languages. Sharing capacity across many languages is precisely the regime in which the **curse of multilinguality** has been observed: for a fixed model size, adding languages eventually degrades per-language representation quality, and lower-resource languages bear the cost [11, 41]. Arabic, despite being a large language globally, is under-represented relative to English in the MS MARCO-derived training signal of these compact rerankers [10], so its relevance judgments are the ones we expect to be least sharp.

This is the stage at which the paper's core bias lives, and where our measurement overturns the naive expectation. We probed the scorer with two anchor populations per language, holding everything else fixed (Section 6; n = 40 pairs per language): a **self-match** population of guaranteed-relevant query/passage pairs, and a **cross-document** population of almost-surely-irrelevant pairs. The naive hypothesis — *the scorer is unfair to Arabic by giving Arabic relevant pairs lower scores* — is **false on our data**. Arabic relevant pairs score, if anything, *higher*: cross-encoder self-match mean 10.534 (Arabic) versus 10.448 (English), and cosine self-match mean 0.805 (Arabic) versus 0.754 (English). What is unfair is the **dispersion** (on this production corpus; §7.8 stress-tests this on an expanded three-model sample and reframes the robust claim as a *significant language-dependent geometry difference* whose magnitude/direction are model-dependent). The Arabic self-match scores are crammed into a far narrower band than the English ones here:

- **Cross-encoder self-match standard deviation: 0.348 (Arabic) versus 0.909 (English)** — English is ≈ 2.6× wider.
- **Cosine self-match standard deviation: 0.062 (Arabic) versus 0.128 (English)** — English is ≈ 2.1× wider.

The same picture holds at the bottom of the scale: the irrelevant (cross-document) Arabic band sits *higher and tighter* than the English one in the cross-encoder space (Arabic mean −6.69, std 1.179; English mean −7.673, std 2.261). With the relevant band slightly higher and *much* tighter, and the irrelevant band also higher and tighter, the **separation between "relevant" and "irrelevant" is smaller for Arabic** even though no Arabic score is lower. The English scorer has a long, expressive dynamic range (its self-match scores reach as low as 5.214 and its irrelevant scores as high as 2.069 — a wide, overlapping geometry it can afford because it has room); the Arabic scorer collapses that range. We formalise this as a **compression ratio** (Definition 4c, Section 4.4) and report it as the load-bearing number of the paper.

The consequence for the pipeline is immediate. A relevance decision is a *cut* placed on this one-dimensional score, exactly the score-distribution thresholding problem studied classically by Manmatha et al. [28], Arampatzis et al. [6], and Arampatzis and Robertson [7], and analogous to choosing a histogram cut à la Otsu [32]. Neural rerankers are known to be imperfectly calibrated [35] — but the bias here is sharper than generic mis-calibration: the *width* of the relevant band is language-dependent, so a cut tuned on English's wide geometry lands in the wrong relative position on Arabic's compressed geometry. We use [35] only for the general point that these scores are uncalibrated, **not** for any Arabic-specific direction, which we measure ourselves.

**Honesty caveat (cosine irrelevant band).** The compression story is clean and one-directional for the **self-match / relevant** distribution in both score spaces. It is *not* uniform: in the cosine cross-document (irrelevant) population the English standard deviation is actually the *smaller* of the two (Arabic 0.144 versus English 0.097). We therefore state the compression claim specifically about the **relevant / self-match band**, where it is unambiguous, and do not over-generalise it to "Arabic is always tighter everywhere."

### 3.3 Stage 3 — Post-retrieval thresholding: a mis-placed global cutoff

The retrieved-and-reranked candidates pass through the relevance gate: accept (ground the answer in the passage) iff the best score `s* = max_c M(q, c)` clears a threshold `τ`. This is the stage at which the compression of Stage 2 turns into an **observable unfairness**, and it is the stage our first mitigation acts on.

Suppose the deployment uses one global cutoff `τ` for every tenant — as the deployed system originally did, with a hand-tuned cross-encoder default of −5.0. Because the Arabic relevant band is compressed and sits slightly higher, while the English relevant band is wide, a single `τ` tuned to sit correctly inside English's separating gap does **not** sit at the corresponding relative position inside Arabic's much narrower gap. The cut is, in effect, calibrated on the wrong geometry for Arabic. Empirically (Section 7) the *correct* per-tenant cutoffs span [−3.29, −1.39] in cross-encoder logits and [0.302, 0.409] in cosine across five tenants drawn from the same corpus; no single value lies inside every tenant's separating band, and the Arabic-containing tenants need the **strictest** cutoffs of all. The crucial cross-check is that this higher Arabic cutoff is **not** an artefact of higher Arabic relevant scores: the Arabic tenants' in-domain lower band edge (`in_lo`, Definition 3) is essentially level with the English tenants' (T3 10.615, T4 10.584 versus T1 10.596, T2 10.521) — the higher cutoff is forced by a *tighter* band, not by a higher level. This is the precise mechanism by which score compression, not a score-level offset, drives the threshold result, and it is what Section 5 exploits without ever inspecting language.

### 3.4 Stage 4 — Lexical cross-script matching: a structural zero for Arabic→English

A separate, *structural* bias enters at the lexical (sparse) retrieval stage and is independent of the scorer's geometry. BM25 and any term-overlap signal match on shared surface tokens. An Arabic-script query and an English-script passage **share no tokens by construction**, so the lexical contribution to retrieving an English passage for an Arabic query is **exactly zero** — not small, but a hard floor of zero. On our bilingual tenant the book-level Arabic→English lexical recall@10 is **0.00** without intervention (Section 7). This is the absolute floor of the bias: it cannot be tuned away by a better threshold because there is no signal to threshold.

Two honesty points scope this correctly. First, the disadvantage is **one-directional**: the English→Arabic direction already achieves recall **1.00** unaided, because Arabic technical books embed English technical terms verbatim, so an English query finds them lexically. We report this asymmetry rather than averaging it away. Second — and this is the most important caveat about cross-script matching — a **dense multilingual encoder (MiniLM/LaBSE [14]) also reaches recall 1.0** at book level, so it is *false* to say "dense retrieval fails Arabic cross-script." The lexical zero is real and structural, but at book level the **net benefit of a glossary over dense-alone is essentially nil**; the glossary's value is elsewhere (cost of roughly 48 µs with no model loaded, versus 16–22 ms per query plus 92–453 s of one-time corpus embedding for the dense encoders, plus tenant-specific terminology and interpretability). We make this scoping explicit here so that the cross-script bias is not overstated: the harm is to the *lexical* half of a hybrid retriever specifically, and is a cost/interpretability problem more than a recall problem once a dense encoder is present. Cross-lingual reranking is itself strongest in-language, with cross-script the hardest case [3, 25], which is why the system gates a cross-lingual retry behind a calibrated same-language miss rather than translating unconditionally.

### 3.5 A note on a fifth, non-Arabic-specific stage

For completeness we flag one further fairness hazard that is *not* a bias against Arabic per se but harms the same users. In a multi-tenant deployment a small, often Arabic, library that **shares academic vocabulary** with a dominant tenant can have its chunks starved out of the global sparse ranking by corpus-wide term statistics; we observe minority shared-vocabulary overlap@5 collapsing from 0.913 to 0.461 as one tenant's dominance rises from 0.50 to 0.985, with a per-tenant sub-index restoring exact retrieval to 1.0 (Section 7). We are explicit that this is driven by **vocabulary overlap, not language** (a distinct-vocabulary minority stays flat), and that **dense retrieval starves *worse* than sparse** here (a FAISS post-filter overlap@5 collapses to 0.322, below the sparse 0.461) — so this is a retrieval-availability *precondition* for fairness, not an Arabic-bias mechanism in the scorer. We treat it as Mitigation 3 in Section 5 and do not count it among the four scorer-/lexical-level bias-entry points above.

---

## 4. Formal Model and Definitions

This section fixes notation and states, precisely, the quantities the paper measures and acts on. Definitions 1–4 formalise the bias (fertility and the two faces of score geometry — level and compression); Definitions 5–7 formalise the per-tenant gate that exploits the geometry. All quantities are defined so that they can be computed from an unlabeled corpus and a frozen scorer, with no retraining and no language-specific code, which is the engineering claim of the paper.

### 4.1 Corpus, tenants, and the scorer

Let a deployment serve a set of **tenants** (users) `T`. Tenant `t ∈ T` owns a private corpus `C_t`, partitioned by source document (book) into `C_t = D_1 ∪ ⋯ ∪ D_{K}` with `K = K(t) ≥ 1` books; a **chunk** `c ∈ C_t` is the unit of retrieval. A **query** is a string `q`. The **relevance scorer** is a fixed (frozen) function

```
M : (query, chunk) ↦ ℝ ,
```

realised in this system either as the multilingual cross-encoder, which returns an **unbounded relevance logit** `M(q, c) ∈ ℝ`, or as the embedding model used as a scorer, which returns a **bounded cosine similarity** `M(q, c) ∈ [-1, 1]`. `M` is uncalibrated: the numeric magnitude of `M(q, c)` has no fixed cross-query or cross-corpus meaning [35], and is the object whose *per-language geometry* this paper characterises.

**Definition 1 (Tokenization fertility).** Let `tok` be a tokenizer and `W` a set of whitespace-delimited words of a language `L`. The fertility of `L` under `tok` is the mean number of subword tokens per word,

```
Fert_tok(L) = ( Σ_{w ∈ W} |tok(w)| ) / |W| .
```

The **fragmentation ratio** of language `L1` relative to `L2` is `Fert_tok(L1) / Fert_tok(L2)`. On our corpus, with `tok` the reranker's own tokenizer, `Fert(Arabic) = 1.946`, `Fert(English) = 1.536`, and the Arabic-over-English fragmentation ratio is **1.27**. Larger fertility means the same content is spread over more tokens, the upstream disadvantage of Section 3.1.

### 4.2 In-domain and out-domain score sets

The geometry of `M` for a tenant is characterised by two **anchor populations**, constructed without labels from the corpus itself. They generalise the relevant / non-relevant score populations of classical score-distribution thresholding [6, 7, 28], but are *constructed* rather than read off judgments.

**Definition 2 (Self-match / in-domain set).** For a chunk `c` in book `D_{b}`, let `head_w(c)` be the first `w` words of `c` (we use `w = 12` for pseudo-queries, `w = 18` for labelled probes). A **self-match pair** is `(head_w(c), c)`: the query is literally the head of the very chunk being scored, so the pair is the most relevant pair that can exist for that chunk. For a deterministic sample of `S` chunks `c_1, …, c_S`, the **in-domain set** is

```
S_in(t) = { M( head_w(c_i), c_i ) : i = 1, …, S } .
```

`S_in` is an *upper* anchor: a guaranteed-relevant band. We note explicitly (and again in the threats to validity) that a self-match is a near-identical query/passage pair — an *upper bound* on relevance, a deliberate design choice, not a sample from a natural query distribution.

**Definition 3 (Cross-book / out-domain set).** With the same sampled chunks, pair each query `head_w(c_i)` (from book `D_{b(i)}`) against a chunk `c'_i` drawn from a **different** book `D_{b'}, b' ≠ b(i)`. The **out-domain set** is

```
S_out(t) = { M( head_w(c_i), c'_i ) : i = 1, …, S } .
```

`S_out` is a *lower* anchor: an almost-surely-irrelevant band. We summarise the bands by **robust edges** rather than extrema, to resist a single outlier pair:

```
out_hi(t) = Q75( S_out(t) )     # 75th percentile: top of the irrelevant band
in_lo(t)  = Q25( S_in(t)  )     # 25th percentile: bottom of the relevant band
```

where `Q_p(·)` is the `p`-th percentile. The empirical cross-check that drives the paper's thesis is stated in these terms: the Arabic tenants' `in_lo` are **not** lower than the English tenants' (`in_lo`: T3 10.615, T4 10.584 versus T1 10.596, T2 10.521), so any stricter Arabic cutoff cannot be attributed to a lower relevant level.

### 4.3 Separability

**Definition 4a (Separability / gap).** For tenant `t` under scorer `M`, the relevant and irrelevant bands are **separated** iff `in_lo(t) > out_hi(t)`; the **separating gap** is

```
gap(t) = in_lo(t) − out_hi(t) ,    defined when positive.
```

A positive gap means the geometry licenses a confident relevance cut between the two bands; a non-positive gap means it does not, and the gate declines to cut (Section 4.5). Separability is the only structural assumption the gate makes about `M`: it does not assume any particular numeric scale, only that self-matches out-rank cross-book pairs.

### 4.4 Score compression — the formal statement of the bias

The level of the relevant band and its *width* are two different things, and the paper's claim is about the second. Let `μ_in^L` and `σ_in^L` be the mean and standard deviation of the self-match scores `S_in` restricted to language `L` (analogously `μ_out^L`, `σ_out^L` for `S_out`).

**Definition 4b (Level offset).** The relevant-band **level offset** between languages is `Δμ = μ_in^{EN} − μ_in^{AR}`. The naive "Arabic scores lower" hypothesis is the claim `Δμ > 0` by a meaningful margin. **On our corpus `Δμ < 0`**: cross-encoder `Δμ ≈ −0.087` (English mean 10.448 vs. Arabic 10.534), cosine `Δμ = 0.754 − 0.805 = −0.051`. Arabic relevant pairs score *slightly higher*. The naive hypothesis is refuted, and we do not use it anywhere.

**Definition 4c (Score-compression ratio).** The relevant-band **compression ratio** of Arabic relative to English under scorer `M` is the variance (dispersion) ratio of the self-match populations,

```
ρ_M = σ_in^{EN} / σ_in^{AR} .
```

`ρ_M > 1` means Arabic relevant scores are *compressed* (a narrower band) relative to English. On our corpus, `ρ_{cross-encoder} = 0.909 / 0.348 ≈ 2.6` and `ρ_{cosine} = 0.128 / 0.062 ≈ 2.1`. This **`ρ_M ≈ 2.1–2.6`** is the formal statement of the bias *on the production corpus*: not `Δμ > 0` (a level shift, refuted), but `ρ_M ≠ 1` (a significant dispersion difference, measured). §7.9 (a clean, matched-pipeline replication on 390 diverse Arabic chunks) shows the *direction* is unstable: `ρ_M > 1` (Arabic narrower) held only in our small single-book sample, whereas on a clean larger sample the deployed cross-encoder gives `ρ_M < 1` (Arabic *wider*) and LaBSE gives `ρ_M ≈ 1`. The robust statement is `ρ_M ≠ 1` (a significant but **direction-unstable** dispersion difference), which is exactly what the per-tenant gate calibrates to without assuming a direction. Compression shrinks the separating gap available to Arabic: holding the irrelevant edge fixed, a smaller `σ_in^{AR}` pulls `in_lo^{AR}` and the whole relevant band toward the irrelevant band, so a cutoff calibrated on English's wide band is mis-placed on Arabic's narrow one. We confine the claim to the self-match / relevant band: in the cosine *irrelevant* band the ratio reverses (`σ_out^{EN}/σ_out^{AR} = 0.097/0.144 < 1`), so we make no uniform "Arabic always tighter" claim.

### 4.5 The per-tenant relevance threshold

The gate places one threshold `τ_t` per tenant per scorer, inside the separating gap of that tenant's own geometry.

**Definition 5 (Per-tenant threshold).** Given a gap-position hyperparameter `α ∈ [0, 1]` and a scorer-specific safe default `τ_default` and clamp band `[lo, hi]`, the threshold is

```
              ⎧ clamp( out_hi(t) + α · ( in_lo(t) − out_hi(t) ), lo, hi )   if in_lo(t) > out_hi(t)  and K(t) ≥ 2 and |C_t| ≥ 6
   τ_t   =    ⎨
              ⎩ τ_default                                                    otherwise
```

with `α = 0.25`, cross-encoder clamp `[-8, 2]`, and cosine clamp `[0, 0.95]`. Equivalently, when the gap exists, `τ_t = out_hi(t) + 0.25 · gap(t)`: the cut sits a quarter of the way up the separating gap from the irrelevant edge — a deliberately **recall-favouring** placement just above the irrelevant band. When the corpus is too thin (`K < 2` or `|C_t| < 6`) or the bands do not separate (`gap ≤ 0`), the method declines to over-fit and falls back to `τ_default`, failing safe rather than emitting a spurious cut.

**Definition 6 (Gate decision).** At query time, with retrieved candidates `c` and best score `s* = max_c M(q, c)`, the gate **accepts** (grounds the answer) iff `s* ≥ τ_t`, and otherwise **rejects** (answers from general knowledge or triggers the conditional cross-lingual fallback).

**Definition 7 (Determinism, cost, and caching).** The `S = 8` pseudo-queries are sampled with no randomness (each is `head_12` of a positionally chosen chunk), so `τ_t` is a deterministic function of `(t, M, state(C_t))`, where `state(C_t)` is a content signature of the corpus. It is computed once at cost `2S = 16` scorer calls and cached under that key; any change to `C_t` changes the signature and triggers exactly one recomputation. No per-query cost is added beyond the single comparison `s* ≥ τ_t`.

### 4.6 Why this absorbs the bias (and what it does not assume)

The threshold of Definition 5 never inspects language. The compensation is a side effect of *where the gap sits*. Both anchor sets `S_in(t)` and `S_out(t)` are drawn from the **same** corpus `C_t` and scored by the **same** frozen `M`, so any per-language behaviour of `M` enters both bands identically and is absorbed into the band edges `in_lo(t)` and `out_hi(t)`. Concretely, because the Arabic relevant band has a different spread (in the clean-corpus case it is compressed, `ρ_M > 1`) — and not because it is higher in level — the percentile-gap geometry lands `τ_t` at a *higher* relative position for an Arabic-heavy tenant, handing it a stricter, geometry-matched cutoff with no language-specific tuning, no labels, and no weight update. This is the formal content of "Arabic fairness as an engineering property, not a training result."

We are explicit about the limits of the formal guarantee, consistent with our results. The compensation is **partial and scorer-dependent**: it is a consequence of `M`'s own geometry, not a scorer-independent law. The pure-Arabic tenant earns a stricter cutoff in *both* score spaces (cross-encoder and cosine), and the two Arabic-containing tenants are the two strictest in the cross-encoder space, but the mixed Arabic+English tenant is the cross-encoder *maximum* yet the cosine *minimum* — a rank-flip across scorers (Section 7). The only structural assumption is separability (Definition 4a); subject to it, the gate places a correct relative cut on whatever scale `M` happens to use for that tenant, which is exactly the property we want for a frozen, on-device, multilingual scorer whose Arabic geometry we have shown to be compressed rather than depressed.


## 5. Mitigations

The measurement of Section 3 isolates a single defect with three faces. The multilingual scorer `M` does not score relevant Arabic pairs *lower* than English ones — on our corpus it scores them slightly higher (cross-encoder self-match mean AR 10.534 vs EN 10.448; cosine AR 0.805 vs EN 0.754) — but it assigns Arabic a significantly different dynamic range — on the clean production corpus it **compresses** it (self-match std AR 0.348 vs EN 0.909, a factor of ~2.6; cosine std AR 0.062 vs EN 0.128, a factor of ~2.1), so the relevant-versus-irrelevant separation shrinks and a global cutoff tuned on English's wide geometry sits in the wrong place for Arabic. Beneath the scorer, two further structural faces of the same disadvantage appear *before* `M` is ever consulted: an Arabic query and an English passage share no surface tokens, so the lexical retriever returns exactly nothing across scripts (BM25 cross-script recall is zero by construction), and a small or new Arabic library that *shares academic vocabulary* with a dominant English tenant has its relevant chunks crowded off the candidate list before any per-tenant filter runs.

Our thesis is that each face can be repaired as an **emergent engineering property of frozen, on-device models** — with no retraining, no fine-tuning, no labels, no language-specific code, and nothing leaving the device. This section presents the three mechanisms as algorithms (5.1–5.3) and then maps each precisely to the Arabic bias it fights (5.4). All three are deployed in `maktaba-web-local` and all three are reduced to practice; we are careful throughout to separate what the mechanism *does* from what we can *measure* about it, and to flag where a benefit is partial, scorer-dependent, or not intrinsically Arabic-specific.

### 5.1 Self-calibrating per-tenant relevance gate

The gate addresses score compression directly: instead of comparing the reranked top score to a fixed universal constant, it derives a threshold `τ_t` for each tenant `t` from that tenant's *own* score geometry, so a tenant whose scores are crammed into a narrow band gets a cutoff placed inside *that* band rather than inside English's wide one. Crucially the procedure is label-free, cold-start, and on-device: it needs no relevance judgements, no usage history, and no network call, and it never inspects the language of the corpus.

Let `M` be the relevance scorer — an unbounded cross-encoder logit, or, when no reranker is available, a bounded embedding cosine. The tenant's private corpus `C_t` is partitioned by source book into `{D_1, …, D_K}`. The gate brackets the unknown relevance cutoff between two anchors it can build *without labels*: a **self-match** anchor that is guaranteed relevant (a passage's own leading words scored against itself — the most relevant pair that can exist), and a **cross-book** anchor that is almost surely irrelevant (those same words scored against a chunk of a *different* book). It then places `τ_t` a recall-favouring fraction of the way into the gap between the irrelevant and relevant bands.

**Algorithm 1 — Per-tenant self-calibrating relevance threshold.**

```
Input:  corpus C_t partitioned into books {D_1..D_K}; relevance scorer M;
        sample size S = 8; gap fraction α = 0.25;
        clamp band [lo, hi]; safe default τ_default.
Output: per-tenant threshold τ_t.

 1:  if K < 2 or |C_t| < 6 then
 2:      return τ_default                              # applicability guard: geometry too thin
 3:  S_in ← ∅ ;  S_out ← ∅
 4:  for i = 1 .. S do                                 # deterministic, no RNG
 5:      c_i  ← i-th evenly-sampled chunk, in book D_{b(i)}
 6:      q_i  ← first 12 words of c_i                  # deterministic pseudo-query
 7:      c'_i ← a chunk of some book D_{b'}, b' ≠ b(i)  # cross-book partner
 8:      S_in  ← S_in  ∪ { M(q_i, c_i)  }              # self-match  (guaranteed relevant)
 9:      S_out ← S_out ∪ { M(q_i, c'_i) }             # cross-book  (almost surely irrelevant)
10:  out_hi ← Q75(S_out)                               # robust top edge of irrelevant band
11:  in_lo  ← Q25(S_in)                                # robust bottom edge of relevant band
12:  if in_lo > out_hi then                            # a separating gap exists
13:      τ_t ← out_hi + α · (in_lo − out_hi)           # α=0.25: a quarter into the gap
14:  else
15:      τ_t ← τ_default                               # no gap: decline to over-fit
16:  τ_t ← clamp(τ_t, lo, hi)                          # [-8, +2] xenc;  [0, 0.95] cosine
17:  cache τ_t under key (tenant, scorer, corpus-state)
18:  return τ_t
```

At query time the gate is simply: rerank the retrieved candidates, take `s* = max_c M(q, c)`, and **accept** (ground the answer in the passage) iff `s* ≥ τ_t`, else **reject** (answer from the model's general knowledge, or trigger the fallback of §5.1.1). Several design choices are load-bearing and we name what each buys. *Quartiles, not extrema:* summarizing the bands by `Q75(S_out)` and `Q25(S_in)` rather than `max`/`min` keeps a single anomalous pair from setting the threshold, at the cost of mild conservatism. *`α = 0.25` is recall-favouring:* placing the cut a quarter of the way up from the irrelevant edge keeps the bar just above the irrelevant band, protecting recall (Section 7 shows recall stays at 1.0 across `α ∈ [0, 1]` on this benchmark, so no recall is actually traded here; we adopt the conservative value as the safe default). *Fail safe on under-determination:* when the corpus is too thin (`K < 2` or `|C_t| < 6`) or no separating gap exists, the method declines to emit a spurious tenant-specific cut and reverts to a safe default. *Cost:* calibration is `2S = 16` scorer calls, paid once per corpus-state and cached under `(tenant, scorer, corpus-state)`; any ingest, delete, or re-chunk changes the corpus-state and triggers exactly one recomputation, and no per-query overhead is added beyond the single comparison `s* ≥ τ_t`.

**Scorer-agnostic construction.** A cross-encoder needs PyTorch and is the heaviest component in the pipeline, which cannot be assumed present on a low-end student laptop. The identical calibration therefore runs over a small `ScorerProfile` abstraction — a scoring function, a safe default, and a clamp band — so it works unchanged on the unbounded cross-encoder logit (clamp `[-8, +2]`) or on the embedding cosine already loaded for dense search (clamp `[0, 0.95]`), the latter needing no extra model and no PyTorch. This is itself a fairness-of-access property: the geometry-matched cutoff is available even where a reranker is not.

#### 5.1.1 Conditional cross-lingual fallback

The threshold also gates a cross-lingual retry, so translation cost — prohibitive on every query for an offline CPU device — is paid only when warranted. Given a query `q` in language `L`: (1) retrieve and rerank within `C_t` in `L`, and compute `s* = max_c M(q, c)`; (2) if `s* ≥ τ_t`, accept and stop — no translation; (3) if `s* < τ_t` (a *calibrated same-language miss*) and translation is enabled, translate `q` **once** into the corpus's other language — the translation cached and bounded by a hard timeout — and retry retrieval+rerank a single time, accepting on the cross-lingual evidence if the retried best score now clears `τ_t`, else rejecting to general knowledge; (4) if translation is disabled, a miss goes straight to general knowledge. Translation is thus conditional: triggered only by a calibrated miss in the original language, at most once. This is motivated by cross-lingual IR evidence that in-language reranking is strongest and cross-lingual matching is the costlier fallback path [3, 25, 29], and it matters most for low-resource Arabic/Iraqi-dialect content [2, 9, 31], where the query and the only relevant book may not share a language. As §5.2 details, this same fallback path is also the *teacher* that grows the glossary.

### 5.2 Continual self-extending cross-script glossary

The glossary repairs the lexical half of the hybrid retriever, which cannot cross scripts at all: an Arabic query and an English passage share no surface tokens, so `BM25(q, c) ≈ 0` for a cross-script pair however relevant it is — the absolute floor of the lexical disadvantage, and zero by construction. The mechanism is a deterministic, additive query-expansion operator coupled to a continual acquisition loop that grows the dictionary from the user's own retrieval failures, with no model in the query-time hot path once a term is learned and no weight ever updated.

**The expansion operator.** For a query `q` with normalized terms, the operator appends the other-language equivalents of every *known* term:

```
expand(q) = q ⊕ { D(w) : w ∈ terms(q), w ∈ dom(D) },
```

where `⊕` is token concatenation and `D` is the bilingual term map. Because expansion only *adds* tokens, lexical recall is monotone — `BM25_recall(expand(q)) ≥ BM25_recall(q)` — and a term absent from `D` simply contributes nothing. The lookup `cross_lingual_terms(q)` matches multi-word Arabic keys before (and in addition to) their single-word components, peels Arabic clitic prefixes (the agglutinated `ال / و / ب / ل / ف / ك / لل`) before lookup, and consults the forward, learned, and reverse maps in turn; the appended equivalents reach passages in the other language through *both* the BM25 and the dense path before Reciprocal Rank Fusion and reranking. The deployed map holds **308 Arabic→English** and **280 English→Arabic** domain pairs. The operator is deterministic, additive, and adds no model call (~48 µs/query, median 48.12, p95 53.57 over 10k calls).

**Algorithm 2 — Continual, relevance-gated term acquisition.**

```
Input:  query q; glossary D (forward / learned / reverse); local lexicon file.
Output: retrieved passages; possibly one new persisted term pair.

 1:  results ← retrieve( expand(q) )                  # deterministic expansion first — model-free
 2:  if results ≠ ∅:  return results                  # in-language (or already-known) hit; done
 3:  if q is a short single-term query and translation enabled:
 4:      t ← translate_once(q)                         # the SAME conditional fallback of §5.1.1
 5:      results ← retrieve( expand(t) )               # one-off cross-lingual retry
 6:      learn_term( strip_prefix(head(q)) → head(t) ) # persist prefix-stripped key → head word
 7:  return results
```

The supervision signal is the *retrieval failure itself* (the miss in line 2) — no human ever labels a pair. `learn_term` writes a prefix-stripped Arabic key mapped to the head word of the translation into a bounded local JSON lexicon (`data/learned_glossary.json`) under guards: it refuses to overwrite a curated entry, rejects degenerate pairs (keys shorter than 3 characters or translations longer than 60 characters), and stops at 5,000 learned pairs so the store cannot grow without limit. From the next query onward the learned term is expanded deterministically in line 1, so the language model in line 4 fires **at most once per term, ever**. What changes over time is a bounded JSON term store, not any model weight — we frame this as *continual training of the retrieval layer, not the generator*: it is auditable, instantly revertible, survives restarts, and admits no catastrophic forgetting, in explicit contrast to continual-retrieval methods that update model weights.

**An honest scoping of the benefit.** We do *not* position the glossary as a competitor to dense multilingual retrieval. As Section 7 reports, both a small on-device encoder and a heavy one (LaBSE [14]) *already* cross scripts at book-level recall 1.0 unaided, so at book level the glossary's net benefit over dense-alone is approximately nil. Its value lies elsewhere and is real: cost (~48 µs with no model versus a multilingual encoder resident in memory at 16–22 ms/query plus a one-time 92–453 s corpus embedding), the contribution of genuine lexical candidates to the RRF pool so the hybrid does not lean entirely on the dense encoder for every cross-script query, interpretability and editability, tenant-specific vocabulary, and the continual loop itself. We also report rather than average away an asymmetry: English→Arabic retrieval is already at recall 1.0 *without* the glossary, because Arabic technical books embed English terms verbatim; the burden falls on the Arabic→English direction only.

### 5.3 Per-tenant BM25 sub-index

The sub-index fixes the third face: an upstream retrieval-*availability* failure in which a minority tenant's relevant chunks are discarded *before* the per-tenant filter runs. The deployed shared-index lexical retriever works in two stages — score the query against one shared BM25 index over the whole corpus `C = ⋃_t C_t`, take the global top-`F` (an over-fetch multiple of `k`), then post-filter to the rows whose tenant equals `t`. The defect lives between the stages: the global top-`F` is committed using a ranking over *all* of `C`, so if one tenant owns the overwhelming majority of the corpus (we observed a 98.5 % / 1.5 % skew live), its passages fill the global top-`F` through two forces — **candidate crowding** (the dominant tenant contributes ~δ of every term's matches) and **statistics capture** (BM25's IDF is set by corpus-wide document frequencies the dominant tenant determines) — and the minority tenant's relevant chunks are cut off before the post-filter ever runs.

The remedy follows directly from the diagnosis: apply the tenant filter *before* scoring, so the BM25 statistics and ranking are computed over `C_t`, not `C`. Within a per-tenant sub-index, IDF weights are the tenant's own (statistics capture eliminated), the global top-`k` is by construction the tenant's top-`k` (crowding eliminated), and no over-fetch is needed (`F = k` suffices because every retrieved passage already belongs to `t`).

**Algorithm 3 — Per-tenant sparse retrieval.**

```
Input : query q; tenant t; shared corpus C (list); top-k.
State : cache U : tenant → (sig, index, sub_corpus)    # bounded, in-RAM.
Output: up to k passages of C_t, tenant-locally ranked.

 1:  sig ← (identity(C), length(C))                    # cheap corpus-state signature
 2:  if U[t] exists and U[t].sig = sig:
 3:      idx, sub ← U[t].index, U[t].sub_corpus         # cache hit
 4:  else:
 5:      sub ← [ c ∈ C : tenant(c) = t ]                # FILTER BEFORE SCORING
 6:      if sub = ∅:  return []
 7:      idx ← BM25_index( normalize(c) for c ∈ sub )   # tenant-local statistics
 8:      if |U| > MAX_TENANTS:  U.clear()               # memory bound; evicted tenant rebuilds
 9:      U[t] ← (sig, idx, sub)
10:  return topk( idx.retrieve(normalize(q), k) )       # F = k; no over-fetch
```

Two choices make the cache correct and cheap. The signature `sig = (identity(C), length(C))` invalidates automatically on any corpus mutation — ingestion either reassigns the corpus list (new identity) or extends it (new length) — so a stale sub-index is detected on the next query with no explicit invalidation call at the mutation sites. The cache is size-bounded (cleared past a fixed tenant count) so a deployment with very many tenants cannot exhaust memory, and if the sub-index cannot be built (unsupported backend, or any build error) the retriever falls back to the shared-index-plus-post-filter path, so the remedy never reduces availability. The cost is one small lazy build per *active* tenant (≈ 2.7 ms for the 58-chunk minority tenant), amortized over all of that tenant's subsequent queries, and zero for tenants who never query. The sub-index restores exact, dominance-independent recovery (oracle-overlap 1.0). Two honesty caveats carry forward to the evaluation: the mechanism is *anti-minority-shared-vocabulary*, not intrinsically anti-Arabic (a distinct-vocabulary minority does not starve, holding flat at overlap 0.91–0.97 across all skews); and the dense path *also* starves under the same crowding (a measured FAISS post-filter collapses to overlap 0.32 at the real skew, in fact below sparse's 0.46), so dense retrieval is not a cure — only the per-tenant sub-index (or in-traversal filtered-ANN as in Filtered-DiskANN [16] / ACORN [34]) is.

### 5.4 Why each mechanism fights a specific Arabic bias

The three mechanisms are not a grab-bag; each is matched to one face of the bias measured in Section 3, and together they are exhaustive over the points in the pipeline where the Arabic disadvantage enters.

**Gate ↦ compressed score geometry (a geometry-matched cutoff).** This is the load-bearing argument, and it is subtle because the gate *never inspects language*. Both anchors `S_in` and `S_out` are drawn from the *same* corpus and scored by the *same* model `M`, so any per-language behaviour of `M` enters both sets identically and is absorbed into where the gap sits. The bias is compression, not a level shift: Arabic relevant scores are not lower — on our corpus they are slightly higher (CE 10.534 vs 10.448) — but their *band is tighter* (std 0.348 vs 0.909), and the irrelevant band sits closer too (cross-doc CE AR −6.69 vs EN −7.673), so the relevant-versus-irrelevant separation shrinks. Because `τ_t = out_hi + α·(in_lo − out_hi)` is anchored to the tenant's own band edges, a compressed band lands the cutoff *higher*, so an Arabic-heavy tenant automatically receives a stricter, geometry-matched bar — without any language-specific tuning. The independent cross-check rules out the naive explanation: the higher Arabic cutoff does *not* come from higher relevant means (the Arabic tenants' in-domain `in_lo` are *not* elevated — T3 10.615, T4 10.584 versus T1 10.596, T2 10.521) but from the tighter band. We are explicit that the compensation is **real but partial and scorer-dependent**: the pure-Arabic tenant T3 is the strictest in *both* score spaces (cross-encoder cutoff −1.59, cosine 0.409, both extremal), and the two Arabic-containing tenants are the two strictest in the cross-encoder space, but the mixed Arabic+English tenant T4 is the cross-encoder maximum (−1.39) yet the cosine minimum (0.302) — a rank-flip across scorers. We therefore claim a geometry-matched corrective tendency, not a clean scorer-independent "Arabic = always strictest" law. A single global English-tuned cutoff (the deployed −5.0 default) would mis-place the bar for these compressed Arabic bands; the gate corrects this as an emergent property of the frozen scorer, with no retraining.

**Glossary ↦ cross-script lexical zero-overlap (an additive bridge).** The Arabic disadvantage here is structural and absolute, not statistical: an Arabic query can *never* lexically match an English passage, so the lexical floor is exactly 0 by construction (measured Arabic→English BM25 recall 0.00). The glossary lifts this floor to 1.0 at ~48 µs with no model and no weight update, and its continual loop recovers held-out Arabic technical terms (entropy / latent / logistic) one-shot from the user's own miss. It fights the bias by *restoring a lexical path that script-disjointness had zeroed out* — and, as scoped above, its honest contribution against a dense baseline is cost, fusion, interpretability, and continual learning rather than book-level recall, since dense already crosses scripts.

**Sub-index ↦ minority(Arabic)-tenant starvation (a fairness precondition).** In a bilingual library, the small or newly-added tenant that *shares academic vocabulary* with a dominant English tenant is — disproportionately, though not exclusively — the Arabic one, and its relevant chunks are crowded off the candidate list before reranking (overlap collapses to 0.46 at the real skew). The sub-index restores exact retrieval to 1.0, which is a retrieval-*availability* fairness precondition for everything downstream: neither the gate nor the glossary can help a chunk that was discarded before `M` was ever consulted. We frame this as a precondition harming minority-vocabulary-sharing (often Arabic) tenants rather than as an Arabic-bias mechanism per se, because the negative control shows the trigger is vocabulary overlap, not language, and because dense starves worse than sparse.

Taken together, the three mechanisms realize the paper's thesis: **Arabic fairness as an engineering property, not a training result.** Each is a frozen, on-device, label-free repair of one face of the compression-driven disadvantage — the gate matches the cutoff to the compressed geometry, the glossary restores the zeroed-out cross-script lexical path, and the sub-index restores the starved minority tenant's candidate availability — and none of the three updates a single model weight.


## 6. Experimental Setup

This section fixes the corpus, models, determinism regime, and metrics under which every number in Section 7 is produced. We hold to the same discipline as our companion papers [45, 46]: each result is regenerated by a deterministic script run by the author on the real on-device corpus, on one CPU machine, with the system's own normaliser, tokenizer, and scorers — no held-out cloud service, no GPU, and no randomness except the seeded sub-sampling that creates controlled tenant-dominance levels (§6.5). We are explicit at the outset that this is a reproducible **micro-benchmark**, not a large labelled IR evaluation, and that the Arabic portion of the live corpus is small; we treat all results as **direction-and-mechanism** evidence and carry that caveat into the limitations (Section 9).

### 6.1 Corpus

We study `maktaba-web-local` (*al-Maktaba al-Natiqa*, "The Speaking Library"), a fully-offline, on-device, multi-tenant Arabic+English digital-library RAG system. The bias-measurement experiments (§7.1–§7.3) run on the live deployed corpus, which is strongly **English-dominated by language**: of the language-attributed chunks, **63 are Arabic-dominant and 3,760 are English-dominant**. This skew is itself part of the phenomenon under study — the multilingual scorer was trained on broadly English-weighted multilingual data, and the corpus it operates on here is likewise English-heavy — but it also means the Arabic sample is **small** (63 chunks; the one unambiguously Arabic book contributes 45 of them), a fact we treat as a first-order threat to validity (Section 9) rather than a detail.

The per-tenant-geometry and gate experiments (§7.4, §7.5) run on the flagship five-tenant corpus of **3,739 chunks across 10 books and 5 themed tenants** [44], whose themes deliberately span language and subject so that score geometry can be read per tenant:

| Tenant | Theme | Language profile | Size (chunks) |
|:------:|-------|------------------|:-------------:|
| T1 | Machine learning / AI | English | 1,535 |
| T2 | Economics | English | 2,010 |
| T3 | Arabic + mathematics | Arabic (with embedded math) | 177 |
| T4 | Mixed AR+EN | Arabic and English | 2,733 |
| T5 | STEM | English | 45 |

The five themed tenants are *overlapping views* over the ten shared books — a single book may belong to more than one tenant — so the per-tenant chunk counts double-count shared books and sum to more than the 3,739 *distinct* chunks in the corpus; the corpus size is the 3,739 unique chunks, while the per-tenant sizes above are the (overlapping) sizes each tenant's gate calibrates on.

The cross-script glossary experiment (§7.6) runs on the real bilingual dominant tenant, which owns both English books (such as machine learning, economics, AI-search, scientific publishing, databases, cybersecurity, and physics) and Arabic books (programming basics, partial fractions) — the mix that creates genuine cross-script retrieval needs. The tenant-starvation experiment (§7.7) runs on the starvation corpus of **3,838 indexed passages across 13 books and two tenants**, with a measured natural skew of **98.5% / 1.5%** between the two tenants (3,725 vs. 58 tenant-attributed passages; the remaining 55 of the 3,838 are unattributed and excluded from the dominance ratio) [45]. We report each experiment against the corpus that is appropriate to the mechanism it isolates, and we name the corpus in every table caption so the provenance of each number is unambiguous.

### 6.2 Models

Two frozen, on-device scorers carry the entire study; **no weights are updated anywhere in this paper**.

- **Cross-encoder reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — a ~52 MB, 100-language MiniLM [39] cross-encoder fine-tuned on the multilingual MS MARCO passage-ranking dataset (mMARCO [10]), which includes Arabic. It emits an **unbounded relevance logit** per (query, passage) pair. This is the scorer the deployed system uses to rerank the fused candidate list, and it supplies the relevance score that the per-tenant gate thresholds.
- **Bi-encoder embedder:** `paraphrase-multilingual-MiniLM-L12-v2` (Reimers & Gurevych) — a 384-dimensional multilingual sentence encoder whose cosine similarity is bounded in the interval from minus one to one (empirically positive on our relevant pairs; self-match means AR 0.805 / EN 0.754). It supplies the dense half of the hybrid retriever and a second, geometrically different score space in which we re-measure every bias result, precisely so that no finding rests on a single scorer's idiosyncrasies.

The heavy-dense baseline in §7.6 is LaBSE [14] (~471 M parameters, ~1.8 GB), used only to answer the "why not just use a strong multilingual encoder?" objection honestly. The deployed pipeline around these models is hybrid: BM25 (the `bm25s` engine [26]) and FAISS-HNSW [21, 27] dense retrieval in parallel, merged by Reciprocal Rank Fusion (RRF, k=60) [12], reranked by the cross-encoder, and finally filtered by the self-calibrating per-tenant relevance gate.

**Tokenizer.** All tokenization-fertility figures (§7.1) are measured with the **reranker's own tokenizer** — i.e., the subword vocabulary the cross-encoder actually sees — so that the fragmentation we report is the fragmentation that downstream scoring inherits, not an artifact of some external tokenizer.

### 6.3 Determinism and reproducibility

Every number is produced by a deterministic script (`exp_arabic_bias.py` for §7.1–§7.3; `exp1_cutoffs` and the flagship gate harness for §7.4–§7.5; `exp_p2_glossary.py`, `exp_p2b_ablation.py`, `exp_p2c_labse.py` for §7.6; `exp_p1_starvation.py`, `exp_p1b_ablation.py`, `exp_p1c_dense.py` for §7.7; `exp_arabic_bias_v2.py` for the expanded multi-model analysis of §7.8 and `exp_arabic_bias_v3_clean.py` for the definitive matched-pipeline test of §7.9). The scorers are frozen and run on CPU; the probe construction is deterministic (passage-derived spans and fixed human-authored query sets, with no RNG in query generation); and the only randomness — the sub-sampling that sets tenant-dominance levels in §7.7 — is seeded and averaged over five fixed seeds (0–4). A reader pointing the released scripts at a comparable corpus obtains the same distributions, recall curves, and cutoffs.

### 6.4 Anchors for measuring score geometry

The bias measurement (§7.1–§7.3) and the gate's calibration (§7.4–§7.5) both rest on two **deterministic, label-free anchors** drawn from the corpus itself and scored by the same model M:

- **Self-match (relevant, upper anchor):** the score M(q_i, c_i) of a pseudo-query q_i (the leading words of chunk c_i) against its own source chunk c_i. This is a guaranteed-relevant pair and an *upper bound on relevance*, not a sample from a real query distribution — a deliberate design choice we flag before a reviewer does (Section 9).
- **Cross-document (irrelevant, lower anchor):** the score M(q_i, c′) of the same pseudo-query against an unrelated chunk c′ from a different book — a guaranteed-irrelevant pair.

For the standalone bias measurement we draw **n = 40 self-match pairs and 40 cross-document pairs per language**, sampled deterministically from the Arabic-dominant and English-dominant chunk pools, and score every pair in *both* the cross-encoder logit space and the cosine space. Because both anchors pass through the *same* scorer for *both* languages, any per-language behaviour of the scorer enters the relevant and the irrelevant band identically — which is exactly what lets us read "compression" off the standard deviations rather than off any external relevance labels (which the corpus does not have).

### 6.5 Metrics

We report:

- **Tokenization fertility** — mean subword tokens per whitespace word, per language, under the reranker's tokenizer (§7.1). The Arabic/English ratio is the fragmentation penalty.
- **Score compression** — the **standard deviation** of the self-match (relevant) band per language, in both score spaces, with the means, medians, and min/max reported alongside so that "compressed dynamic range" is verifiable, not asserted (§7.2–§7.3). The English-minus-Arabic mean difference makes the "Arabic is not lower" claim falsifiable.
- **Per-tenant cutoffs** — the gate's calibrated threshold τ_t per tenant, in both score spaces, together with the per-tenant in-domain lower anchor `in_lo` = Q25(S_in) used as the cross-check that the Arabic cutoffs come from a *tighter band*, not from lower scores (§7.4).
- **Gate accuracy** — accuracy, precision, recall, and F1 on a 60-IN / 60-OUT probe, against three reference policies (fixed default, label-tuned global oracle, per-tenant oracle), plus the count of off-topic acceptances and an α sensitivity ablation (§7.5).
- **Cross-script book-level recall@10** — whether any passage of the target book appears in the top-10, by direction, with and without glossary expansion, and against dense baselines (§7.6).
- **Oracle-overlap@5 and mean candidate yield** — the fraction of the tenant-local top-5 that the shared path actually recovers, as tenant dominance rises, for sparse and dense paths and the per-tenant sub-index (§7.7).

---

## 7. Results

We lead with the **refutation of the intuitive story**. The folk hypothesis — and an earlier draft of our own — was that the multilingual scorer assigns Arabic relevant pairs *lower* scores than English. Our own data refute this. The measured bias is **score compression**: the scorer crams Arabic relevant scores into a much narrower band than English's, shrinking the relevant-versus-irrelevant separation, so a global cutoff tuned on English's wide geometry is mis-placed for Arabic. Sections 7.2–7.4 establish this with three independent readings; Sections 7.5–7.7 show the three frozen-model mitigations that follow from it.

### 7.1 Tokenization fertility: Arabic is fragmented 1.27× more

Under the reranker's own tokenizer, on our corpus, Arabic words are split into substantially more subword pieces than English words.

**Table 1 — Subword tokenization fertility (reranker tokenizer, our corpus).**

| Language | Tokens per word | Ratio (AR / EN) |
|----------|:---------------:|:---------------:|
| English | 1.536 | — |
| Arabic | 1.946 | **1.27×** |

\begin{center}
\includegraphics[width=0.55\linewidth]{figures/fig8_fertility.pdf}
\end{center}


Arabic is fragmented **1.27× more** than English. This is the entry point of the bias, upstream of any scoring decision: a richly inflected, clitic-bearing word is shattered into pieces the model must reassemble, so the relevance signal for Arabic is carried by more, individually less informative, subword units. The general phenomenon — that subword tokenizers are systematically less efficient and less fair for morphologically rich and non-Latin-script languages — is established across many languages by Petrov et al. [36], Ahia et al. [4], and Rust et al. [38]; none of those works reports an Arabic-specific headline number, so we cite them as the **general evidence base** and report **our own measured 1.27×** as the corpus-specific fertility penalty our scorer actually inherits. We are careful not to overclaim the causal chain: this fragmentation is *upstream* and plausibly *contributes to* the downstream compression of §7.2, but our experiments do not isolate the fraction of compression attributable to tokenization alone, and we do not assert one.

### 7.2 The core bias is score compression, not lower scores

We measure the standard deviation of the guaranteed-relevant self-match band per language, in both score spaces (n = 40 per language).

**Table 2 — Self-match (relevant) score compression. `exp_arabic_bias.py`, n = 40 / language.**

| Score space | Arabic std | English std | EN / AR | Arabic mean | English mean | EN − AR mean |
|-------------|:----------:|:-----------:|:-------:|:-----------:|:------------:|:------------:|
| Cross-encoder logit | **0.348** | **0.909** | **2.61×** | 10.534 | 10.448 | **−0.087** |
| Cosine similarity | **0.062** | **0.128** | **2.06×** | 0.805 | 0.754 | **−0.051** |

\begin{center}
\includegraphics[width=1.00\linewidth]{figures/fig2_compression.pdf}
\end{center}


The English relevant band is **~2.6× wider** than Arabic's in cross-encoder logit space and **~2.1× wider** in cosine space. The model is, in a precise sense, **less discriminative for Arabic**: it has fewer effective gradations with which to separate a strong match from a weak one. The means tell the second half of the story, and it is the half that refutes the folk hypothesis: **the English-minus-Arabic mean difference is negative in both spaces** (−0.087 logits; −0.051 cosine), i.e. Arabic relevant pairs score *slightly higher*, not lower. The compression is visible in the extremes as well: in the cross-encoder space English's self-match band runs from a minimum of **5.214** to a maximum of 10.857 — a long left tail that gives English its width — whereas Arabic is bunched in **[9.485, 10.891]**; in cosine space English spans [0.432, 1.000] against Arabic's compressed [0.657, 0.913].

The bias is therefore **not** that Arabic scores lower. It is that Arabic's *dynamic range collapses*. A single global cutoff calibrated on English's wide geometry sits in the wrong place for Arabic's narrow one — which is the failure the per-tenant gate (§7.5) is built to correct without retraining. (These are this *small, single-book* Arabic sample's magnitudes; §7.8–§7.9 stress-test them on expanded, clean, matched-pipeline samples and find the *direction does not replicate* — on a 390-chunk diverse Arabic collection the deployed cross-encoder makes Arabic significantly *wider*, not narrower. The robust, transferable claim is therefore a statistically-significant but **direction-unstable** language-dependent geometry difference, handled by per-tenant calibration.)

### 7.3 Where the separation shrinks: the irrelevant band sits closer

Compression of the relevant band would not matter if the irrelevant band moved out of the way. It does the opposite: Arabic's irrelevant (cross-document) band sits **closer** to its relevant band, shrinking the separation that any threshold must exploit.

**Table 3 — Cross-document (irrelevant) band. `exp_arabic_bias.py`, n = 40 / language.**

| Score space | Arabic mean (std) | English mean (std) |
|-------------|:-----------------:|:------------------:|
| Cross-encoder logit | **−6.69** (1.179) | **−7.673** (2.261) |
| Cosine similarity | 0.130 (0.144) | 0.147 (0.097) |

In the cross-encoder space the Arabic irrelevant mean (−6.69) is **higher** than English's (−7.673) *and* tighter (std 1.179 vs. 2.261): Arabic's irrelevant scores are pulled up toward its compressed relevant band, so the relevant-vs-irrelevant gap is smaller for Arabic on both ends. This is the operational meaning of "less discriminable": not a lower relevant score, but a **smaller margin** between relevant and irrelevant. We flag one honest exception so the compression story is not over-sold: in the **cosine** space the irrelevant-band std is actually *smaller* for English (0.097) than Arabic (0.144). The clean, robust compression claim is specifically about the **relevant / self-match** band (Table 2), which is tighter for Arabic in *both* score spaces; we do not claim Arabic is uniformly tighter everywhere.

### 7.4 Per-tenant cutoff differentiation, and why it is not a level effect

When the gate calibrates a threshold per tenant on the flagship five-tenant corpus, the **Arabic-containing tenants receive the strictest cutoffs** — and the in-domain anchors prove this comes from a tighter band, not from lower relevant scores.

**Table 4 — Per-tenant calibrated cutoffs τ_t (canonical run, `exp1_cutoffs`).**

| Tenant | Theme | Cross-encoder τ_t | Cosine τ_t |
|:------:|-------|:-----------------:|:----------:|
| T1 | EN ML/AI | −2.93 | 0.382 |
| T2 | EN economics | −2.75 | 0.329 |
| T3 | **Arabic + math** | **−1.59** | **0.409** |
| T4 | **Mixed AR+EN** | **−1.39** | 0.302 |
| T5 | STEM (EN) | −3.29 | 0.325 |

\begin{center}
\includegraphics[width=1.00\linewidth]{figures/fig4_pertenant.pdf}
\end{center}


In the cross-encoder space the two Arabic-containing tenants (T3, T4) are the **two strictest** of the five, with the cutoff span [−3.29, −1.39] covering ≈1.9 logits — a single global threshold cannot serve both ends. In the cosine space the pure-Arabic tenant T3 has the **highest cutoff of all** (0.409). We are honest that the compensation is **partial and scorer-dependent**, not a universal law: only T3 is strictest in *both* spaces; the mixed tenant T4 is cross-encoder-strictest (−1.39) yet cosine-*loosest* (0.302) — a rank-flip we report rather than smooth over. We also note (footnote, not headline) that the absolute cutoffs vary run-to-run — a second run gave AR −1.79/−1.27 vs. EN −3.31/−2.75 — same direction, different absolutes; we report the canonical run throughout and do not mix the two.

The decisive evidence that this is a **geometry** effect, not a **level** effect, is the in-domain lower anchor `in_lo` = Q25 of the self-match band per tenant:

**Table 5 — In-domain lower anchor `in_lo` (cross-encoder). Arabic tenants are NOT lower.**

| Tenant | Language | `in_lo` (Q25 of S_in) |
|:------:|----------|:---------------------:|
| T1 | EN | 10.596 |
| T2 | EN | 10.521 |
| T3 | **Arabic** | **10.615** |
| T4 | **Mixed AR+EN** | **10.584** |

\begin{center}
\includegraphics[width=1.00\linewidth]{figures/fig3_cutoff.pdf}
\end{center}


The Arabic tenants' relevant scores are **not lower** — T3 (10.615) is in fact the *highest* of the four, and T4 (10.584) is above English T2 (10.521). The Arabic cutoffs are stricter purely because the Arabic band is **tighter**: with the same percentile-gap geometry, a compressed relevant band places the calibrated threshold higher. This is the §7.2 compression result reappearing, independently, in the cutoff machinery of the real five-tenant system — and it is the load-bearing fairness result of the paper. The gate never inspects language; it reads each tenant's own score geometry, and the geometry hands Arabic-heavy tenants a stricter, matched bar automatically.

### 7.5 Mitigation 1 — the self-calibrating gate: precision 0.71 → 0.94

We evaluate the gate on a 60-IN / 60-OUT probe pooled across the five tenants, against three reference policies. With α = 0.25 the gate places τ_t = out_hi + α·(in_lo − out_hi).

**Table 6 — Gate accuracy on the 60-IN / 60-OUT probe.**

| Policy | Labels needed? | Acc | Prec | Rec | F1 | Off-topic accepted (of 60) |
|--------|:--------------:|:---:|:----:|:---:|:--:|:--------------------------:|
| Fixed default (−5.0) | no | 0.792 | 0.706 | 1.000 | 0.828 | 25 |
| **Self-calibrated gate** | **no** | **0.967** | **0.938** | **1.000** | **0.968** | **4** |
| Label-tuned global oracle (+3.04) | yes | — | — | — | **1.000** | — |
| Per-tenant oracle | yes | — | — | — | 0.992 | — |

\begin{center}
\includegraphics[width=0.95\linewidth]{figures/fig5_gate.pdf}
\end{center}


The label-free gate lifts **precision from 0.706 to 0.938** at recall pinned to 1.0, raising F1 from 0.828 to 0.968 and removing **21 of the 25** off-topic acceptances that the fixed default lets through (25 → 4). It does this with **no labels and no training**, sitting *between* the per-tenant oracle (0.992) and just below the label-tuned global oracle (1.000) — and we explicitly **do not claim to beat** those oracles, which are unobtainable without labels the deployment does not have. The α sensitivity confirms the choice is not knife-edge:

**Table 7 — α ablation (recall = 1.0 throughout).**

| α | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|:----:|:----:|:----:|:----:|:----:|:----:|
| F1 | 0.736 | 0.845 | **0.968** | 0.992 | 0.992 | 0.992 |

F1 rises monotonically and then plateaus. We flag the plateau honestly: the apparent insensitivity at **α ≥ 0.75 is a clamp artifact** — at those settings all five tenants' cutoffs hit the clamp ceiling — not genuine robustness across the full range. Two scope caveats on this headline (carried into Section 9): the F1 number is **pooled across tenants and not Arabic-specific**, and recall pins to 1.0 partly because IN probes are passage-derived self-matches with their source left in the pool. The Arabic-fairness claim therefore rides on the **cutoff geometry of §7.4**, not on this F1.

### 7.6 Mitigation 2 — cross-script glossary: AR→EN recall 0.00 → 1.00, with an honest dense-parity caveat

Lexical retrieval cannot cross scripts: an Arabic query token can never match an English passage, so the Arabic→English book-level floor is **exactly zero by construction**. The deterministic glossary (308 AR→EN + 280 EN→AR pairs) closes it.

**Table 8 — Cross-script book-level recall@10 by method (same corpus, same queries).**

| Method | AR→EN | EN→AR | Per-query cost | Model |
|--------|:-----:|:-----:|----------------|-------|
| BM25, no glossary | **0.00** | 1.00 | ~µs | none |
| **BM25 + glossary (ours)** | **1.00** | 1.00 | +48 µs | none (308/280-pair map) |
| Dense — MiniLM (deployed) | 1.00 | 1.00 | ~16 ms/query | ~118 M |
| Dense — LaBSE (heavy) | 1.00 | 1.00 | ~22 ms/query | ~471 M / ~1.8 GB |

\begin{center}
\includegraphics[width=0.85\linewidth]{figures/fig6_crossscript.pdf}
\end{center}


The glossary lifts **Arabic→English recall from 0.00 to 1.00** at a median **~48 µs** with **no model**. The disadvantage is one-directional: English→Arabic is already 1.00 *without* the glossary, because Arabic technical books embed English terms (identifiers, formulae, transliterated loanwords) verbatim — we report this asymmetry rather than average it away. The continual loop recovers held-out terms one-shot: three Arabic terms verified absent from the static map (al-intrubiya → entropy, al-kamin → latent, al-lujisti → logistic) each go **0.0 → 1.0** after a single acquisition, deterministic and model-free thereafter, with **no weight update**.

We state the honest caveat plainly. **At this book-level metric the glossary's *net* benefit over dense-alone is nil:** both the deployed small MiniLM encoder and the heavy 1.8 GB LaBSE already cross scripts at recall 1.0 unaided. The glossary's genuine value is therefore **not** that it beats dense retrieval — it is **cost** (~48 µs and no model, versus a 16–22 ms per-query dense encode plus a one-time corpus-embedding step of ~92 s for MiniLM and ~453 s for LaBSE), interpretability/editability, the lexical-half fusion contribution, and the continual loop for tenant-specific terms. We scope "dense fails Arabic" strictly to **discriminability/compression** (§7.2), never to cross-script recall, where it does not.

### 7.7 Mitigation 3 — per-tenant BM25 sub-index: starvation, and dense starves too

In a bilingual library the small (often Arabic) tenant that *shares academic vocabulary* with a dominant tenant is starved: its relevant passages are crowded out of the global top-N before the tenant post-filter runs. Oracle-overlap@5 — the fraction of the tenant-local top-5 the shared path actually recovers — collapses as dominance rises.

**Table 9 — Shared-index minority retrieval vs. tenant dominance (shared-vocabulary probes, F = 18k, k = 5).**

| Dominance δ | mean yield (of 5) | oracle-overlap@5 |
|:-----------:|:-----------------:|:----------------:|
| 0.50 | 4.72 | 0.913 |
| 0.80 | 4.33 | 0.833 |
| 0.90 | 3.53 | 0.707 |
| 0.95 | 3.11 | 0.631 |
| 0.98 | 2.29 | 0.483 |
| **0.985 (real)** | **2.11** | **0.461** |

At the real 98.5% skew the minority tenant recovers only **46%** of its tenant-local top-5 (overlap 0.913 → **0.461**), losing more than half its correct candidates before reranking. A negative control isolates the cause as **vocabulary overlap, not language or size**: on distinct-vocabulary probes, overlap stays flat (0.968 → 0.914 from δ = 0.50 to 0.985). The per-tenant sub-index restores exact retrieval to **overlap 1.0** at every dominance level, at a one-time ~2.7 ms build cost.

The honest correction to the intuitive fix is that **dense retrieval starves too** — it is not immune:

**Table 10 — Measured dense (FAISS) post-filter oracle-overlap@5 vs. dominance.**

| Dominance δ | dense post-filter overlap@5 |
|:-----------:|:---------------------------:|
| 0.50 | 1.00 |
| 0.90 | 0.862 |
| 0.95 | 0.684 |
| 0.98 | 0.360 |
| **0.985 (real)** | **0.322** |

\begin{center}
\includegraphics[width=0.92\linewidth]{figures/fig7_starvation.pdf}
\end{center}

### 7.8 Statistical significance, multi-model replication, and an honest robustness analysis (expanded sample)

The results in §7.2–§7.4 are measured on the small, clean production corpus (63 Arabic chunks). To test whether the score-geometry effect is real or a small-sample artifact — the first concern any reviewer raises — we (i) expanded the Arabic sample roughly five-fold by chunking four additional **real** Arabic educational books (an Iraqi-university Arabic-language curriculum, a digital-logic-design Q&A, a computer-skills curriculum, and an AlSafwa University PowerPoint course), giving **309 Arabic chunks**; (ii) **balanced** English by down-sampling it to the same n = 309, so any width difference cannot be a small-Arabic-sample artifact; (iii) replicated the measurement on **three frozen scorers** (the cross-encoder logit, the paraphrase-multilingual-MiniLM cosine, and LaBSE cosine); and (iv) ran proper significance tests — Brown–Forsythe/Levene and Fligner–Killeen for equality of *spread* (the compression claim), Welch's t and Mann–Whitney U for *location* (the "Arabic lower?" claim), and a 10,000-resample bootstrap 95% CI on the std-ratio ρ = sd(EN)/sd(AR). The four added book texts are kept local and not redistributed; only the derived statistics are released (`exp_arabic_bias_v2.json`).

**Table 11 — Multi-model self-match geometry (expanded, balanced n = 309 per language).**

| Scorer | AR std | EN std | ρ = EN/AR | bootstrap 95% CI of ρ | Levene p | Fligner p | mean gap (EN−AR) | Welch p |
|--------|:------:|:------:|:--------:|:---------------------:|:--------:|:---------:|:----------------:|:-------:|
| Cross-encoder | 0.928 | 0.623 | 0.67 | [0.44, 0.98] | 0.0036 | 3×10⁻⁵ | +0.183 | 0.004 |
| MiniLM | 0.159 | 0.113 | 0.71 | [0.63, 0.80] | 4×10⁻⁵ | 3×10⁻⁴ | +0.027 | 0.016 |
| LaBSE | 0.093 | 0.113 | **1.22** | **[1.06, 1.41]** | 9×10⁻⁵ | 4×10⁻⁵ | −0.037 | 1×10⁻⁵ |

\begin{center}
\includegraphics[width=1.00\linewidth]{figures/fig9_multimodel.pdf}
\end{center}

Two facts stand out, and we report both honestly. **First, the spread differs significantly between Arabic and English on every model** — Levene and Fligner reject equality of variance at p < 0.01 for all three scorers (Figure 9). The geometry mismatch is real and statistically robust, not noise. **Second, the *direction* is model-dependent.** On LaBSE the compression direction *appears* to hold here — Arabic's relevant band is *narrower* (ρ = 1.22, bootstrap CI [1.06, 1.41]) and Arabic scores slightly *higher* (mean gap −0.037, Welch p < 10⁻⁵) — **but this too is a chunking artifact: under the clean matched pipeline of §7.9 even LaBSE shows no spread difference.** On the cross-encoder and MiniLM, however, the *expanded* Arabic band is *wider* (ρ < 1), the opposite of compression, with Arabic marginally lower (cross-encoder mean gap +0.183, p = 0.004). The bootstrap CIs exclude parity on all three models (Figure 10) — the difference is significant — but they fall on *different sides* of 1.0.

\begin{center}
\includegraphics[width=0.95\linewidth]{figures/fig10_bootstrap_ci.pdf}
\end{center}

**Where does the cross-encoder reversal come from?** A per-book diagnostic (Figure 11) localizes it to *chunk-extraction quality*, not language. The two cleanly-chunked Arabic sources reproduce v1 exactly — `live_corpus` self-match std **0.347** (matching §7.2's 0.348) and the cleanly-extracted `digital_logic` book **0.308** — both far tighter than the balanced English 0.623, i.e. compression. The two books we extracted from raw PDFs with a crude chunker — the Arabic-language *curriculum* (which contains Qur'anic verses and poetry; std 1.297) and *computer-skills* (std 1.097) — inflate the pooled Arabic variance. In other words, the expansion inadvertently compared *crude-chunked Arabic* against *production-clean English*; the wider pooled Arabic band is a **chunk-quality confound, not a language reversal**. The clean matched-pipeline test that would settle the magnitude was blocked when the production corpus became unavailable mid-study; we flag it as the most important future work (§9, T1).

\begin{center}
\includegraphics[width=1.00\linewidth]{figures/fig11_perbook.pdf}
\end{center}

**Query-shape robustness.** Repeating the cross-encoder measurement with shorter (6-word) and mid-span pseudo-queries gives std-ratios 1.02 and 1.00 (CIs spanning 1.0): the cross-encoder magnitude — and even the sign of its ratio — depends on the probe shape, while LaBSE's compression is stable. **Tokenization fertility** on the expanded sample is **1.32×** (Arabic 2.04 vs. English 1.55 tokens/word), consistent with §7.1.

**A confound still remained.** Because the four expanded Arabic books were chunked with a *crude* pipeline (pdftotext + a naïve splitter) while English came from the production corpus, §7.8 still compares *unlike* pipelines (Figure 11), so it cannot adjudicate the direction. The clean, matched-pipeline test that removes this last confound — and that proves decisive — is §7.9.
### 7.9 The definitive test: a clean, matched-pipeline replication overturns the compression *direction*

To remove every confound at once, we chunked four real Arabic books through the **exact production pipeline** used to build the English corpus — PyMuPDF page extraction → the system's own Arabic normalizer (`smart_normalize`: strips tatweel and diacritics, unifies orthographic variants) → `RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)` with the deployed separators — yielding **390 cleanly, identically chunked Arabic chunks** (live corpus 63 + curriculum 89 + PowerPoint 53 + computer-skills 125 + digital-logic 60) compared against **390 balanced production English chunks**. Both sides now pass through the *same* pipeline; the comparison is finally apples-to-apples, on a sample six times larger than v1 (`exp_arabic_bias_v3_clean.py`). This test **supersedes** the mixed-pipeline numbers of §7.8.

**Table 12 — Definitive matched-pipeline result (production-chunked, balanced n = 390).**

| Scorer | AR std | EN std | ρ = EN/AR | bootstrap 95% CI | spread test | mean gap (EN−AR) | location test |
|--------|:------:|:------:|:--------:|:----------------:|-------------|:----------------:|---------------|
| Cross-encoder | 1.109 | 0.623 | **0.56** | [0.39, 0.82] | Levene 8×10⁻⁵, Fligner 9×10⁻⁹ → **Arabic WIDER** | **+0.251** | Welch 1×10⁻⁴ → **Arabic LOWER** |
| MiniLM | 0.182 | 0.114 | **0.63** | [0.55, 0.72] | Levene 3×10⁻⁵ → **Arabic WIDER** | +0.009 | Welch 0.40 → no difference |
| LaBSE | 0.113 | 0.111 | 0.99 | [0.88, 1.11] | Levene 0.43 → **no difference** | −0.034 | Welch 2×10⁻⁵ → Arabic higher |

\begin{center}
\includegraphics[width=1.00\linewidth]{figures/fig12_definitive.pdf}
\end{center}

The result is decisive and, for the original headline, deflationary. **The "Arabic is score-compressed (narrower)" finding does not survive a clean, matched-pipeline, larger sample.** On the cross-encoder — the *deployed* reranker — Arabic's relevant band is now significantly *wider* (ρ = 0.56, CI [0.39, 0.82] entirely *below* parity) and its mean significantly *lower* (gap +0.251, p = 10⁻⁴); on MiniLM Arabic is again wider; on LaBSE there is no detectable spread difference at all (Figure 12). The clean test thus *reverses* the cross-encoder direction relative to v1 and *erases* it on LaBSE.

**Why v1 found the opposite: a corpus-homogeneity artifact.** The v1 Arabic sample (63 chunks) came from essentially one coherent book, so its self-match scores were homogeneous and tight (std 0.348). A *diverse* Arabic collection — five books spanning a verse-laden language curriculum, slide decks, skills manuals, and technical Q&A — produces a *wide* self-match band (std 1.109), because a scorer self-matches a homogeneous prose passage far more consistently than a heterogeneous mix of headings, lists, and fragments. **The relevant-band width is driven primarily by the collection's internal homogeneity and genre, not by the language** — and v1's "compression" was the narrowness of a single-book Arabic sample mistaken for a language property.

**The honest, final picture.** Pulling the three experiments together: (i) tokenization fertility (~1.3× for Arabic) is the one stable, language-level disadvantage; (ii) the relevant-score *geometry* genuinely differs between Arabic and English — significantly so on most model/corpus combinations — but its **direction and magnitude are unstable**, flipping with corpus size, genre/homogeneity, and model (Arabic narrower in a small homogeneous corpus; wider *and* lower on the deployed reranker in a large diverse one; indistinguishable on LaBSE); and (iii) consequently **neither the folk claim "Arabic scores lower" nor our earlier claim "Arabic is compressed" is a robust, transferable law** — each is a corpus- and model-specific observation. What *is* robust is the engineering consequence, and it is stronger than any fixed-direction bias would be: **because the cross-lingual geometry cannot be predicted from the language alone, a single global cutoff — or any fixed per-language correction — is guaranteed to be mis-placed for some collections, and the only sound remedy is to *measure* each tenant's own geometry and calibrate the cutoff to it.** That is exactly what the self-calibrating per-tenant gate does (§7.5): it never assumes a direction; it reads whatever spread and location a tenant's scores exhibit and places the cut accordingly. The instability we document is therefore not a weakness of the result — it is the central justification for the design.





The dense path collapses to **0.322** at the real skew — in fact *below* the sparse path's 0.461 — refuting "dense is immune." Crowding afflicts both paths; the per-tenant index returns both to 1.0. Two honesty points close the result: starvation is driven by **shared vocabulary, not Arabic per se** (it is a fairness *precondition* that happens to harm minority, often-Arabic, tenants), and the only structural cure on the sparse side is the per-tenant sub-index (the dense side additionally admits in-traversal predicate filtering as in Filtered-DiskANN [16] and ACORN [34]).

---


## 8. Discussion

### 8.1 The bias is geometric, and so is the cure

The central empirical finding of this paper is uncomfortable precisely because it contradicts the intuitive story. We did not find that the shared multilingual scorer assigns Arabic *lower* relevance scores. On every relevant-pair measurement it assigned Arabic scores that were slightly *higher* than English: cross-encoder self-match mean 10.534 (AR) versus 10.448 (EN), cosine self-match mean 0.805 (AR) versus 0.754 (EN). A practitioner who audited only the means would conclude there is no Arabic disadvantage at all, and would be wrong.

The disadvantage lives in the *second moment*, not the first. The model crams Arabic relevant-pair scores into a band roughly 2.6x tighter than English's on the cross-encoder logit (std 0.348 vs 0.909) and 2.1x tighter on cosine (std 0.062 vs 0.128). Because the irrelevant (cross-document) band for Arabic also sits higher and tighter (CE −6.69, std 1.179) than English's (CE −7.673, std 2.261), the *gap* between "relevant" and "irrelevant" — the only quantity a threshold actually exploits — shrinks for Arabic. The scorer is not less generous to Arabic; it is less *decisive* about Arabic. This is what we mean by score compression: a loss of discriminability, a narrower dynamic range, a ruler with fewer marks. These magnitudes, however, are **not transferable**: a definitive clean, matched-pipeline replication on a six-times-larger, diverse Arabic sample (§7.9) *overturns the direction* — the deployed cross-encoder there makes Arabic *wider and lower*, and LaBSE shows no difference. The v1 narrowness was a single-book corpus-homogeneity artifact. What is robust is that the geometry differs significantly by language while its *direction* does not — so the load-bearing claim is not 'Arabic is compressed' but 'the cutoff must be measured per tenant because the geometry cannot be assumed.'

This reframing matters because it changes what a fix has to do. If Arabic genuinely scored lower, the remedy would be a per-language additive offset — a bias term, the kind of thing that invites endless tuning and that bakes a language assumption into code. But a single global cutoff calibrated on English's wide geometry is mis-placed for Arabic not because Arabic needs a *bonus*, but because the cutoff sits at the wrong point in a differently-shaped distribution. The honest remedy is therefore not to push Arabic scores around; it is to *stop assuming one geometry*. The self-calibrating per-tenant gate (§5.1) does exactly this: it reads each tenant's own `out_hi` and `in_lo` percentile anchors and places the cutoff inside whatever gap that tenant's score distribution actually exhibits. Because Arabic's band is compressed, the same percentile-gap rule lands the Arabic-heavy tenants' cutoffs *higher* — T3 (Arabic+math) at −1.59 and T4 (mixed) at −1.39 are the two strictest cross-encoder cutoffs in the system, against −2.93/−2.75/−3.29 for the English tenants — without the gate ever inspecting a single character of script. The fairness correction falls out of the geometry. It is, in the most literal sense, emergent.

### 8.2 Fairness as an engineering property of frozen, on-device models

The thesis we want to defend is narrow and, we believe, defensible: **Arabic fairness here is an engineering property of frozen models, not a training result.** No weights were updated to obtain any of the three mitigations. The reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) and the embedder (`paraphrase-multilingual-MiniLM-L12-v2`) are exactly the public checkpoints, with exactly their measured Arabic compression intact. We did not fine-tune the compression away — we would not know how to do so on 63 Arabic chunks without overfitting, and a student in Karbala running this offline on a CPU has neither the data nor the hardware to retrain a 100-language cross-encoder regardless. The contribution is to show that the disadvantage can be *contained at the system layer*: a percentile gate that matches each tenant's geometry, a JSON glossary that bridges a structural lexical gap, and a per-tenant sub-index that prevents a dominant library from burying a minority one.

This stance is deliberately the opposite of the dominant "retrain for fairness" reflex, and it inherits the practical advantages of the training-free RAG-control line of work — gating *whether and what* to retrieve without auxiliary heads or labelled rewards [8, 40, 42] — but pushes it one step further into *per-tenant* score-geometry calibration rather than a single global retrieve/skip decision. Three properties follow from keeping the models frozen:

1. **It is auditable and revertible.** Every mitigation is a thresholds file, a glossary JSON, or an index partition — inspectable artifacts a maintainer can read, diff, and roll back. There is no opaque weight delta to explain to a reviewer or to a user who asks why a result was filtered. The continual glossary in particular is *continual learning of the retrieval layer, not of weights* (§5.2); contrast streaming-corpus methods that adapt the retriever by updating parameters, which gain expressiveness at the cost of auditability and a risk of catastrophic forgetting.

2. **It runs where the user is.** The whole point of an offline, on-device library for Iraqi students is that there is no GPU and no server round-trip. The gate costs `2S = 16` scorer calls per (re)calibration and is then cached; the glossary costs ~48 µs per query with no model loaded at all; the per-tenant index is a few milliseconds to build for a small tenant. These are the budgets that on-device RAG work targets [33], and they are met here without sacrificing the fairness behaviour. Notably, the gate is *scorer-agnostic*: it runs identically on the unbounded cross-encoder logit and on the bounded bi-encoder cosine, so a device too weak to run the cross-encoder still gets a calibrated Arabic-aware cutoff from cosine alone — fairness of *access*, not only of *ranking*.

3. **It needs no labels.** Relevance labels for Arabic educational passages at the per-tenant level simply do not exist for most deployments, and producing them is exactly the cost that low-resource settings cannot pay. The gate manufactures its own anchors from the corpus (self-match as a guaranteed-relevant upper anchor, cross-book as an irrelevant lower anchor) and reads the geometry off them. This places it in the lineage of unsupervised score-distribution thresholding [6, 28, 32] but specialised to *per-tenant* recalibration, which is what the compression result demands.

### 8.3 Implications for low-resource languages generally

We measured a 1.27x subword fragmentation penalty for Arabic under the reranker's own tokenizer (1.946 vs 1.536 tokens/word). We do not claim this number for any language but our own corpus, and we do not claim it is *the* cause of the compression — only a plausible contributor. But the *direction* is consistent with a now-substantial body of evidence that subword tokenizers trained on English-dominated data fragment non-Latin and morphologically rich languages more heavily, at a measurable cost in efficiency, sequence length, and downstream quality [4, 36, 38], and with the broader finding that a fixed-capacity multilingual encoder spreads representational budget unevenly across its languages — the "curse of multilinguality" [11, 41]. The general lesson is that the *symptom* a low-resource language exhibits in a shared model may not be a lower score; it may be a *compressed* one, and a compressed score is invisible to any audit that looks only at means.

If that is right, then the per-tenant, geometry-matching gate is not an Arabic-specific trick. Any deployment that pools a shared multilingual scorer across communities with different score geometries — different scripts, different domains, different fragmentation profiles — inherits the same mis-placed-cutoff problem and can apply the same fix, because the gate reads geometry rather than language. The glossary generalises in the same spirit: it bridges a *structural* lexical floor (an Arabic query can never lexically match an English passage, so AR→EN BM25 recall is exactly 0 by construction) for any script pair where the burden is one-directional. We were careful to report that asymmetry rather than average it away: EN→AR was already 1.0 unaided, because Arabic technical books embed English terms verbatim, so the entire burden falls on AR→EN — and the glossary's *book-level* win there is not recall (a dense multilingual encoder reaches 1.0 too; §8.4) but cost, interpretability, and tenant-specific vocabulary acquired one-shot from a miss.

### 8.4 What we are *not* claiming

We are not claiming Arabic is universally and symmetrically disadvantaged. The compression result is clean for the *relevant / self-match* distribution; on the irrelevant cosine band English is actually slightly *tighter* (std 0.097 vs 0.144), so "Arabic is always the narrower distribution" would be an overstatement and we do not make it. We are not claiming the gate beats a label-tuned oracle: the label-free gate reaches F1 0.968 on the pooled IN/OUT probe, but a global cutoff tuned *on the labels* reaches F1 1.000 and a per-tenant oracle reaches 0.992 — the gate is *near-oracle without labels*, which is the honest and useful claim, not a victory over supervision. We are not claiming the F1 number is an Arabic result; it is a pooled probe over five tenants on a small English-dominated corpus, with recall pinned to 1.0 because the IN queries are easy passage-derived self-matches with their source left in the pool. The Arabic-fairness argument rides on the *cutoff geometry* (the per-tenant `in_lo` cross-check: T3 10.615, T4 10.584 are not lower than T1 10.596, T2 10.521 — so the stricter Arabic cutoff comes from a tighter band, not lower scores), not on the headline F1. And we are explicitly *not* claiming the glossary beats dense multilingual retrieval at the book level: at that granularity its net benefit over a dense encoder is essentially nil, and we say so plainly. Its value is the ~48 µs no-model cost, the interpretability, the lexical-half fusion contribution, and the continual one-shot acquisition of held-out tenant terms — not cross-script recall, which dense already provides.

Finally, the starvation mitigation is *not* intrinsically anti-Arabic, and we resist the temptation to dress it up as such. Per-tenant BM25 starvation is driven by shared *vocabulary*, not by language: a distinct-vocabulary minority tenant stays flat (control 0.968 → 0.914), while a minority tenant that *shares academic vocabulary* with a 98.5%-dominant tenant collapses to overlap@5 = 0.461. It harms whichever minority shares vocabulary with the dominant tenant — frequently, in a bilingual academic library, the small Arabic one — and the per-tenant sub-index restores it to 1.0. We also report the inconvenient fact that *dense retrieval starves worse than sparse* under the same skew (FAISS post-filter overlap collapses to 0.322, below sparse's 0.461), which refutes any claim that "just use dense embeddings" solves minority-tenant availability. It does not; only per-tenant isolation (or in-traversal filtered ANN [16, 34]) does.

### 8.5 On the word "bias": calibration mechanism, fairness outcome

A reviewer may reasonably object that what we measure — a language-dependent difference in score *geometry* and threshold placement — is a *calibration* problem rather than *bias* in the social-fairness sense. We agree the **mechanism is calibration**, and we have characterized it as such throughout (a discriminability/score-geometry disparity, not a moral claim). We keep the word *bias* in the **information-retrieval-fairness** sense used by the RAG-fairness literature [19, 43]: a *systematic, language-conditioned difference in retrieval outcomes* that, left uncorrected by a one-size cutoff, surfaces worse evidence for Arabic queries than for English ones. The two framings are not in tension — the calibration mismatch *is* the mechanism of the retrieval-fairness gap — and our contribution (per-tenant calibration) is agnostic to the label one prefers. Following our expanded and definitive analyses (§7.8–§7.9) we have deliberately softened all fixed-direction language: the robust claim is a *significant but direction-unstable* language-dependent geometry mismatch, and "compression" names only what we saw in one small single-book sample, not a transferable law.

## 9. Limitations and Threats to Validity

We follow the convention of our prior work in stating threats explicitly and in roughly descending order of how much they should temper the reader's confidence.

**T1 — Sample size, corpus quality, and the fragility of the compression *magnitude*.** The clean production corpus is 3,760 English-dominated chunks against only 63 Arabic; the single unambiguously-Arabic book contributes 45 of them. We addressed the small-sample concern directly (§7.8) by expanding Arabic to **n = 309** from five sources, balancing English to the same n, replicating on **three frozen models**, and running significance tests — which establish that the relevant-band spread differs **significantly** by language on every model (Levene/Fligner p < 0.01). But that same rigorous analysis — and a definitive clean, matched-pipeline replication (§7.9) that chunks 390 diverse Arabic chunks through the *exact* production pipeline used for English — exposed a deeper, honest finding: the *direction* of the geometry difference is **not stable**. The original compression (Arabic narrower) was a corpus-homogeneity artifact of a single-book 63-chunk sample; on a clean, larger, diverse sample the deployed cross-encoder makes Arabic significantly *wider and lower*, and LaBSE shows no difference. We therefore no longer claim Arabic is compressed (nor that it scores lower) as a transferable law — only that the geometry differs significantly and unpredictably, which is the case the per-tenant gate is built for. (Residual confound for the curious: the diverse Arabic books span more genres than the coherent English textbooks, so part of the wider Arabic band reflects genre heterogeneity; disentangling genre from language with a genre-matched bilingual corpus is the remaining future work.) The original phrasing about chunking through the same production pipeline as English and re-measuring on a clean, matched, larger sample; this was begun here but blocked when the production corpus became unavailable mid-study. Accordingly we no longer quote the original 2.6×/2.1× ratios as universal constants — the robust, transferable claim is the **statistically-significant language-dependent geometry mismatch**, not a fixed magnitude.

**T2 — Passage-span probes, not natural questions.** The gate's IN anchors are self-match pairs: a query is the first ~12 words of a chunk, scored against that same chunk. This is a deliberate design choice — a guaranteed-relevant *upper anchor* on relevance — and not a model of how a student actually asks a question. It inflates IN scores and is the reason recall pins to 1.0 in the gate probe. Real queries are shorter, noisier, and often not lexically contained in any single passage; we have not measured the gate against a natural-question distribution, and the F1 figure should be read with that caveat foremost. The cross-document anchors are likewise an idealised lower bound.

**T3 — The glossary's book-level net benefit over dense is ~nil.** We restate this here as a validity threat, not only as honesty. Because a dense multilingual encoder (MiniLM, LaBSE [14]) already crosses scripts at recall 1.0 at the book level, the glossary does *not* demonstrate that "dense multilingual retrieval fails Arabic" — it does not. Any reader who takes the AR→EN 0.00 → 1.00 result as evidence against dense retrieval has misread it: that 0.00 is the *sparse-lexical* floor, which is zero by construction. The glossary's defensible contributions are cost (~48 µs vs 16–22 ms/query plus a 92–453 s one-time corpus-embedding step), interpretability, and continual tenant-specific acquisition — not retrieval quality over dense.

**T4 — Single machine, single CPU, two specific checkpoints.** Every number is a micro-benchmark from one CPU machine with one reranker and one embedder. Timings (48 µs glossary, 16 scorer calls per calibration, per-tenant build times) are machine-specific. The compression magnitudes could differ under a different multilingual checkpoint; the *qualitative* claim (compression, not lower scores) is what we expect to transfer, and even that we have only verified on these two models. Cutoff *absolute* values also vary run to run (a separate run gave AR −1.79/−1.27 vs EN −3.31/−2.75); we report the canonical run throughout and note that only the *direction* (Arabic-containing tenants strictest in cross-encoder; pure-Arabic strictest in cosine) is stable, while the mixed tenant rank-flips between score spaces (T4 is cross-encoder-strictest but cosine-loosest) — so the compensation is *partial and scorer-dependent*, not a universal law.

**T5 — No end-to-end answer-quality study.** We measure retrieval-stage and gate-stage quantities: discriminability, cutoff placement, recall, overlap@5, latency. We do *not* measure whether the final generated answer from the downstream local LLM is more correct, more helpful, or more fair for Arabic users as a result. Fairness in RAG can be undermined or surfaced specifically at the generation stage [19, 43], and we have not run that evaluation. The claims in this paper stop at the retrieval/gating boundary; the leap from "Arabic passages are gated with a geometry-matched cutoff" to "Arabic users get better answers" is one we have not earned and do not make.

**T6 — Gate ablation plateau is partly an artifact.** The α-sensitivity ablation (F1 = 0.736 / 0.845 / 0.968 / 0.992 / 0.992 / 0.992 at α = 0.00 / 0.10 / 0.25 / 0.50 / 0.75 / 1.00) plateaus from α ≥ 0.5. We caution that the plateau at α ≥ 0.75 is a clamp-to-bound artifact (all five tenants hit the clamp), not evidence of genuine insensitivity to α; the operating point α = 0.25 was chosen below the clamp region.

**T7 — Construct validity of "relevance" anchors.** Self-match and cross-book scores are proxies for "relevant" and "irrelevant." A self-match pair over-represents lexical overlap and under-represents paraphrase relevance; a cross-book pair can occasionally be topically related (two STEM books). The percentile anchors (`out_hi = Q75(S_out)`, `in_lo = Q25(S_in)`) and the gap test `in_lo > out_hi` are designed to be robust to a few such contaminations, but the proxy is a proxy. Modern neural rerankers are also known to be imperfectly calibrated [35], which is part of *why* per-tenant recalibration helps, but it also means the raw scores the gate reads are not probabilities.

## 10. Conclusion

We set out to ask whether a frozen, offline, multilingual retrieval system is fair to Arabic, and to fix it if not. The answer turned out to be more precise — and less flattering to the obvious narrative — than expected. The shared multilingual scorer does not penalise Arabic with lower relevance scores; if anything Arabic relevant pairs score marginally higher. The penalty is **not a fixed direction.** Our first, small, single-book measurement showed Arabic *compressed* (a ~2.6× tighter cross-encoder band); but an expanded three-model analysis (§7.8) and a definitive clean, matched-pipeline replication on 390 diverse Arabic chunks (§7.9) overturn that direction — on the deployed cross-encoder Arabic is significantly *wider and lower*, and on LaBSE there is no difference at all. The one stable language-level disadvantage is ~1.3× heavier subword fragmentation; the relevant-score geometry genuinely and significantly differs by language, but its **direction and magnitude are unstable**, flipping with corpus homogeneity, genre, and model. The robust conclusion is the engineering one, and it is stronger than any fixed bias: because the geometry cannot be predicted from the language, *any* single global cutoff — or fixed per-language correction — is mis-placed for some collections, so the cutoff must be **measured per tenant**. That instability is the central justification for the self-calibrating gate.

Against that diagnosis we contributed three mitigations, none of which touches a model weight. A self-calibrating per-tenant relevance gate reads each tenant's own score geometry and, purely as a consequence of Arabic's compressed band, hands Arabic-heavy tenants stricter, geometry-matched cutoffs (T3 −1.59, T4 −1.39, the two strictest) while reaching label-free F1 0.968 — near-oracle without labels, never claiming to beat the label-tuned oracle. A continual cross-script glossary lifts the structural AR→EN lexical floor from 0.00 to 1.00 and acquires held-out terms one-shot with no weight update, at ~48 µs and with the honest caveat that its book-level benefit over dense retrieval is essentially cost and interpretability, not recall. A per-tenant BM25 sub-index restores a vocabulary-sharing minority tenant from overlap@5 0.461 back to 1.0, fixing a starvation that dense retrieval, far from solving, actually worsens.

The unifying claim is deliberately modest in scope and, we hope, sturdy: **Arabic fairness here is an emergent property of careful engineering over frozen on-device models, not a result of retraining.** It is achieved without labels, without a GPU, and without writing a single line of language-specific code — which is precisely what makes it deployable for the offline, low-resource, Arabic-and-Iraqi educational setting we built it for. The results are micro-benchmarks on a small English-dominated corpus on one machine, with no end-to-end answer-quality study; they establish mechanism and direction, not magnitudes for Arabic at large. We release the artifact so that both the diagnosis and the three fixes can be reproduced, contested, and extended.

## Reproducibility

All experiments are deterministic and CPU-only. The bias-measurement script `exp_arabic_bias.py` emits the JSON `exp_arabic_bias.json` reproduced in §7.1–§7.3; the expanded multi-model analysis (§7.8) and the definitive clean matched-pipeline test (§7.9) are produced by `exp_arabic_bias_v2.py` and `exp_arabic_bias_v3_clean.py` (the latter reuses the production chunker — PyMuPDF + the system Arabic normalizer + a 500/100 recursive splitter — so that the added Arabic books are chunked identically to English). All result JSONs are released. The original script emits including the self-match/cross-document score statistics (n=40/language), the tokenization-fertility computation under the reranker's own tokenizer, and the compression summary. Pseudo-query sampling is deterministic (evenly-spaced chunk indices; first 12 words as the query head), so re-runs reproduce the same anchors. We pin the two public checkpoints — reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` and embedder `paraphrase-multilingual-MiniLM-L12-v2` — and report run-to-run variation in cutoff absolutes where it occurs (§9, T4). The gate (Algorithm 1), glossary expansion and one-shot acquisition (Algorithm 2), and the per-tenant sub-index policy are specified in Appendix A in sufficient detail to re-implement independently. The released artifact contains the scripts, the results JSON, the deployed glossary (308 AR→EN + 280 EN→AR pairs), and the configuration constants (S=8, α=0.25, clamp ranges, guard thresholds). Because the corpus is small and English-dominated, we report it as a fixed micro-benchmark rather than a benchmark suite, and we caution against treating the magnitudes as transferable (§9, T1, T4).

## Declarations

**Funding.** No funding was received for this work.

**Competing interests.** The author declares no competing interests. (A patent disclosure related to the per-tenant relevance gate was prepared and subsequently abandoned; the author has chosen to publish the work openly and asserts no patent rights over it.)

**Data and code availability.** The experiment scripts, results JSON, deployed glossary, and configuration are released as a reproduction artifact under an open licence. The underlying book corpus contains third-party copyrighted texts and is not redistributed; the per-language statistics, derived measurements, and synthetic/anchor probes needed to reproduce every reported number are included.

**Author contributions.** Ayman Kazim Yousef is the sole author and conducted all design, implementation, experiments, analysis, and writing. The editorial "we" is used throughout.

**Ethics.** No human subjects, personal data, or user studies were involved. All retrieval is offline and per-tenant isolated by `user_id`; no user data leaves the device.

**Use of AI tools.** AI assistance was used for drafting and editing prose; all numbers, claims, and their honesty constraints are the author's and are tied to released, reproducible artifacts.

## Appendix A. Algorithms

The notation follows §4: `M(q, c)` is the relevance score of query `q` against chunk `c` under scorer `M` (either the cross-encoder logit, unbounded, or the bi-encoder cosine, bounded in [−1, 1]); `C_t` is the chunk set of tenant `t`; `Q_p(·)` is the `p`-th percentile.

### Algorithm 1 — Self-calibrating per-tenant relevance gate

```
# Calibration: run once per (tenant t, scorer M, corpus-state signature),
# then cache the threshold. Cost = 2S scorer calls. No labels, no training,
# never inspects language.

function CALIBRATE_GATE(t, M, S=8, alpha=0.25,
                        clamp_lo, clamp_hi,   # CE [-8,+2]; cosine [0,0.95]
                        K_min=2, C_min=6):
    C_t  <- chunks of tenant t
    if number_of_books(C_t) < K_min or |C_t| < C_min:
        return SAFE_DEFAULT(M)                 # too small to calibrate; fail safe

    # S deterministic pseudo-queries: evenly-spaced chunks, first 12 words each
    Q   <- [ head(c_i, 12_words) for c_i in evenly_sampled(C_t, S) ]

    S_in  <- [ M(q_i, c_i)        for (q_i, c_i) in zip(Q, sampled_chunks) ]   # relevant upper anchor (self-match)
    S_out <- [ M(q_i, c'_i)       for q_i, c'_i in cross_book_pairs(Q, C_t) ]  # irrelevant lower anchor

    out_hi <- Q75(S_out)
    in_lo  <- Q25(S_in)

    if in_lo > out_hi:                          # a usable gap exists
        tau_t <- out_hi + alpha * (in_lo - out_hi)
    else:                                        # no separation; do not over-filter
        tau_t <- SAFE_DEFAULT(M)

    tau_t <- clip(tau_t, clamp_lo, clamp_hi)
    CACHE[(t, M, signature(C_t))] <- tau_t
    return tau_t

# Decision at query time: accept the top reranked candidate iff it clears tau_t.
function GATE_ACCEPT(s_star, t, M):
    tau_t <- CACHE.get((t, M, signature(C_t)))  or  CALIBRATE_GATE(t, M)
    return (s_star >= tau_t)
```

The geometry argument (why this is fair to Arabic) is in §4 and §8.1: both `S_in` and `S_out` are drawn from the *same* corpus and scored by the *same* `M`, so any per-language behaviour — including Arabic's compressed band — enters both anchors identically and is absorbed into where the gap sits. A compressed band yields a higher `in_lo`-anchored cutoff automatically.

### Algorithm 2 — Continual self-extending cross-script glossary

```
# Query-time expansion: additive, model-free, recall-monotone (~48 us/query).
function EXPAND(q, D):                           # D: bilingual dictionary (JSON)
    out <- tokens(q)
    for key in match_multiword_first(q, D):      # multi-word keys before components
        out <- out ∪ D[key]
    for w in tokens(q):
        w_s <- STRIP_AR_CLITICS(w)               # peel ال/و/ب/ل/ف/ك/لل
        if w_s in D:
            out <- out ∪ D[w_s]
    return out                                    # union only: never removes a term

# Continual one-shot acquisition: a same-language retrieval MISS is the signal.
# The LLM fires at most once per term, ever; thereafter deterministic & model-free.
function ON_MISS(q, D):
    if is_short_single_term(q) and same_language_miss(q):
        src <- STRIP_AR_CLITICS(head(q))
        tr  <- LLM_TRANSLATE_ONCE(q)             # cached; timeout-bounded
        dst <- head(tr)
        if VALID(src, dst):                      # |src|>=3, |dst|<=60 chars,
            LEARN_TERM(D, src -> dst)            #   and src not a curated key
    # else: fall through to normal retrieval / cross-lingual fallback

function LEARN_TERM(D, src -> dst):
    if src in CURATED(D): return                 # never overwrite curated pairs
    if |D| >= 5000: return                       # bounded local store
    D[src] <- D.get(src, {}) ∪ {dst}
    persist(D)                                    # local JSON; auditable, revertible
```

Acquisition updates a JSON store, never model weights (§8.2). On the three verified held-out terms (entropy / latent / logistic) this recovers AR→EN retrieval from 0.0 to 1.0 with no weight update.

### Per-tenant BM25 sub-index (policy, for completeness)

```
function TENANT_BM25(query, t):
    C_t   <- filter(corpus, user_id == t)        # filter BEFORE scoring: tenant-local IDF
    sig   <- (identity(C_t), length(C_t))        # auto-invalidates on corpus change
    index <- CACHE_BM25.get(sig)
    if index is None:
        try:
            index <- build_bm25s(C_t)            # lazy, on first query
            CACHE_BM25.put(sig, index, max_tenants=200, evict=LRU)
        except BuildError:
            return SHARED_INDEX_THEN_POSTFILTER(query, t)   # safe fallback
    return index.search(query, F=k)              # no over-fetch needed once tenant-local
```

## Appendix B. Glossary of Terms

- **Score compression (a corpus-specific, non-transferable reading).** The phenomenon in which a shared multilingual scorer maps a language's relevant-pair scores into a narrower band than another's. Observed in our small single-book v1 Arabic sample (cross-encoder std AR 0.348 vs EN 0.909, ~2.6×) but **not** on a clean, larger, matched-pipeline sample, where the deployed reranker makes Arabic *wider* (§7.9) — i.e. compression was a corpus-homogeneity artifact, not a language law. We retain the term only to name that initial observation.
- **Discriminability.** The separation between the relevant and irrelevant score distributions; the quantity a threshold actually exploits. Compression shrinks it for Arabic even though Arabic's relevant mean is not lower on the clean corpus (the direction is model-dependent; §7.8).
- **Tokenization fertility (subword penalty).** Average number of subword tokens per word under the scorer's own tokenizer. Measured AR 1.946 vs EN 1.536 (1.27x), a plausible contributor to compression but not asserted as its sole cause.
- **Self-match anchor (`S_in`).** Score of a passage-derived query against its own source chunk; a guaranteed-relevant *upper* anchor on relevance. A deliberate idealisation, not a natural-question model (§9, T2).
- **Cross-document / cross-book anchor (`S_out`).** Score of a query against a chunk from a different book; an *irrelevant lower* anchor.
- **`out_hi`, `in_lo`.** The Q75 of `S_out` and the Q25 of `S_in` respectively; the two percentile anchors the gate uses. The gap test is `in_lo > out_hi`.
- **`tau_t` (per-tenant cutoff).** The relevance threshold for tenant `t`: `tau_t = out_hi + α·(in_lo − out_hi)`, clamped. Canonical cross-encoder values: T1 −2.93, T2 −2.75, T3 −1.59, T4 −1.39, T5 −3.29; cosine: T1 0.382, T2 0.329, T3 0.409, T4 0.302, T5 0.325.
- **α (alpha).** Interpolation weight placing the cutoff inside the gap; operating point α = 0.25.
- **Self-calibrating gate.** The label-free, training-free mechanism (Algorithm 1) that calibrates `tau_t` from a tenant's own score geometry; never inspects language.
- **Tenant.** An isolated per-user (or per-library) partition of the corpus, addressed by `user_id`. Tenant sizes here span two orders of magnitude (T5 = 45 chunks to T4 = 2,733).
- **Frozen model.** A public checkpoint used with no weight updates. All three mitigations operate over frozen models.
- **Emergent fairness (engineering property).** Fairness behaviour that arises from system-layer mechanisms over frozen models rather than from retraining; the paper's thesis.
- **Cross-script glossary.** A bilingual JSON dictionary (308 AR→EN + 280 EN→AR pairs) used for additive, recall-monotone query expansion at ~48 µs with no model (Algorithm 2).
- **Structural lexical floor.** The fact that an Arabic query can never lexically match an English passage, making AR→EN BM25 recall exactly 0 by construction — the lexical-side floor the glossary lifts to 1.0.
- **Continual one-shot acquisition.** Learning a new term from a single retrieval miss by writing a translation pair into the glossary JSON (never into weights); auditable and revertible.
- **Tenant starvation.** The collapse of a minority tenant's retrieval (overlap@5 to 0.461 at 98.5% dominance) when it shares vocabulary with a dominant tenant in a shared index. Driven by vocabulary overlap, not language; fixed to 1.0 by a per-tenant sub-index. Dense retrieval starves worse (0.322) than sparse.
- **Per-tenant sub-index.** A BM25 index built over a single tenant's filtered chunks, so IDF/document-frequency statistics become tenant-local; built lazily and RAM-cached with auto-invalidation.
- **Oracle (label-tuned).** A threshold tuned on labels, hence undeployable without them. Global-fixed oracle F1 1.000; per-tenant oracle F1 0.992. The label-free gate (0.968) is *near*-oracle and is not claimed to beat these.
- **Off-topic acceptance.** An irrelevant passage admitted past the gate. The fixed −5.0 default admits 25; the calibrated gate admits 4.


---

## References

[1] Abdul-Mageed, M., Elmadany, A., Nagoudi, E. M. B. (2021). ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic. *ACL-IJCNLP 2021*, pp. 7088–7105. <https://doi.org/10.18653/v1/2021.acl-long.551>

[2] Abdul-Mageed, M., Zhang, C., Elmadany, A., Bouamor, H., Habash, N. (2021). NADI 2021: The Second Nuanced Arabic Dialect Identification Shared Task. *WANLP 2021*, pp. 244–259. <https://aclanthology.org/2021.wanlp-1.28/>

[3] Adeyemi, M., Oladipo, A., Pradeep, R., Lin, J. (2024). Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages. *ACL 2024 (Short Papers)*, pp. 650–656. <https://doi.org/10.18653/v1/2024.acl-short.59>

[4] Ahia, O., Kumar, S., Gonen, H., Kasai, J., Mortensen, D. R., Smith, N. A., Tsvetkov, Y. (2023). Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models. *EMNLP 2023*, pp. 9904–9923. <https://doi.org/10.18653/v1/2023.emnlp-main.614>

[5] Antoun, W., Baly, F., Hajj, H. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding. *OSACT4 (LREC 2020)*. <https://aclanthology.org/2020.osact-1.2/>

[6] Arampatzis, A., Kamps, J., Robertson, S. (2009). Where to Stop Reading a Ranked List? Threshold Optimization Using Truncated Score Distributions. *SIGIR 2009*. <https://doi.org/10.1145/1571941.1572031>

[7] Arampatzis, A., Robertson, S. (2011). Modeling Score Distributions in Information Retrieval. *Information Retrieval* 14(1), 26–46. <https://doi.org/10.1007/s10791-010-9145-5>

[8] Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*. <https://arxiv.org/abs/2310.11511>

[9] Bechiri, E. B., Lanasri, D. (2026). DziriBOT: RAG Based Intelligent Conversational Agent for Algerian Arabic Dialect. *arXiv:2602.02270*. <https://arxiv.org/abs/2602.02270>

[10] Bonifacio, L., Jeronymo, V., Abonizio, H. Q., Campiotti, I., Fadaee, M., Lotufo, R., Nogueira, R. (2021). mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset. *arXiv:2108.13897*. <https://doi.org/10.48550/arXiv.2108.13897>

[11] Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). *ACL 2020*, pp. 8440–8451. <https://doi.org/10.18653/v1/2020.acl-main.747>

[12] Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*, pp. 758–759. <https://doi.org/10.1145/1571941.1572114>

[13] da Silva de Oliveira, M. V., de Andrade Silva, J., de Lima Fontão, A. (2025). Fairness Testing in Retrieval-Augmented Generation: How Small Perturbations Reveal Bias in Small Language Models. *arXiv:2509.26584*. <https://doi.org/10.48550/arXiv.2509.26584>

[14] Feng, F., Yang, Y., Cer, D., Arivazhagan, N., Wang, W. (2022). Language-agnostic BERT Sentence Embedding (LaBSE). *ACL 2022*, pp. 878–891. <https://doi.org/10.18653/v1/2022.acl-long.62>

[15] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., Wang, H. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*. <https://doi.org/10.48550/arXiv.2312.10997>

[16] Gollapudi, S., Karia, N., Sivashankar, V., Krishnaswamy, R., Begwani, N., Raz, S., Lin, Y., Zhang, Y., Mahapatro, N., Srinivasan, P., Singh, A., Simhadri, H. V. (2023). Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters. *WWW 2023*, pp. 3406–3416. <https://doi.org/10.1145/3543507.3583552>

[17] Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020*. <https://arxiv.org/abs/2002.08909>

[18] Habash, N. Y. (2010). Introduction to Arabic Natural Language Processing. *Synthesis Lectures on Human Language Technologies*, Morgan & Claypool. <https://doi.org/10.2200/S00277ED1V01Y201008HLT010>

[19] Hu, M., Wu, H., Guan, Z., Zhu, R., Guo, D., Qi, D., Li, S. (2024). No Free Lunch: Retrieval-Augmented Generation Undermines Fairness in LLMs, Even for Vigilant Users. *arXiv:2410.07589*. <https://doi.org/10.48550/arXiv.2410.07589>

[20] Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., Grave, E. (2022). Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever). *TMLR*. <https://arxiv.org/abs/2112.09118>

[21] Johnson, J., Douze, M., Jégou, H. (2021). Billion-Scale Similarity Search with GPUs (FAISS). *IEEE Transactions on Big Data* 7(3), 535–547. <https://doi.org/10.1109/TBDATA.2019.2921572>

[22] Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*, pp. 6769–6781. <https://doi.org/10.18653/v1/2020.emnlp-main.550>

[23] Keleg, A., Goldwater, S., Magdy, W. (2023). ALDi: Quantifying the Arabic Level of Dialectness of Text. *EMNLP 2023*, pp. 10597–10611. <https://doi.org/10.18653/v1/2023.emnlp-main.655>

[24] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. <https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html>

[25] Lin, J., Alfonso-Hermelo, D., Jeronymo, V., Kamalloo, E., Lassance, C., Nogueira, R., Ogundepo, O., Rezagholizadeh, M., Thakur, N., Yang, J.-H., Zhang, X. (2023). Simple Yet Effective Neural Ranking and Reranking Baselines for Cross-Lingual Information Retrieval. *arXiv:2304.01019*. <https://doi.org/10.48550/arXiv.2304.01019>

[26] Lù, X. H. (2024). BM25S: Orders of Magnitude Faster Lexical Search via Eager Sparse Scoring. *arXiv:2407.03618*. <https://doi.org/10.48550/arXiv.2407.03618>

[27] Malkov, Yu. A., Yashunin, D. A. (2020). Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (HNSW). *IEEE TPAMI* 42(4), 824–836. <https://doi.org/10.1109/TPAMI.2018.2889473>

[28] Manmatha, R., Rath, T. M., Feng, F. (2001). Modeling Score Distributions for Combining the Outputs of Search Engines. *SIGIR 2001*. <https://doi.org/10.1145/383952.384005>

[29] Nie, J.-Y. (2010). Cross-Language Information Retrieval. *Synthesis Lectures on Human Language Technologies*, Morgan & Claypool. <https://doi.org/10.2200/S00266ED1V01Y201005HLT008>

[30] Nogueira, R., Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*. <https://doi.org/10.48550/arXiv.1901.04085>

[31] Obeid, O., Zalmout, N., Khalifa, S., Taji, D., Oudah, M., Alhafni, B., Inoue, G., Eryani, F., Erdmann, A., Habash, N. (2020). CAMeL Tools: An Open Source Python Toolkit for Arabic Natural Language Processing. *LREC 2020*. <https://aclanthology.org/2020.lrec-1.868/>

[32] Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics* 9(1), 62–66. <https://doi.org/10.1109/TSMC.1979.4310076>

[33] Park, T., Lee, G., Kim, M.-S. (2025). MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG. *arXiv:2507.01079*. <https://doi.org/10.48550/arXiv.2507.01079>

[34] Patel, L., Kraft, P., Guestrin, C., Zaharia, M. (2024). ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data. *PACMMOD* 2(3) (SIGMOD 2024). <https://doi.org/10.1145/3654923>

[35] Penha, G., Hauff, C. (2021). On the Calibration and Uncertainty of Neural Learning to Rank Models for Conversational Search. *EACL 2021*. <https://doi.org/10.18653/v1/2021.eacl-main.12>

[36] Petrov, A., La Malfa, E., Torr, P. H. S., Bibi, A. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. *NeurIPS 2023* (arXiv:2305.15425). <https://proceedings.neurips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html>

[37] Robertson, S. E., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval* 3(4), 333–389. <https://doi.org/10.1561/1500000019>

[38] Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., Gurevych, I. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. *ACL-IJCNLP 2021*, pp. 3118–3135. <https://doi.org/10.18653/v1/2021.acl-long.243>

[39] Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *NeurIPS 2020*. <https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>

[40] Wang, Y., Wei, L., Ling, H. (2025). Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG (TARG). *arXiv:2511.09803*. <https://doi.org/10.48550/arXiv.2511.09803>

[41] Wu, S., Dredze, M. (2020). Are All Languages Created Equal in Multilingual BERT? *RepL4NLP 2020*, pp. 120–130. <https://doi.org/10.18653/v1/2020.repl4nlp-1.16>

[42] Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H. (2024). Corrective Retrieval Augmented Generation (CRAG). *arXiv:2401.15884*. <https://doi.org/10.48550/arXiv.2401.15884>

[43] Zhang, Z., Li, N., Liu, Q., Li, R., Gao, W., Mao, Q., Huang, Z., Yu, B., Tao, D. (2025). The Other Side of the Coin: Exploring Fairness in Retrieval-Augmented Generation. *arXiv:2504.12323*. <https://doi.org/10.48550/arXiv.2504.12323>

[44] Yousef, A. K. (2026). Self-Calibrating Per-Tenant Relevance Gating with a Conditional Cross-Lingual Fallback for Offline Multilingual RAG. *Maktaba technical-paper series (flagship)*. Zenodo. <https://doi.org/10.5281/zenodo.20688577>

[45] Yousef, A. K. (2026). Sparse-Retrieval Tenant Starvation: A Lexical Post-Filtering Failure Mode in Shared-Index Multi-Tenant Retrieval, and a Per-Tenant Sub-Index Remedy. *Maktaba technical-paper series (P1)*.

[46] Yousef, A. K. (2026). A Self-Extending Bilingual Glossary for Offline Cross-Script Retrieval: Continual, Label-Free Term Acquisition Without Model Weight Updates. *Maktaba technical-paper series (P2)*.
