"""
Cross-encoder re-ranking for retrieved passages.

Why cross-encoders beat bi-encoders for re-ranking:
  Bi-encoders (our embeddings) encode query and passage independently.
  Cross-encoders process (query, passage) JOINTLY — the model attends across both,
  capturing fine-grained relevance signals that bi-encoders miss.

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - 52 MB, multilingual (100 languages incl. Arabic + English)
  - Trained on MS MARCO passage ranking
  - ~30-80 ms for 20 pairs on CPU

Usage in the pipeline:
  BM25 + Vector + HyDE → RRF top-20 → cross-encoder rerank → top-6 returned

Falls back to RRF ordering if model is unavailable (no crash, no degradation).
"""
import os
import logging
from typing import List, Optional
from langchain_core.documents import Document

logger  = logging.getLogger("reranker")
_MODEL  = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
_ce     = None
_tried  = False


def _load() -> Optional[object]:
    global _ce, _tried
    if _tried:
        return _ce
    _tried = True
    try:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ce = CrossEncoder(_MODEL, max_length=256, device=device)
        logger.info(f"Cross-encoder ready: {_MODEL} [{device.upper()}]")
    except Exception as e:
        logger.warning(f"Cross-encoder unavailable ({e}) — reranking disabled")
    return _ce


def score_pairs(pairs: List[tuple]) -> List[float]:
    """Raw cross-encoder relevance scores for (query, passage) pairs.

    Exposed for the self-calibrating relevance gate, which needs the score
    DISTRIBUTION over a tenant's content — not a ranked list. Returns [] if the
    model is unavailable."""
    model = _load()
    if model is None or not pairs:
        return []
    try:
        return [float(s) for s in model.predict(list(pairs), show_progress_bar=False)]
    except Exception as e:
        logger.warning(f"score_pairs failed: {e}")
        return []


def warm_up():
    """Load model AND run a dummy prediction to trigger PyTorch JIT compilation."""
    model = _load()
    if model is not None:
        try:
            # Dummy prediction — forces PyTorch to compile the compute graph
            # so the first real request doesn't pay this ~1s one-time cost.
            model.predict([("warmup", "warmup")], show_progress_bar=False)
            logger.info("Cross-encoder warm-up prediction done")
        except Exception as e:
            logger.debug(f"Warm-up prediction skipped: {e}")


# Cross-encoder relevance cutoff. Pairs scoring below this are dropped as
# off-topic, so an unrelated query (no matching book) returns nothing and the
# caller answers from general knowledge instead of forcing a bad book into the
# prompt. Calibrated on the live corpus: genuine matches scored >= -4.3 while
# clear mismatches (e.g. an Arabic programming query vs an English economics
# chunk) scored <= -5.5. Default -5.0 sits in that gap; tune via RERANK_MIN_SCORE.
_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-5.0"))


def rerank(query: str, docs: List[Document], top_k: int = 6,
           min_score: Optional[float] = None) -> List[Document]:
    """
    Score every (query, doc) pair with the cross-encoder, drop pairs below the
    relevance cutoff, and return the top_k survivors sorted by score.

    Input: top ~20 candidates from RRF.
    Output: the best top_k that clear `min_score` (may be empty → no book match).
    """
    if not docs:
        return docs

    model = _load()
    if model is None:
        return docs[:top_k]

    cutoff = _MIN_SCORE if min_score is None else min_score
    try:
        # Passage truncation: ~250 chars is enough signal for the cross-encoder
        pairs  = [(query, doc.page_content[:250]) for doc in docs]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
        kept   = [(s, d) for s, d in ranked if float(s) >= cutoff]

        if logger.isEnabledFor(logging.INFO):
            top = ", ".join(f"{float(s):.2f}" for s, _ in ranked[:top_k])
            logger.info(f"[RERANK] top scores: {top} | cutoff={cutoff} "
                        f"| kept {len(kept)}/{len(ranked)}")

        return [doc for _, doc in kept[:top_k]]
    except Exception as e:
        logger.warning(f"Reranking failed: {e} — using RRF order")
        return docs[:top_k]
