"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.
Paper: https://arxiv.org/abs/2401.18059

Problem: Long books contain chapters/sections the user queries at different
granularities.  Flat chunking misses high-level ("describe the book's
approach to X") and cross-chapter ("how do chapters 3 and 7 relate?") queries.

Solution — summary tree built AT INGESTION TIME:

  Level 0 : original leaf chunks  (already in ChromaDB)
  Level 1 : cluster L0 chunks → summarise each cluster  (chapter-level)
  Level 2 : cluster L1 summaries → summarise each cluster (book-level)

At QUERY TIME, the hybrid retriever searches all levels transparently.
RAPTOR summaries carry metadata raptor_level=1/2 so context labels them clearly.

Requirements:
  - sklearn (already in requirements)
  - Ollama running locally (skipped if not available)
  - embed_query from active embeddings model
"""
import logging
import numpy as np
from typing import List, Optional
from langchain_core.documents import Document

logger = logging.getLogger("raptor")

# ── Tuning knobs ──────────────────────────────────────────────────────────────
_MIN_DOCS_FOR_RAPTOR = 8    # skip book if fewer leaf chunks than this
_MIN_CLUSTER_SIZE    = 3    # discard clusters smaller than this
_MAX_L1_CLUSTERS     = 6    # was 12 — halved to cut RAPTOR time by ~50% (6 LLM calls vs 14)
_SUMMARY_WORD_LIMIT  = "80-100"  # was "120-180" — shorter summaries = faster generation


# ── LLM summarisation ─────────────────────────────────────────────────────────

def _summarise_cluster(
    docs: List[Document],
    book_title: str,
    level: int,
) -> Optional[str]:
    """
    Summarise a cluster of documents into one paragraph using offline LLM.
    Returns None if Ollama unavailable or result too short.
    """
    try:
        import backend.llm.offline_llm as llm
        if not llm._ping_ollama():
            return None

        excerpts = "\n\n".join(d.page_content[:500] for d in docs[:8])
        lang_hint = (
            "Write in Arabic if the text is Arabic, English if the text is English."
        )
        result = llm.chat(
            system=(
                f"Summarise the following passages from '{book_title}' "
                f"into one precise paragraph ({_SUMMARY_WORD_LIMIT} words). "
                f"Capture the key ideas and technical concepts. "
                f"{lang_hint} No introductory phrases."
            ),
            user=excerpts,
        )
        result = result.strip()
        return result if len(result) > 50 else None
    except Exception as e:
        logger.debug(f"Summarisation failed: {e}")
        return None


# ── K-means clustering ────────────────────────────────────────────────────────

def _cluster_docs(
    docs: List[Document],
    n_clusters: int,
    embeddings_model,
) -> List[List[Document]]:
    """
    Embed docs and group by k-means.
    Returns clusters with at least _MIN_CLUSTER_SIZE members.
    """
    from sklearn.cluster import KMeans

    logger.debug(f"Embedding {len(docs)} docs for clustering…")
    # embed_documents batches all texts in one call — ~10x faster than N embed_query calls
    vecs   = np.array(embeddings_model.embed_documents([d.page_content for d in docs]))
    n      = min(n_clusters, len(docs))
    labels = KMeans(n_clusters=n, random_state=42, n_init="auto").fit_predict(vecs)

    groups: dict[int, List[Document]] = {}
    for doc, lbl in zip(docs, labels):
        groups.setdefault(int(lbl), []).append(doc)

    return [g for g in groups.values() if len(g) >= _MIN_CLUSTER_SIZE]


# ── One RAPTOR level ──────────────────────────────────────────────────────────

def _build_level(
    docs: List[Document],
    book_title: str,
    embeddings_model,
    level: int,
) -> List[Document]:
    """
    Cluster docs, summarise each cluster, return summary Documents.
    """
    if len(docs) < _MIN_DOCS_FOR_RAPTOR:
        logger.info(f"RAPTOR L{level}: only {len(docs)} docs — skipping")
        return []

    n_clusters = min(_MAX_L1_CLUSTERS, max(2, len(docs) // 7))
    logger.info(f"RAPTOR L{level}: {len(docs)} docs → {n_clusters} clusters")

    try:
        clusters = _cluster_docs(docs, n_clusters, embeddings_model)
    except Exception as e:
        logger.warning(f"RAPTOR clustering failed (L{level}): {e}")
        return []

    summaries: List[Document] = []
    for i, cluster in enumerate(clusters):
        text = _summarise_cluster(cluster, book_title, level)
        if not text:
            continue

        pages = sorted(
            {d.metadata.get("page", 0) for d in cluster if d.metadata.get("page")}
        )
        page_range = (
            f"{pages[0]}-{pages[-1]}" if len(pages) > 1
            else str(pages[0]) if pages
            else "?"
        )

        summaries.append(Document(
            page_content=text,
            metadata={
                "book_title":   book_title,
                "source":       f"RAPTOR-L{level}",
                "raptor_level": level,
                "page_range":   page_range,
                "cluster_id":   i,
                "cluster_size": len(cluster),
            },
        ))
        logger.debug(
            f"  Cluster {i}: {len(cluster)} docs "
            f"(pages {page_range}) → summary {len(text)} chars"
        )

    logger.info(
        f"RAPTOR L{level}: {len(summaries)}/{len(clusters)} "
        f"clusters summarised for '{book_title}'"
    )
    return summaries


# ── Public entry point ────────────────────────────────────────────────────────

def ingest_raptor(
    docs: List[Document],
    book_title: str,
    vector_store,
    max_levels: int = 2,
) -> int:
    """
    Build RAPTOR summary tree for a book and store summaries in vector_store.

    Args:
        docs:        leaf chunks for this book (from _run_ingestion)
        book_title:  used in metadata and prompts
        vector_store: ChromaDB instance
        max_levels:  how many summary layers to build (1 or 2 recommended)

    Returns:
        total number of summary chunks added (0 if Ollama unavailable)
    """
    # Gate: need Ollama for summarisation
    try:
        import backend.llm.offline_llm as llm
        if not llm._ping_ollama():
            logger.info(
                f"RAPTOR skipped for '{book_title}' — Ollama not running. "
                f"Start Ollama to enable hierarchical summaries."
            )
            return 0
    except Exception:
        return 0

    from backend.database.vector_db import embeddings as _emb

    current = docs
    total   = 0

    for level in range(1, max_levels + 1):
        layer = _build_level(current, book_title, _emb, level=level)
        if not layer:
            break
        vector_store.add_documents(layer)
        total   += len(layer)
        current  = layer   # next level clusters this level's summaries

    if total:
        logger.info(
            f"RAPTOR complete: {total} summary chunks added for '{book_title}'"
        )
    return total
