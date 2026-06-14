"""
HyDE — Hypothetical Document Embeddings.

Problem being solved:
  A user's question ("ما هي الذاكرة العشوائية؟") lives in "question embedding space".
  Book passages live in "passage embedding space".
  The two spaces don't perfectly overlap, so cosine similarity misses good passages.

HyDE solution:
  1. Ask the LLM to write a short hypothetical book passage that answers the question.
  2. Embed *that passage* (it now lives in passage embedding space).
  3. Use that embedding to search the vector store.
  → Retrieval lands closer to real book passages.

Requirements:
  - Ollama running locally (skipped gracefully if not).
  - embed_query function from the active embeddings model.
"""
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger("hyde")

_SYSTEM = (
    "You are a passage-generation assistant for a book-search system. "
    "Write a single factual passage (80-120 words) as if it were extracted "
    "verbatim from a technical book answering the question. "
    "Write in the SAME LANGUAGE as the question. "
    "No introduction, no 'According to...', no conclusion — just the passage itself."
)


def _generate_passage(query: str) -> str | None:
    try:
        import backend.llm.offline_llm as llm
        if not llm._ping_ollama():
            return None
        passage = llm.chat(system=_SYSTEM, user=f"Question: {query}")
        passage = passage.strip()
        return passage if len(passage) > 40 else None
    except Exception as e:
        logger.debug(f"HyDE generation skipped: {e}")
        return None


def retrieve_with_hyde(
    query: str,
    vector_store,
    k: int = 12,
) -> List[Document]:
    """
    Returns top-k documents retrieved via the hypothetical passage embedding.
    Returns [] if Ollama is not running or generation fails — caller uses
    regular BM25+vector results as primary.
    """
    passage = _generate_passage(query)
    if passage is None:
        return []

    try:
        from backend.database.vector_db import embeddings as _emb
        vec  = _emb.embed_query(passage)
        docs = vector_store.similarity_search_by_vector(vec, k=k)
        logger.debug(f"HyDE: {len(docs)} docs via hypothetical passage")
        return docs
    except Exception as e:
        logger.warning(f"HyDE vector search failed: {e}")
        return []
