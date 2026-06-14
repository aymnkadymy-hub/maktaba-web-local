"""
User fact extraction — Mem0-inspired, uses existing ChromaDB + offline_llm.

After each exchange (async, fire-and-forget):
  1. LLM extracts 0-3 short facts about the user from the exchange
  2. Facts are stored in ChromaDB with fact_type="user_fact" metadata

At query time:
  get_user_facts(user_id, query) → retrieves relevant facts via similarity search
  Result included in the context sent to the LLM.

Facts extracted:
  - Domain / specialization ("متخصص في الشبكات")
  - Books of interest ("يقرأ كتاب TCP/IP")
  - Recurring topics ("يسأل كثيراً عن البروتوكولات")
  - Preferences ("يفضل الشرح بالأمثلة العملية")

Deduplication: ChromaDB similarity search prevents storing near-identical facts.
"""
import json
import time
import logging
from typing import List

logger = logging.getLogger("fact_extractor")


def _extract_json_array(text: str) -> list:
    """Find the first well-formed JSON array using bracket-matching (robust to stray brackets)."""
    pos = 0
    while True:
        start = text.find('[', pos)
        if start == -1:
            return []
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        pass
                    pos = start + 1
                    break
        else:
            break
    return []


_MIN_EXCHANGE_LEN = 30   # skip very short exchanges (greetings, etc.)
_FACT_K           = 3    # facts to retrieve per query
_DEDUP_THRESHOLD  = 0.92 # cosine similarity above this = duplicate, skip

_EXTRACT_SYSTEM = """\
You are a memory extraction assistant. Given one Q&A exchange, extract 0-3 short facts
about the USER (their interests, expertise, books they mentioned, domain, preferences).
Output ONLY a valid JSON array of short strings (Arabic or English matching the exchange).
If nothing notable: output [].
Examples: ["متخصص في الشبكات", "يقرأ كتاب TCP/IP", "يفضل الشرح بالأمثلة"]"""


def _extract_with_llm(user_msg: str, bot_msg: str) -> List[str]:
    # Use Ollama ONLY — never Groq, to avoid burning API credits after every exchange.
    try:
        from backend.llm.offline_llm import _ping_ollama, _ollama_call
        if not _ping_ollama():
            return []   # Ollama absent — skip silently
        exchange = (
            f"المستخدم: {user_msg[:400]}\n"
            f"البوت: {bot_msg[:400]}"
        )
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": exchange},
        ]
        raw   = _ollama_call(messages).strip()
        facts = _extract_json_array(raw)
        if facts:
            return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    except Exception as e:
        logger.debug(f"Fact extraction LLM error: {e}")
    return []


def _user_fact_filter(user_id: str):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    return Filter(must=[
        FieldCondition(key="metadata.fact_type", match=MatchValue(value="user_fact")),
        FieldCondition(key="metadata.user_id",   match=MatchValue(value=user_id)),
    ])


def _is_duplicate(fact: str, user_id: str) -> bool:
    try:
        from backend.database.vector_db import vector_store
        # similarity_search_with_score returns List[Tuple[Document, float]]
        # score is cosine similarity in [0, 1] — higher means more similar
        results = vector_store.similarity_search_with_score(
            fact, k=1, filter=_user_fact_filter(user_id)
        )
        if results and results[0][1] >= _DEDUP_THRESHOLD:
            return True
    except Exception:
        pass
    return False


def _store_facts(facts: List[str], user_id: str, session_id: str):
    if not facts:
        return
    try:
        from backend.database.vector_db import vector_store
        from langchain_core.documents import Document

        new_docs = []
        for fact in facts:
            if not _is_duplicate(fact, user_id):
                new_docs.append(Document(
                    page_content=fact,
                    metadata={
                        "fact_type":  "user_fact",
                        "user_id":    user_id,
                        "session_id": session_id,
                        "ts":         int(time.time()),
                    },
                ))

        if new_docs:
            vector_store.add_documents(new_docs)
            logger.debug(f"Stored {len(new_docs)} new facts for user '{user_id}'")
    except Exception as e:
        logger.debug(f"Fact storage error: {e}")


def get_user_facts(user_id: str, query: str) -> str:
    """
    Retrieve relevant user facts.
    Fast path: FAISS with user_fact+user_id filter (~30ms).
    Fallback: return "" — never block on Qdrant brute-force (would take 13+ s).
    """
    try:
        from backend.database.vector_db import faiss_index, embeddings as _emb

        if not faiss_index.ready:
            return ""   # FAISS building — skip facts to avoid blocking chat

        query_vec = _emb.embed_query(query)

        # fetch_multiplier=2000 → scan ~6000 candidates; user_facts are rare (<0.01%)
        results = faiss_index.search(
            query_vec,
            k=_FACT_K,
            filter_fn=lambda meta: (
                meta.get("fact_type") == "user_fact"
                and meta.get("user_id") == user_id
            ),
            threshold=0.0,
            fetch_multiplier=2000,
        )
        if results:
            facts = [doc.page_content for doc, _ in results]
            return "معلومات عن المستخدم:\n" + "\n".join(f"- {f}" for f in facts)
    except Exception as e:
        logger.debug(f"Fact retrieval error: {e}")
    return ""


def extract_and_store_async(
    user_msg: str, bot_msg: str, user_id: str, session_id: str
):
    """
    Fire-and-forget — called after each exchange.
    Skips very short exchanges that carry no useful facts.
    Delays a few seconds (see the sleep below) so the next user message isn't
    blocked by a concurrent LLM call.
    """
    if len(user_msg) + len(bot_msg) < _MIN_EXCHANGE_LEN:
        return

    def _run():
        import time as _t
        _t.sleep(3)   # yield to the LLM stream; 3 s is enough for most responses (was 8)
        facts = _extract_with_llm(user_msg, bot_msg)
        if facts:
            _store_facts(facts, user_id, session_id)

    from backend.rag.native_embeddings import submit_background_task
    submit_background_task(_run)
