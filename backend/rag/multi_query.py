"""
Multi-query expansion for improved retrieval recall.

Two modes:
  Template (always): strips question starters, normalises Arabic alef variants,
                     produces 2-4 variants with zero latency.
  LLM (optional):    uses offline Ollama to rephrase the query 2 more ways.
                     Only fires when use_llm=True AND Ollama is running.
"""
import re
import logging
from typing import List

logger = logging.getLogger("multi_query")

_AR_Q = re.compile(
    r'^(ما هو|ما هي|كيف يعمل|كيف تعمل|كيف|لماذا|متى|أين|من هو|من هي|ماذا'
    r'|اشرح|شرح|عرّف|عرف|اذكر|ما معنى|ما المقصود بـ?|ما الفرق بين'
    r'|قارن بين|ما هي خطوات|كيف يمكن)\s+',
    flags=re.IGNORECASE,
)
_EN_Q = re.compile(
    r'^(what is|what are|how does|how do|why is|why are|explain|define'
    r'|describe|tell me about|what does|compare|difference between'
    r'|steps to|how to)\s+',
    flags=re.IGNORECASE,
)


def expand_query(query: str, use_llm: bool = False) -> List[str]:
    """
    Returns a list of query variants.  Original is always first.
    Max 5 variants total.
    """
    q       = query.strip()
    variants: List[str] = [q]
    seen    = {q.lower()}

    def _add(v: str):
        v = v.strip()
        if v and v.lower() not in seen and len(v) > 2:
            variants.append(v)
            seen.add(v.lower())

    # 1. Strip question opener → keyword form
    kw = _AR_Q.sub('', q).strip()
    if kw == q:
        kw = _EN_Q.sub('', q).strip()
    if kw != q:
        _add(kw)
        # also normalise alef inside keyword form
        _add(re.sub(r'[أإآٱ]', 'ا', kw))

    # 2. Normalise alef in full query  (catches "أحمد" vs "احمد" mismatches)
    _add(re.sub(r'[أإآٱ]', 'ا', q))

    # 3. Strip common Arabic prefixes (وال / فال / بال / كال)
    stripped = re.sub(r'\b[وفبكل]ال', 'ال', kw or q)
    _add(stripped)

    # 4. Optional LLM rephrasings
    if use_llm and len(variants) < 5:
        _add_llm_variants(q, variants, seen, max_new=5 - len(variants))

    return variants[:5]


def _add_llm_variants(
    query: str,
    variants: list,
    seen: set,
    max_new: int = 2,
):
    try:
        import backend.llm.offline_llm as llm
        if not llm._ping_ollama():
            return

        result = llm.chat(
            system=(
                "You are a search-query rephrasing engine. "
                "Output ONLY rephrased queries, one per line, no numbering. "
                "Match the language of the input exactly."
            ),
            user=(
                f"Rephrase this search query {max_new} different ways:\n{query}"
            ),
        )
        for line in result.strip().splitlines():
            line = re.sub(r'^[\d\-\.\)]+\s*', '', line).strip()
            if line and line.lower() not in seen:
                variants.append(line)
                seen.add(line.lower())
    except Exception as e:
        logger.debug(f"LLM query expansion skipped: {e}")
