"""
Rolling conversation summary — token-aware context manager.

How it works:
  - Last RECENT_MSGS messages are always included verbatim
  - Older messages get summarized by the LLM into ~80 words
  - Summary is stored in PostgreSQL as role='summary' and reused until
    SUMMARIZE_EVERY new exchanges accumulate past the window
  - Everything runs within TOKEN_BUDGET chars of context

Token estimation (Arabic-aware):
  Arabic chars (/2.5) + other chars (/4.0) — avoids the word.split() mistake
  since Arabic words like "وبالذاكرة" are 1 word but 4-5 tokens.
"""
import logging
from typing import List, Dict

logger = logging.getLogger("summarizer")

RECENT_MSGS     = 6     # always verbatim
SUMMARIZE_EVERY = 4     # regenerate summary every N exchanges past window
TOKEN_BUDGET    = 1400  # max tokens for history block sent to LLM

# Tracks the last message count at which summarization was triggered per session.
# Prevents re-firing the same trigger when the count hasn't changed between requests.
_summarized_at: dict[str, int] = {}


def estimate_tokens(text: str) -> int:
    """Arabic-aware character-based token estimate."""
    arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')
    other  = len(text) - arabic
    return int(arabic / 2.5 + other / 4.0)


def _format_messages(messages: List[Dict]) -> str:
    return "\n".join(
        f"{'المستخدم' if m['role'] == 'user' else 'البوت'}: {m['content']}"
        for m in messages
    )


def _summarize_with_llm(messages: List[Dict]) -> str:
    try:
        from backend.llm.offline_llm import chat
        text   = _format_messages(messages)
        system = "أنت مساعد يلخص المحادثات. الخلاصة يجب أن تكون موجزة وواضحة."
        prompt = f"لخّص المحادثة التالية في 60-80 كلمة محتفظاً بالنقاط الجوهرية:\n\n{text}"
        return chat(system, prompt).strip()
    except Exception as e:
        logger.debug(f"Summarizer LLM error: {e}")
        return ""


async def get_context_smart(session_id: str) -> str:
    """
    Return history string within TOKEN_BUDGET:
      [ملخص المحادثة السابقة: ...]   ← if summary exists
      المستخدم: ...
      البوت: ...                      ← last RECENT_MSGS messages
    Triggers background summarization when needed.
    """
    try:
        from backend.memory.storage import (
            get_session_context, get_latest_summary, get_message_count,
        )

        count   = await get_message_count(session_id)
        recent  = await get_session_context(session_id, limit=RECENT_MSGS)
        summary = await get_latest_summary(session_id)

        parts: List[str] = []
        if summary:
            parts.append(f"[ملخص المحادثة السابقة: {summary}]")
        if recent:
            parts.append(_format_messages(recent))

        context = "\n\n".join(parts)

        # Enforce token budget — truncate from the start if over
        if estimate_tokens(context) > TOKEN_BUDGET:
            # Keep only recent messages if summary+recent is still too long
            context = _format_messages(recent)

        # Trigger async summarization when history grows beyond window.
        # count is message count (2 per exchange), so divide by 2 first.
        # _summarized_at guard: skip if we already triggered at this exact count
        # (prevents multiple concurrent requests firing duplicate background tasks).
        older_count     = count - RECENT_MSGS
        older_exchanges = older_count // 2
        if (older_exchanges > 0
                and older_exchanges % SUMMARIZE_EVERY == 0
                and _summarized_at.get(session_id) != count):
            _summarized_at[session_id] = count
            _trigger_summarization(session_id, count)

        return context

    except Exception as e:
        logger.debug(f"get_context_smart error: {e}")
        return ""


def _trigger_summarization(session_id: str, total_count: int):
    """Fire-and-forget background task — runs on C++ thread pool, never blocks."""
    def _run():
        import asyncio
        loop = asyncio.new_event_loop()   # Windows-safe: explicit new loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_do_summarize(session_id, total_count))
        except Exception as e:
            logger.debug(f"Background summarization error: {e}")
        finally:
            loop.close()

    from backend.rag.native_embeddings import submit_background_task
    submit_background_task(_run)


async def _do_summarize(session_id: str, total_count: int):
    """Fetch older messages, summarize, and persist."""
    from backend.memory.storage import get_session_context, save_summary

    all_msgs = await get_session_context(session_id, limit=total_count)
    older    = all_msgs[:-RECENT_MSGS] if len(all_msgs) > RECENT_MSGS else []
    if not older:
        return

    summary = _summarize_with_llm(older)
    if summary:
        await save_summary(session_id, summary)
        logger.debug(f"Summary saved for session {session_id[:8]}…")
