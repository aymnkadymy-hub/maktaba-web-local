"""
Chat endpoints: /chat, /chat/stream, /chat/image.
Registered in server_backend.py via make_chat_router().
"""
import os
import re
import uuid
import json
import queue
import asyncio
import threading
import time as _time_mod
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, AliasChoices

from backend.core.messages import _MSG

logger = logging.getLogger("backend")

# ── Canned small-talk responses ───────────────────────────────────────────────
_CANNED_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "small_talk_canned.json")
_canned_responses: dict[str, str] = {}
try:
    with open(_CANNED_PATH, encoding="utf-8") as _f:
        _canned_responses = {k.strip(): v for k, v in json.load(_f).items()}
    logger.info(f"Canned responses loaded: {len(_canned_responses)} entries")
except Exception as _e:
    logger.warning(f"Could not load canned responses: {_e}")

_ARABIC_DIACRITICS = re.compile(r'[ً-ٰٟ]')

def _lookup_canned(text: str) -> str | None:
    """Return a predefined response for exact/normalized matches, or None."""
    normalized = _ARABIC_DIACRITICS.sub('', text.strip().rstrip('!؟?،,.').strip())
    return _canned_responses.get(normalized) or _canned_responses.get(normalized.lower())

# ── Language detection (chat-only, not imported from server_backend) ──────────
try:
    from langdetect import detect as _langdetect
    def _detect_lang(text: str) -> str:
        try:
            return _langdetect(text)
        except Exception:
            return "ar"
except ImportError:
    def _detect_lang(text: str) -> str:
        arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')
        return "ar" if arabic / max(len(text), 1) > 0.2 else "en"

try:
    from backend.utils.language_guard import validate_language, APOLOGY_MSG
    _LANG_GUARD_OK = True
except Exception:
    _LANG_GUARD_OK = False
    def validate_language(_: str) -> bool:  # type: ignore[misc]
        return True
    APOLOGY_MSG = "عذراً، يرجى الكتابة بالعربية أو الإنجليزية."

# Flush stream every N tokens for immediate first-token latency
_STREAM_FLUSH_EVERY = 3

# Cap chat answer length — keeps replies concise and prevents the model from
# spiralling into long, incoherent output (with foreign-token bleed) over messy
# multi-book context. ~800 tokens ≈ 550 words, ample for a chat answer.
_CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "800"))

# ── Chinese-char sanitizer ────────────────────────────────────────────────────
_CHINESE_CHARS = re.compile(r'[一-鿿㐀-䶿　-〿＀-￯]')
# Cyrillic never appears legitimately in this app's Arabic/English answers, but
# qwen sometimes bleeds a stray Russian token ("мест", "память") into Arabic.
# Strip such runs as a safety net (repeat_penalty tuning reduced but didn't
# fully eliminate it). Latin is NOT stripped here — it can be valid (code, terms).
_CYRILLIC_RUN = re.compile(r'[Ѐ-ӿ]+')

# ── Latin-Arabic garbage detector ────────────────────────────────────────────
# Matches Arabizi transliteration garbage like "ya3ini", "7ashin", "sa7", "w2in"
# — tokens mixing Latin letters AND digits, which a model hallucinates when it
# confuses dialect examples with output. We deliberately do NOT strip pure
# alphabetic Latin words: "name", "number", "Python", "FAISS", "machine learning"
# are legitimate technical terms / code identifiers and must survive intact.
_LATIN_GARBAGE_RE = re.compile(
    r'(?<!\S)'                       # token start (whitespace-bounded)
    r'(?=[a-zA-Z0-9]*[a-zA-Z])'      # contains at least one letter
    r'(?=[a-zA-Z0-9]*\d)'            # AND at least one digit  → Arabizi
    r'[a-zA-Z0-9]{2,}'               # the mixed token itself
    r'(?:\s+[a-zA-Z0-9]+){0,4}'      # plus any trailing latin/num run
    r'(?!\S)',                       # token end
)

# Inline + fenced code is stashed before stripping so identifiers are never touched.
_CODE_SPAN_RE = re.compile(r'```[\s\S]*?```|`[^`\n]+`')

# Lowercase Latin glued directly onto an Arabic letter is a BPE bleed
# ("عcompter" for عدّاد, "سؤals" for سؤال). Uppercase acronyms glued to the
# Arabic article ("الCPU", "الAPI") are legitimate, so only lowercase runs of
# 2+ letters are stripped — and only when fused to Arabic with no space.
_ARABIC_GLUE_RE = re.compile(r'(?<=[؀-ۿ])[a-z]{2,}')


def _strip_latin_garbage(text: str) -> str:
    """
    Remove Arabizi transliteration garbage (latin+digit tokens) from Arabic
    responses. Code spans are preserved verbatim, and pure English words/terms
    are left untouched — only digit-mixed transliteration is removed.
    """
    if not text:
        return text
    ar = sum(1 for c in text if '؀' <= c <= 'ۿ')
    if ar / max(len(text), 1) < 0.4:
        return text   # mostly English — don't touch it

    # Stash code so variable names / identifiers are never stripped.
    code_spans: list[str] = []
    def _stash(m):
        code_spans.append(m.group(0))
        return f"\x00C{len(code_spans)-1}\x00"
    staged = _CODE_SPAN_RE.sub(_stash, text)

    # Preserve leading/trailing spaces — critical for streaming chunk boundaries.
    leading  = staged[0]  == ' ' if staged else False
    trailing = staged[-1] == ' ' if staged else False
    cleaned = _LATIN_GARBAGE_RE.sub('', staged)
    cleaned = _ARABIC_GLUE_RE.sub('', cleaned)        # strip Latin fused to Arabic
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    if not cleaned:
        cleaned = staged
    if leading:
        cleaned = ' ' + cleaned
    if trailing:
        cleaned = cleaned + ' '

    for i, span in enumerate(code_spans):
        cleaned = cleaned.replace(f"\x00C{i}\x00", span)
    return cleaned

_MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2000"))

# ── Prompt-injection guard ────────────────────────────────────────────────────
_INJECTION_RE = re.compile(
    # 1-2 qualifiers: catches both "ignore previous instructions" and the
    # canonical "ignore all previous instructions" phrasing
    r'ignore\s+(?:(?:previous|all|above|prior)\s+){1,2}(instructions?|context|prompt|system)|'
    r'forget\s+(all|everything|previous)|'
    r'disregard\s+(all|previous|above|prior)|'
    r'you\s+are\s+now\s+(?!a\s*(helpful|book|library))|'
    r'pretend\s+(you\s+are|to\s+be)\s+(?!helpful)|'
    r'act\s+as\s+(a\s+)?(?!helpful|book|library|assistant)|'
    r'<\s*/?\s*system\s*>|'
    r'\[INST\]|\[/?SYS\]|<<SYS>>',
    re.IGNORECASE,
)


def _check_injection(text: str) -> bool:
    """Return True if text contains a prompt-injection pattern."""
    return bool(_INJECTION_RE.search(text))


def _sanitize_response(text: str) -> str:
    if _CHINESE_CHARS.search(text):
        from backend.llm.offline_llm import _strip_chinese_preamble
        text = _strip_chinese_preamble(text)
        if _CHINESE_CHARS.search(text):
            leading  = text[0]  == ' ' if text else False
            trailing = text[-1] == ' ' if text else False
            text = _CHINESE_CHARS.sub('', text)
            text = re.sub(r'  +', ' ', text).strip()
            if leading:
                text = ' ' + text
            if trailing:
                text = text + ' '
    if _CYRILLIC_RUN.search(text):
        text = re.sub(r'\s*' + _CYRILLIC_RUN.pattern + r'\s*', ' ', text).strip()
    text = _strip_latin_garbage(text)
    return text


# ── Small-talk sub-categories ─────────────────────────────────────────────────
_FAREWELL = [
    "لا شكر", "لا، شكر", "خلاص", "بس ", "كافي", "وداع", "باي", "مو محتاج", "ماكو شي",
    "bye", "goodbye", "cya",
]
# Bare negations — when the user just says "لا" or "لأ" alone, treat as a refusal
_REFUSAL = frozenset(["لا", "لأ", "la", "nope", "no"])
_HOWRU = [
    "شخبار", "شلون", "كيفك", "كيفكم", "كيف حال", "كيف الحال",
    "how are you", "how r u", "how's it",
]
_WANTTALK = [
    "i want to talk", "i want to chat", "can we talk", "can we chat",
    "let's talk", "lets talk",
]
_SHAKO_MAKO = ["شكو ماكو", "شكو ما كو", "ماكو شكو", "لا شكو"]
_DIALECT_META = [
    # full forms
    "شلون ترد", "شلون تجاوب", "شلون تگول", "شلون تحچي", "شلون تحكي",
    "شو تگول", "شو تقول", "شو ترد", "كيف ترد", "كيف تجاوب",
    "احچي عراقي", "احكي عراقي", "تكلم عراقي",
    # contracted Iraqi forms: شت- = شلون ت-
    "شتجاوب", "شتگول", "شتقول", "شترد", "شتحچي", "شتحكي", "شتكلم",
    # hypothetical scenario triggers
    "إذا گالك", "اذا گالك", "إذا قالك", "اذا قالك",
    "لو گالك", "لو قالك", "إذا واحد گالك", "اذا واحد گالك",
]

# ── Correction / confusion signals ────────────────────────────────────────────
# When user sends one of these, the previous bot response was likely wrong —
# skip history to prevent contaminating the new response with old hallucinations.
_CORRECTION_SIGNALS = [
    "مو هيج", "مو صح", "هذا غلط", "غلط", "ما صح", "لا هيج", "مو كذا",
    "ما فهمت", "مو فاهم", "اشرح", "وضح", "شنو قصدك", "شو قصدك",
    "what do you mean", "thats wrong", "that's wrong", "incorrect",
]

def _extract_last_bot_reply(history: str) -> str:
    """Extract last bot reply from formatted history for anti-repeat injection."""
    lines = history.split('\n')
    for line in reversed(lines):
        if line.startswith('البوت: '):
            return line[7:].strip()[:80]
    return ""


def _is_correction_or_confusion(query: str) -> bool:
    """Return True if the user is correcting the bot or asking for clarification."""
    q = query.strip()
    if len(q) <= 6:   # very short — likely "شنو", "كيف", "وين", etc.
        return True
    q_low = q.lower()
    return any(sig in q_low for sig in _CORRECTION_SIGNALS)


# Concise English system prompt. The main system prompt is heavily Arabic/Iraqi,
# which dragged English answers back into Arabic — especially when the retrieved
# book context was Arabic. For English queries we swap to this so the model has
# no Arabic bias, and we skip dialectize() on the result.
ENGLISH_SYSTEM_PROMPT = (
    "You are a helpful library assistant. Reply in clear, correct English ONLY — "
    "never switch to Arabic. Answer directly and concisely. If book context is "
    "provided and relevant, answer from it and cite the book; otherwise answer "
    "from general knowledge without mentioning the books. Do not repeat the "
    "question back."
)


def _is_english_query(text: str) -> bool:
    """True when the message should be answered in English (used to pick the
    system prompt and to skip Iraqi dialect post-processing)."""
    return _lang_directive(text).startswith("Respond in English")


_FUSHA_DIRECTIVE = "أجب بالعربية الفصحى السليمة فقط، بدون أي كلمة عامية أو لهجة محلية."


def _lang_directive(text: str) -> str:
    # The user may write in ANY Arabic dialect — we always answer in Modern
    # Standard Arabic (Fusha). The model understands the dialect input on its own;
    # no dialect detection is needed (MARBERT was removed — it forced dialect output).
    s  = text.strip()
    ar = sum(1 for c in s if '؀' <= c <= 'ۿ')
    if ar == 0:
        return "Respond in English only."
    if ar / max(len(s), 1) > 0.15:
        return _FUSHA_DIRECTIVE
    lang = _detect_lang(s) if len(s) >= 15 else "ar"
    return _FUSHA_DIRECTIVE if lang == "ar" else "Respond in English only."


def _get_user_facts(query: str, user_id: str) -> str:
    try:
        from backend.memory.fact_extractor import get_user_facts
        return get_user_facts(user_id, query)
    except Exception:
        return ""


def _extract_source(context: str) -> str:
    m = re.search(r'\[([^\]—\-]+)', context)
    return m.group(1).strip() if m else ""


def _build_user_content(
    query: str, context: str, source: str, history: str,
    book_list: str, user_facts: str = ""
) -> str:
    directive  = _lang_directive(query)
    is_english = directive.startswith("Respond in English")

    if source == "small_talk":
        q_low     = query.strip().lower()
        q_bare    = q_low.rstrip("!؟?،,.").strip()
        is_refusal = (q_bare in _REFUSAL or any(w in q_low for w in _FAREWELL))
        is_shako   = any(w in q_low for w in _SHAKO_MAKO)
        is_dialect_meta = any(w in q_low for w in _DIALECT_META)

        if is_english:
            ctx_block = "[Greeting — reply in one short friendly English sentence]"
        elif is_refusal:
            ctx_block = "[المستخدم ودّع أو رفض — ودّعه بالفصحى بجملة قصيرة مثل 'حسناً، في أمان الله.' ثم توقف]"
        elif is_shako:
            ctx_block = "[المستخدم ألقى تحية عامية تعني 'لا جديد' — رد بالفصحى مباشرة: 'كل شيء على ما يرام، الحمد لله. كيف حالك؟']"
        elif any(w in q_low for w in _HOWRU):
            ctx_block = "[المستخدم يسأل عن حالك — رد بالفصحى بجملة واحدة: 'أنا بخير، الحمد لله. وأنت؟']"
        elif is_dialect_meta:
            ctx_block = (
                "[المستخدم يتحدث بالعامية أو يريد دردشة — افهم قصده ورد عليه بالعربية الفصحى "
                "بجملة قصيرة طبيعية ومتنوعة، بدون أي كلمة عامية. السؤال: " + query + "]"
            )
        elif any(w in q_low for w in _WANTTALK):
            ctx_block = "[المستخدم يريد محادثة — رحّب به بالفصحى واسأله عن الموضوع الذي يريده]"
        elif _is_correction_or_confusion(query):
            ctx_block = "[المستخدم يطلب توضيحاً — اسأله بالفصحى: 'ما الموضوع الذي تريد أن أوضّحه لك؟']"
        else:
            ctx_block = "[تحية — رد بجملة فصيحة قصيرة ومتنوعة، ممنوع 'كيف يمكنني مساعدتك']"
        book_list = ""
    elif source == "web":
        ctx_block = (
            f"[Web result — use it to answer]\n{context}" if is_english
            else f"[نتيجة من الإنترنت]\n{context}"
        )
    elif source == "books":
        ctx_block = (
            f"[Book context — use it if relevant, otherwise say the info is not in the books]\n{context}"
            if is_english else
            "[سياق من الكتب — استخدمه إذا كان ذا صلة بالسؤال فقط، "
            "وإذا لم يكن ذا صلة قل مباشرة 'هاي المعلومة مو موجودة بالكتب']\n"
            f"{context}"
        )
    else:
        ctx_block = (
            "[No context — answer from general knowledge, do not mention books]"
            if is_english else
            "[لا يوجد سياق — السؤال عام، أجب من معرفتك العامة مباشرة بدون أي ذكر للكتب]"
        )

    facts_block = f"\n\n{user_facts}" if user_facts else ""

    if history and not _is_correction_or_confusion(query):
        last_reply = _extract_last_bot_reply(history)
        anti_repeat = (
            f"\n[ممنوع تكرار هذا الرد بالضبط: {last_reply}]"
            if last_reply and len(last_reply) > 15 else ""
        )
        n_lines = history.count('\n') + 1
        topic_anchor = "\n[استمر في نفس موضوع المحادثة — لا تغيّر الموضوع]" if n_lines >= 4 else ""
        history_block = (
            f"\n\n<سجل_المحادثة>\n{history}\n</سجل_المحادثة>"
            f"{anti_repeat}{topic_anchor}"
        )
    else:
        history_block = ""

    bl = f"[قائمة الكتب المتاحة]\n{book_list}\n\n" if book_list else ""
    q_label = "Question:" if is_english else "السؤال:"
    return (
        f"{directive}\n\n{bl}{ctx_block}"
        f"{facts_block}{history_block}\n\n{q_label} {query}"
    )


# ── Shared request validation (/chat + /chat/stream) ─────────────────────────
def _preflight(user_id: str, session_id: str, user_input: str,
               check_rate_fn, is_dup_fn):
    """Run the four request checks shared by both chat endpoints.

    Returns (http_status, message, extra_headers) on rejection, or None when
    the request is valid. The caller decides the response shape (HTTP error
    for /chat, SSE error event for /chat/stream).
    """
    if not check_rate_fn(user_id):
        return (429, _MSG.RATE_CHAT(int(os.getenv("CHAT_MAX_PER_MINUTE", "30"))),
                {"Retry-After": "60"})
    if is_dup_fn(session_id, user_input):
        return 429, _MSG.DUPLICATE_MSG, None
    if len(user_input) > _MAX_INPUT_CHARS:
        return 400, _MSG.TOO_LONG(_MAX_INPUT_CHARS), None
    if _check_injection(user_input):
        logger.warning(f"[SECURITY] Injection attempt blocked user={user_id[:8]}")
        return 400, "رسالة تحتوي على محتوى غير مسموح", None
    return None


def _sse_oneshot(payload: dict) -> StreamingResponse:
    """Single SSE event followed by [DONE] — used for early-exit responses."""
    async def _gen():
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream")


# ── Request model ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_input: str = Field(
        ..., validation_alias=AliasChoices("prompt", "message", "user_input")
    )
    user_id:    str           = "unknown"
    session_id: Optional[str] = None


# ── Shared context preparation ────────────────────────────────────────────────
async def _prepare_request_context(req: ChatRequest, session_id: str, loop):
    from backend.core.context import resolve_context, is_library_query, get_book_list
    from backend.core.session import (
        get_history, get_recent_user_text, get_session_book, set_session_book,
    )

    # Fast RAM-only reads first — anchor follow-up retrieval on the topic + book.
    prev_user_text = await get_recent_user_text(session_id)
    pref_book      = get_session_book(session_id)

    (context, ctx_source), history, user_facts_pre = await asyncio.gather(
        loop.run_in_executor(
            None, resolve_context, req.user_input, req.user_id, prev_user_text, pref_book),
        get_history(session_id),
        loop.run_in_executor(None, _get_user_facts, req.user_input, req.user_id),
    )
    user_facts = "" if ctx_source == "small_talk" else user_facts_pre

    # Remember the book this turn drew from → session affinity for the next turn.
    if ctx_source == "books":
        set_session_book(session_id, _extract_source(context))

    lang_ok = True
    if ctx_source != "small_talk":
        lang_ok = validate_language(req.user_input)

    book_list    = get_book_list(req.user_id) if is_library_query(req.user_input) else ""
    user_content = _build_user_content(
        req.user_input, context, ctx_source, history, book_list, user_facts
    )
    return context, ctx_source, user_content, lang_ok


# ── Router factory ────────────────────────────────────────────────────────────
def make_chat_router(
    get_current_user,
    system_prompt: str,
    check_rate_fn,
    is_dup_fn,
    circuit_open_fn,
    circuit_fail_fn,
    circuit_ok_fn,
):
    from backend.core.context import self_verify_response
    from backend.core.session import save_exchange
    import backend.llm.offline_llm as llm
    from backend.api.functions import (
        add_tools_to_prompt, extract_tool_call, execute_tool, strip_tool_call
    )
    from backend.core.state import ENABLE_FUNCTION_CALLING, BOOKS_DIR
    from backend.database.vector_db import scroll_all

    router = APIRouter()

    async def _resolve_session(req_session_id, user_id: str) -> str:
        """
        Return a session_id that belongs to user_id.
        If a session_id is provided but owned by a different user, generate a new one
        (prevents cross-user session injection without blocking legitimate reuse).
        """
        if not req_session_id:
            return str(uuid.uuid4())
        try:
            from backend.core.session import get_session_owner as _owner
            owner = await _owner(req_session_id)
            if owner is not None and owner != user_id:
                logger.warning(
                    f"[SECURITY] session_id {req_session_id[:8]}… belongs to "
                    f"different user — assigning new session to {user_id[:8]}…"
                )
                return str(uuid.uuid4())
        except Exception as e:
            # Fail closed: if ownership can't be verified, never reuse the
            # requested id — issue a fresh session instead.
            logger.error(f"[SECURITY] session ownership check failed ({e}) — "
                         f"issuing new session for {user_id[:8]}…")
            return str(uuid.uuid4())
        return req_session_id

    @router.post("/chat")
    async def chat_endpoint(req: ChatRequest,
                            current=Depends(get_current_user)):
        try:
            req.user_id = current.user_id
            session_id  = await _resolve_session(req.session_id, current.user_id)

            rejected = _preflight(req.user_id, session_id, req.user_input,
                                  check_rate_fn, is_dup_fn)
            if rejected:
                status, msg, hdrs = rejected
                raise HTTPException(status_code=status, detail=msg, headers=hdrs)

            _tc0 = _time_mod.perf_counter()
            loop = asyncio.get_running_loop()

            context, ctx_source, user_content, lang_ok = await _prepare_request_context(
                req, session_id, loop
            )
            _tc_rag = _time_mod.perf_counter()

            if not lang_ok:
                return {"response": APOLOGY_MSG, "source": "", "session_id": session_id,
                        "from_books": False, "from_web": False}

            if circuit_open_fn():
                raise HTTPException(status_code=503, detail=_MSG.CIRCUIT_OPEN)

            _is_en      = _is_english_query(req.user_input)
            _base_sys   = ENGLISH_SYSTEM_PROMPT if _is_en else system_prompt
            _sys_prompt = add_tools_to_prompt(_base_sys, enable=ENABLE_FUNCTION_CALLING)
            _tc1 = _tc_rag
            for _attempt in range(2):
                try:
                    response_text = await loop.run_in_executor(
                        None, lambda: llm.chat(_sys_prompt, user_content, num_predict=_CHAT_MAX_TOKENS)
                    )
                    circuit_ok_fn()
                    break
                except Exception as _llm_err:
                    circuit_fail_fn()
                    if _attempt == 0:
                        logger.warning(f"[LLM] attempt 1 failed, retrying in 500ms: {_llm_err}")
                        await asyncio.sleep(0.5)
                    else:
                        raise

            if ENABLE_FUNCTION_CALLING:
                _tool_call = extract_tool_call(response_text)
                if _tool_call:
                    _tool_result = execute_tool(
                        _tool_call, books_dir=BOOKS_DIR, scroll_fn=scroll_all)
                    _follow_up = f"{strip_tool_call(response_text)}\n\nنتيجة الأداة: {_tool_result}"
                    try:
                        response_text = await loop.run_in_executor(
                            None, lambda r=_follow_up: llm.chat(_base_sys, r))
                        circuit_ok_fn()
                    except Exception:
                        response_text = _follow_up

            logger.info(
                f"[TIMING] /chat  rag={int((_tc1-_tc0)*1000)}ms "
                f"llm={int((_time_mod.perf_counter()-_tc1)*1000)}ms "
                f"total={int((_time_mod.perf_counter()-_tc0)*1000)}ms  src={ctx_source}"
            )
            # Answers are produced directly in MSA (Fusha) / English — no dialect post-processing
            response_text = _sanitize_response(response_text)

            # Generation instability guard: single chars or empty output get a safe fallback
            if len(response_text.strip()) < 5:
                response_text = "عذراً، لم أفهم سؤالك جيداً. هل يمكنك إعادة صياغته؟"

            if ctx_source == "books":
                _rt, _ctx, _qi = response_text, context, req.user_input
                response_text = await loop.run_in_executor(
                    None, lambda: self_verify_response(_rt, _ctx, _qi)
                )

            await save_exchange(session_id, req.user_input, response_text, req.user_id)

            return {
                "response":   response_text,
                "source":     _extract_source(context) if ctx_source == "books" else "",
                "session_id": session_id,
                "from_books": ctx_source == "books",
                "from_web":   ctx_source == "web",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=_MSG.LLM_ERROR(e))

    @router.post("/chat/stream")
    async def chat_stream_endpoint(req: ChatRequest,
                                   current=Depends(get_current_user)):
        req.user_id = current.user_id
        session_id  = await _resolve_session(req.session_id, current.user_id)

        rejected = _preflight(req.user_id, session_id, req.user_input,
                              check_rate_fn, is_dup_fn)
        if rejected:
            return _sse_oneshot({"error": rejected[1]})

        _t0  = _time_mod.perf_counter()
        loop = asyncio.get_running_loop()

        context, ctx_source, user_content, lang_ok = await _prepare_request_context(
            req, session_id, loop
        )
        _t_rag = _time_mod.perf_counter()
        logger.info(f"[TIMING] /stream rag={int((_t_rag-_t0)*1000)}ms  src={ctx_source}")

        if not lang_ok:
            return _sse_oneshot({"token": APOLOGY_MSG})

        if circuit_open_fn():
            return _sse_oneshot({"error": _MSG.CIRCUIT_OPEN})

        # ── Canned response lookup (bypasses LLM entirely) ────────────────────
        if ctx_source == "small_talk":
            _canned = _lookup_canned(req.user_input)
            if _canned:
                async def _canned_stream(_r=_canned, _sid=session_id, _uid=req.user_id, _q=req.user_input):
                    yield f"data: {json.dumps({'token': _r}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    await save_exchange(_sid, _q, _r, _uid)
                logger.info(f"[CANNED] hit for: {req.user_input[:40]}")
                return StreamingResponse(_canned_stream(), media_type="text/event-stream")

        logger.info(
            f"[TIMING] /stream pre-llm={int((_time_mod.perf_counter()-_t0)*1000)}ms "
            f"prompt_len={len(user_content)}"
        )
        # English queries use the English system prompt and skip dialectization
        _is_en   = _is_english_query(req.user_input)
        _base_sys = ENGLISH_SYSTEM_PROMPT if _is_en else system_prompt
        full_response: list[str] = []
        _MAX_STREAM_BYTES = 100_000   # 100 KB cap — prevents RAM exhaustion on runaway output

        async def event_generator():
            try:
                inner_loop = asyncio.get_running_loop()

                def _start_produce() -> queue.Queue:
                    q: queue.Queue = queue.Queue()
                    def _produce():
                        try:
                            for tok in llm.chat_stream(_base_sys, user_content, num_predict=_CHAT_MAX_TOKENS):
                                q.put(tok)
                        except Exception as exc:
                            q.put(exc)
                        finally:
                            q.put(None)
                    threading.Thread(target=_produce, daemon=True).start()
                    return q

                token_queue = _start_produce()
                tok_buf: list[str] = []
                _attempt   = 0
                _tok_count = 0
                _loop_window: list[str] = []   # sliding window for loop detection
                _LOOP_WIN   = 80               # chars to keep in window
                _LOOP_REPS  = 3               # how many times a phrase must repeat

                def _detect_loop(buf: list[str]) -> bool:
                    """True if the last ~80 chars contain a repeated 10-char phrase."""
                    text = "".join(buf)
                    if len(text) < _LOOP_WIN:
                        return False
                    tail = text[-_LOOP_WIN:]
                    # pick a phrase from the middle of the tail and count its occurrences
                    phrase = tail[20:30]
                    if len(phrase) < 5:
                        return False
                    return text.count(phrase) >= _LOOP_REPS

                while True:
                    try:
                        item = await inner_loop.run_in_executor(
                            None, lambda: token_queue.get(timeout=120)
                        )
                    except queue.Empty:
                        if tok_buf:
                            chunk = _sanitize_response("".join(tok_buf))
                            full_response.append(chunk)
                            yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'error': _MSG.STREAM_TIMEOUT})}\n\n"
                        break

                    if item is None:
                        if tok_buf:
                            chunk = _sanitize_response("".join(tok_buf))
                            full_response.append(chunk)
                            yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
                        circuit_ok_fn()
                        break

                    if isinstance(item, Exception):
                        circuit_fail_fn()
                        if _attempt == 0:
                            _attempt += 1
                            logger.warning(f"[LLM/stream] attempt 1 failed, retrying: {item}")
                            await asyncio.sleep(0.5)
                            token_queue = _start_produce()
                            tok_buf = []
                            _tok_count = 0
                            _loop_window = []
                            continue
                        yield f"data: {json.dumps({'error': str(item)})}\n\n"
                        return

                    tok_buf.append(item)
                    _tok_count += 1
                    _loop_window.append(item)
                    if len("".join(_loop_window)) > _LOOP_WIN * 2:
                        _loop_window = _loop_window[-40:]

                    if _detect_loop(full_response + tok_buf):
                        logger.warning("[LLM/stream] loop detected — stopping early")
                        if tok_buf:
                            chunk = _sanitize_response("".join(tok_buf))
                            full_response.append(chunk)
                            yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
                        circuit_ok_fn()
                        break

                    # Hard cap: stop streaming if response is unreasonably large
                    if sum(len(c) for c in full_response) + len("".join(tok_buf)) > _MAX_STREAM_BYTES:
                        logger.warning("[LLM/stream] response size cap reached — stopping early")
                        if tok_buf:
                            chunk = _sanitize_response("".join(tok_buf))
                            full_response.append(chunk)
                            yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
                        circuit_ok_fn()
                        break

                    if _tok_count >= _STREAM_FLUSH_EVERY:
                        chunk = _sanitize_response("".join(tok_buf))
                        full_response.append(chunk)
                        yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
                        tok_buf = []
                        _tok_count = 0

                # Send the full sanitized response as a replace event for a clean
                # final markdown render. Answers are already in MSA (Fusha) / English.
                full_text = "".join(full_response)
                full_text = _sanitize_response(full_text)
                if len(full_text.strip()) < 5:
                    full_text = "عذراً، لم أفهم سؤالك جيداً. هل يمكنك إعادة صياغته؟"
                replace_evt = json.dumps({"replace": full_text}, ensure_ascii=False)
                yield f"data: {replace_evt}\n\n"

                meta = json.dumps({
                    "meta": {
                        "from_books": ctx_source == "books",
                        "from_web":   ctx_source == "web",
                        "source":     _extract_source(context) if ctx_source == "books" else "",
                    }
                }, ensure_ascii=False)
                yield f"data: {meta}\n\n"
                yield "data: [DONE]\n\n"

                await save_exchange(
                    session_id, req.user_input, full_text, req.user_id
                )
            except Exception as e:
                logger.error(f"[stream] event_generator error: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Known image magic bytes: JPEG FF D8, PNG 89 50 4E 47, GIF 47 49 46, WEBP 52 49 46 46
    _IMG_MAGIC = [b'\xff\xd8', b'\x89PNG', b'GIF8', b'RIFF']

    @router.post("/chat/image")
    async def chat_image_endpoint(
        file:       UploadFile     = File(...),
        user_input: str            = Form(""),
        user_id:    str            = Form(""),
        session_id: Optional[str] = Form(None),
        current=Depends(get_current_user),
    ):
        import base64
        user_id    = current.user_id
        session_id = session_id or str(uuid.uuid4())

        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail=_MSG.NOT_IMAGE)

        # Read header first for magic-bytes check, then read rest
        header = await file.read(12)
        if not any(header.startswith(sig) for sig in _IMG_MAGIC):
            raise HTTPException(status_code=400, detail=_MSG.NOT_IMAGE)
        # Read remaining bytes (header already consumed)
        rest     = await file.read()
        img_data = header + rest
        if len(img_data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=_MSG.IMAGE_TOO_LARGE)

        img_b64 = base64.b64encode(img_data).decode()
        prompt  = user_input.strip() or "ماذا يوجد في هذه الصورة؟ صف ما تراه بالتفصيل."
        loop    = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: llm.chat_vision(prompt, img_b64)
            )
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            raise HTTPException(status_code=500, detail=_MSG.IMG_FAIL(e))

        await save_exchange(session_id, f"[صورة] {prompt}", response, user_id)
        return {"response": response, "session_id": session_id}

    return router
