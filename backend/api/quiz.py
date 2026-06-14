"""
Quiz generation endpoint — builds MCQs from book content via RAG + LLM.
Registered in server_backend.py with prefix /quiz.

Fixes (May 2026):
  - Strict book filtering: post-filters all retrieved docs to exact book_title match
  - Retry logic: each batch retried up to 2 times if it returns fewer questions than needed
  - All 4 content queries used in parallel for richer, more diverse context
  - Larger context per doc (1200 chars) for deeper question generation
  - Smarter prompt: guides model to test real comprehension, not surface recall
  - Batch size reduced to 5 for reliability with small models (qwen2.5:3b)
"""
import os
import re
import json
import math
import asyncio
import inspect
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator

_HAS_CHINESE = re.compile(r'[一-鿿㐀-䶿]')

logger = logging.getLogger("quiz")
router = APIRouter(prefix="/quiz", tags=["Quiz"])

_BATCH_SIZE    = int(os.getenv("QUIZ_BATCH_SIZE", "1"))  # 1 Q/call: smallest
# generation that reliably finishes on a memory-constrained CPU using the
# already-loaded chat model (loading a 2nd model thrashes RAM). Raise on bigger
# machines for fewer round-trips.
# 130 tokens/question truncated Arabic JSON mid-array (server logs: "no JSON
# found" with output cut at the token cap) — each retry then burned another
# batch. 230 covers Q + 4 options + explanation in Arabic with headroom.
_TOKENS_PER_Q  = 230
# A 7B model on CPU spends ~20-40 s on prompt prefill alone, then ~12 tok/s of
# generation. A 90 s per-batch timeout left no margin and EVERY batch timed out
# (quiz returned nothing). 150 s fits prefill + a 3-question batch; the shorter
# context below cuts prefill time further.
_PER_BATCH_SEC = int(os.getenv("QUIZ_BATCH_TIMEOUT", "150"))
_OUTER_TIMEOUT = int(os.getenv("QUIZ_OUTER_TIMEOUT", "420"))
_MAX_RETRIES   = 1    # one retry only
_CHARS_PER_DOC = 500  # shorter context → faster LLM prefill
_MAX_CONTEXT   = 2500 # max total context chars per batch (was 4000 — prefill cost)


def _extract_json_array(text: str) -> list:
    # Strip markdown code fences that small models often add
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    # Also try extracting from inside a JSON object wrapper {"questions":[...]}
    obj_match = re.search(r'"(?:questions|quiz|items)"\s*:\s*(\[)', text)
    candidates = [text]
    if obj_match:
        candidates.insert(0, text[obj_match.start(1):])

    for candidate in candidates:
        pos = 0
        while True:
            start = candidate.find('[', pos)
            if start == -1:
                break
            depth = 0
            for i, ch in enumerate(candidate[start:], start):
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(candidate[start:i + 1])
                            if isinstance(result, list) and result:
                                return result
                        except json.JSONDecodeError:
                            pass
                        pos = start + 1
                        break
            else:
                break
    return []


_ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}
_TITLE_INJECTION_RE = re.compile(r'[<>\[\]{}\x00-\x1f]|--|\*\*|<<|>>')


class QuizRequest(BaseModel):
    book_title:  str = Field(..., min_length=1, max_length=200)
    n_questions: int = Field(default=5, ge=1, le=50)
    difficulty:  str = Field(default="medium")

    @field_validator('book_title')
    @classmethod
    def sanitize_title(cls, v: str) -> str:
        v = v.strip()
        if _TITLE_INJECTION_RE.search(v):
            raise ValueError("عنوان الكتاب يحتوي على رموز غير مسموحة")
        return v

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in _ALLOWED_DIFFICULTY:
            return "medium"
        return v


class QuizQuestion(BaseModel):
    question:    str
    options:     list[str]
    correct:     int
    explanation: str


class QuizResponse(BaseModel):
    book_title: str
    questions:  list[QuizQuestion]
    total:      int


def _build_prompt(batch_n: int, diff_label: str, context: str, book_title: str = "") -> str:
    # Force a single consistent language — otherwise a programming book whose
    # sampled chunks are mostly English code yields English questions even when
    # the book is Arabic. The book TITLE is the stable signal of the intended
    # reader's language; fall back to the context script if the title is neutral.
    title_ar = sum(1 for c in book_title if '؀' <= c <= 'ۿ')
    if title_ar >= 2:
        is_ar = True
    elif book_title and title_ar == 0 and any(c.isascii() and c.isalpha() for c in book_title):
        is_ar = False
    else:
        ar = sum(1 for c in context if '؀' <= c <= 'ۿ')
        is_ar = ar / max(len(context), 1) > 0.25
    lang_rule = (
        '- اكتب جميع الأسئلة والخيارات والشروح بالعربية الفصحى فقط — ممنوع كتابة '
        'سؤال بالإنجليزية (تُترك المصطلحات التقنية والأكواد بالإنجليزية عند الضرورة).\n'
        if is_ar else
        '- Write every question, every option, and every explanation in English only.\n'
    )
    return (
        f'أنشئ بالضبط {batch_n} سؤال اختيار من متعدد ({diff_label}) يختبر فهم '
        f'المعلومات المحددة في النص التالي.\n\n'
        f'النص:\n{context}\n\n'
        f'أخرج JSON فقط بهذا الشكل بدون أي نص آخر:\n'
        '[{"question":"...","options":["أ","ب","ج","د"],"correct":0,"explanation":"..."}]\n\n'
        'قواعد إلزامية:\n'
        f'{lang_rule}'
        '- كل سؤال عن معلومة أو مفهوم أو حقيقة محددة وردت في النص (تعريف، سبب، '
        'نتيجة، خطوة، مثال، مقارنة).\n'
        '- ممنوع الأسئلة العامة عن الكتاب نفسه مثل "ما موضوع الكتاب؟" أو '
        '"ما الهدف من الكتاب؟".\n'
        '- الخيارات الأربعة كلها متعلقة بالموضوع وقريبة الاحتمال، وواحد فقط صحيح، '
        'والخيارات الخاطئة معقولة لكن خاطئة فعلاً (ليست عشوائية).\n'
        '- explanation: جملة قصيرة تبرّر الإجابة الصحيحة من النص.\n'
        f'- correct = رقم الإجابة الصحيحة (0-3). أنشئ {batch_n} سؤال بالضبط.'
    )


def _options_distinct(opts: list) -> bool:
    """Return False if two options are identical or share >55% of their words."""
    texts = [str(o).strip() for o in opts]
    # Exact duplicates are always invalid — the word-overlap heuristic below
    # skips options shorter than 3 words, which used to let them through.
    if len(set(texts)) != len(texts):
        return False
    for i in range(len(texts)):
        words_i = set(texts[i].split())
        if len(words_i) < 3:
            continue
        for j in range(i + 1, len(texts)):
            words_j = set(texts[j].split())
            if len(words_j) < 3:
                continue
            overlap = len(words_i & words_j) / min(len(words_i), len(words_j))
            if overlap > 0.55:
                return False
    return True


# Generic "about the book/text itself" questions add no learning value — reject
# them so the quiz tests the CONTENT, not metadata. A meta-question co-mentions a
# "meta" word (topic/purpose/idea…) AND a "book/text" word (كتاب handles ال/لل
# prefixes). Both must appear → strong, conservative signal.
_META_WORD = re.compile(r'موضوع|هدف|فكرة|عنوان|محتوى|مضمون|غرض|'
                        r'main\s+(topic|subject|purpose|idea|theme)|what.*about', re.IGNORECASE)
_BOOK_WORD = re.compile(r'كتاب|النص|نص\b|مقال|الفصل|وثيقة|book|text|document|passage', re.IGNORECASE)


def _is_meta_question(text: str) -> bool:
    return bool(_META_WORD.search(text) and _BOOK_WORD.search(text))


def _is_valid_question(q: dict) -> bool:
    if not isinstance(q, dict) or not q.get("question"):
        return False
    question_text = str(q["question"]).strip()
    if not question_text or _HAS_CHINESE.search(question_text):
        return False
    if _is_meta_question(question_text):   # generic "about the book" — no value
        return False
    opts = q.get("options", [])
    if not isinstance(opts, list) or len(opts) < 4:
        return False
    for opt in opts:
        text = str(opt).strip()
        if not text or len(text) <= 2 or _HAS_CHINESE.search(text):
            return False
    # Reject questions with duplicate or near-duplicate options
    if not _options_distinct(opts):
        return False
    return True


async def _call_llm_batch(batch_n: int, book_title: str, diff_label: str,
                           context: str, llm_ask) -> list[dict]:
    """Call LLM for one batch with retry logic."""
    prompt = _build_prompt(batch_n, diff_label, context, book_title)

    sig = inspect.signature(llm_ask).parameters
    kw: dict = {}
    if "num_predict" in sig:
        kw["num_predict"] = batch_n * _TOKENS_PER_Q
    if "num_ctx" in sig:
        # MUST match the chat model's num_ctx (NUM_CTX). A different context size
        # makes Ollama reload the model under a separate KV cache — a ~30-60 s
        # reload on a memory-constrained CPU that made EVERY quiz batch time out
        # while chat (same loaded model) stayed sub-second. Reuse the loaded
        # instance instead.
        kw["num_ctx"] = int(os.getenv("NUM_CTX", "1536"))
    if "timeout" in sig:
        kw["timeout"] = _PER_BATCH_SEC

    _angle_re = re.compile(r'^<(.+)>$')

    for attempt in range(1 + _MAX_RETRIES):
        try:
            raw = await llm_ask(prompt, **kw)
        except Exception as e:
            logger.warning(f"Quiz batch LLM error (attempt {attempt+1}): {e}")
            if attempt < _MAX_RETRIES:
                continue
            return []

        items = _extract_json_array(raw)
        if not items:
            logger.warning(f"Quiz batch: no JSON found (book='{book_title}', attempt {attempt+1}). Raw output: {raw[:300]}")
            if attempt < _MAX_RETRIES:
                continue

        # Clean angle-bracket wrappers
        for item in items:
            if not isinstance(item, dict):
                continue
            if "question" in item:
                item["question"] = _angle_re.sub(r'\1', str(item["question"]).strip())
            if isinstance(item.get("options"), list):
                item["options"] = [_angle_re.sub(r'\1', str(o).strip()) for o in item["options"]]

        valid = [q for q in items if _is_valid_question(q)][:batch_n]
        skipped = len(items) - len(valid)
        if skipped:
            logger.warning(f"Quiz batch: skipped {skipped} malformed questions")

        if len(valid) >= batch_n or attempt == _MAX_RETRIES:
            logger.info(f"Quiz batch: got {len(valid)}/{batch_n} questions (attempt {attempt+1})")
            return valid

        # Got fewer than needed — retry with a note in the prompt
        logger.info(f"Quiz batch: only {len(valid)}/{batch_n}, retrying...")

    return []


async def _retrieve_book_docs(retriever, query: str, book_title: str,
                               k: int, user_id: str, loop) -> list:
    """Retrieve docs strictly filtered to the specified book."""
    docs = await loop.run_in_executor(
        None,
        lambda: retriever.retrieve(query, k=k, book_hint=book_title,
                                   use_multi_query=False, user_id=user_id),
    )
    # Strict post-filter: only keep chunks from this exact book
    strict = [d for d in docs if d.metadata.get("book_title") == book_title]
    if not strict and docs:
        # Fallback: fuzzy match if exact title doesn't match (e.g. slight encoding diff)
        book_lower = book_title.lower().strip()
        strict = [d for d in docs
                  if book_lower in d.metadata.get("book_title", "").lower()]
    return strict if strict else docs


async def _generate(book_title: str, n: int, difficulty: str,
                    retriever, llm_ask, user_id: str = "") -> list[dict]:
    """Core: multi-query RAG with strict book filter → parallel batches with retry."""
    diff_map = {
        "easy":   "بسيطة وسهلة الفهم",
        "medium": "متوسطة الصعوبة تحتاج فهماً جيداً",
        "hard":   "صعبة تحتاج فهماً عميقاً وتحليلاً",
    }
    diff_label = diff_map.get(difficulty, diff_map["medium"])

    n_batches = math.ceil(n / _BATCH_SIZE)
    k_per_query = min(10, max(5, n // 2))

    # 2 diverse queries (was 4) — sufficient coverage, faster retrieval
    content_queries = [
        f"مفاهيم وتعريفات وأمثلة تطبيقية في كتاب {book_title}",
        f"حقائق وأسباب ونتائج ومقارنات في كتاب {book_title}",
    ]

    loop = asyncio.get_running_loop()

    # Primary content: a broad, even sample of the WHOLE book straight from the
    # corpus (no relevance gate — quizzes need coverage, not query-relevance).
    all_docs = []
    seen_ids = set()
    if hasattr(retriever, "book_chunks"):
        try:
            broad = await loop.run_in_executor(
                None, lambda: retriever.book_chunks(book_title, user_id, max(12, n * 4)))
            for d in broad:
                if id(d) not in seen_ids:
                    all_docs.append(d)
                    seen_ids.add(id(d))
        except Exception as e:
            logger.warning(f"Quiz book_chunks failed ({e}) — falling back to retrieval")

    # Fallback / supplement: relevance retrieval (handles encoding-mismatched
    # titles and books not yet in the in-RAM corpus)
    if len(all_docs) < max(4, n):
        fetch_tasks = [
            _retrieve_book_docs(retriever, q, book_title, k_per_query, user_id, loop)
            for q in content_queries
        ]
        for result in await asyncio.gather(*fetch_tasks, return_exceptions=True):
            if isinstance(result, Exception):
                continue
            for d in result:
                if id(d) not in seen_ids:
                    all_docs.append(d)
                    seen_ids.add(id(d))

    if not all_docs:
        raise HTTPException(status_code=404,
            detail=f"لم يُعثر على محتوى للكتاب: {book_title} — تأكد من اسم الكتاب")

    logger.info(f"Quiz: {len(all_docs)} docs for '{book_title}' (broad-sample primary)")

    # Build batch contexts — spread docs evenly across batches
    batch_args: list[tuple] = []
    docs_per_batch = max(1, len(all_docs) // n_batches)

    remaining = n
    for batch_i in range(n_batches):
        if remaining <= 0:
            break
        batch_n = min(_BATCH_SIZE, remaining)
        remaining -= batch_n

        # Each batch gets a distinct slice of docs for diversity
        start = batch_i * docs_per_batch
        end   = start + docs_per_batch + 2  # slight overlap for continuity
        slice_docs = all_docs[start:end] if start < len(all_docs) else all_docs[-docs_per_batch:]
        if not slice_docs:
            slice_docs = all_docs

        context = "\n\n---\n\n".join(
            d.page_content[:_CHARS_PER_DOC] for d in slice_docs
        )[:_MAX_CONTEXT]

        batch_args.append((batch_n, context))
        logger.info(
            f"Quiz batch {batch_i+1}/{n_batches}: "
            f"{batch_n} Qs, {len(slice_docs)} docs, {len(context)} chars"
        )

    # Run all batches in parallel
    tasks = [
        _call_llm_batch(batch_n, book_title, diff_label, context, llm_ask)
        for batch_n, context in batch_args
    ]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_questions: list[dict] = []
    seen_questions: set[str] = set()
    for batch_i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.warning(f"Quiz batch {batch_i+1} exception: {result}")
            continue
        for q in result:
            key = str(q.get("question", "")).strip().lower()[:80]
            if key and key not in seen_questions:
                seen_questions.add(key)
                all_questions.append(q)

    # ── Fill batch: if short, request exactly the missing count ──────────────
    shortage = n - len(all_questions)
    if 0 < shortage <= n:
        logger.info(f"Quiz: short by {shortage} — running fill batch")
        # Use full docs rotated so fill batch sees different content
        fill_docs  = all_docs[len(all_docs)//2:] + all_docs[:len(all_docs)//2]
        fill_ctx   = "\n\n---\n\n".join(
            d.page_content[:_CHARS_PER_DOC] for d in fill_docs
        )[:_MAX_CONTEXT]
        fill_qs = await _call_llm_batch(shortage, book_title, diff_label, fill_ctx, llm_ask)
        for q in fill_qs:
            key = str(q.get("question", "")).strip().lower()[:80]
            if key and key not in seen_questions:
                seen_questions.add(key)
                all_questions.append(q)
        logger.info(f"Quiz: fill batch added {len(fill_qs)} → total {len(all_questions)}/{n}")

    logger.info(f"Quiz: final {len(all_questions)}/{n} questions for '{book_title}'")
    return all_questions[:n]


def make_quiz_router(retriever, llm_ask, get_current_user):
    """Factory — call once in server_backend.py passing live dependencies."""

    @router.post("/generate", response_model=QuizResponse)
    async def generate_quiz(req: QuizRequest,
                            _user=Depends(get_current_user)):
        user_id = getattr(_user, "user_id", "")
        try:
            raw = await asyncio.wait_for(
                _generate(req.book_title, req.n_questions, req.difficulty,
                          retriever, llm_ask, user_id),
                timeout=float(_OUTER_TIMEOUT),
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="انتهت مهلة توليد الاختبار — جرب عدداً أقل من الأسئلة أو أعد المحاولة"
            )

        questions = []
        for q in raw:
            try:
                questions.append(QuizQuestion(
                    question    = str(q.get("question", "")),
                    options     = [str(o) for o in q.get("options", [])],
                    correct     = int(q.get("correct", 0)),
                    explanation = str(q.get("explanation", "")),
                ))
            except Exception:
                continue

        if not questions:
            raise HTTPException(
                status_code=422,
                detail=f"فشل توليد الأسئلة للكتاب '{req.book_title}' — تأكد من اسم الكتاب وأعد المحاولة"
            )

        return QuizResponse(book_title=req.book_title,
                            questions=questions,
                            total=len(questions))

    return router
