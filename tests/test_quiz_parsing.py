"""Quiz LLM-output parsing — the production failure point for quiz generation."""
import pydantic
import pytest

from backend.api.quiz import (
    QuizRequest, _extract_json_array, _is_valid_question,
)

_VALID_Q = {
    "question": "ما هو دايود زينر؟",
    "options": ["صمام ينظم الجهد", "مقاومة متغيرة", "مكثف كيميائي", "محول كهربائي"],
    "answer": 0,
}


def test_extract_plain_array():
    raw = '[{"question": "س؟", "options": ["1111", "2222", "3333", "4444"]}]'
    items = _extract_json_array(raw)
    assert len(items) == 1 and items[0]["question"] == "س؟"


def test_extract_with_markdown_fence():
    raw = '```json\n[{"question": "س؟", "options": ["ا", "ب", "ج", "د"]}]\n```'
    assert len(_extract_json_array(raw)) == 1


def test_extract_from_object_wrapper():
    raw = '{"questions": [{"question": "س؟", "options": ["ا", "ب", "ج", "د"]}]}'
    assert len(_extract_json_array(raw)) == 1


def test_extract_with_surrounding_prose():
    raw = 'إليك الأسئلة:\n[{"question": "س؟"}]\nانتهى.'
    assert len(_extract_json_array(raw)) == 1


def test_extract_truncated_json_returns_empty():
    # The exact production failure: output cut mid-array by the token cap
    raw = '[\n  {"question": "ما هو دايود", "options": ["صمام ين'
    assert _extract_json_array(raw) == []


def test_extract_garbage_returns_empty():
    assert _extract_json_array("لا يوجد JSON هنا إطلاقاً") == []
    assert _extract_json_array("") == []


def test_valid_question_accepted():
    assert _is_valid_question(_VALID_Q)


def test_question_with_few_options_rejected():
    q = dict(_VALID_Q, options=_VALID_Q["options"][:3])
    assert not _is_valid_question(q)


def test_question_with_duplicate_options_rejected():
    q = dict(_VALID_Q, options=["نفس الخيار"] * 4)
    assert not _is_valid_question(q)


def test_question_with_chinese_rejected():
    q = dict(_VALID_Q, question="ما هو 电压؟")
    assert not _is_valid_question(q)


def test_quiz_request_normalizes_difficulty():
    assert QuizRequest(book_title="كتاب", difficulty="HARD").difficulty == "hard"
    assert QuizRequest(book_title="كتاب", difficulty="غريب").difficulty == "medium"


def test_quiz_request_rejects_injection_in_title():
    with pytest.raises(pydantic.ValidationError):
        QuizRequest(book_title="<script>alert(1)</script>")


# ── Meta-question filter (quiz quality) ───────────────────────────────────────
from backend.api.quiz import _is_meta_question


def test_meta_questions_detected():
    for q in ["ما هو الموضوع الرئيسي للكتاب؟", "ما الهدف من هذا الكتاب؟",
              "What is the main topic of the book?", "ما مضمون النص؟"]:
        assert _is_meta_question(q), f"should flag meta: {q}"


def test_content_questions_not_meta():
    for q in ["ما هو المتغير في البرمجة؟", "ماذا يفعل أمر الطباعة؟",
              "أي خوارزمية تستخدم للبحث بالعرض؟", "What is a variable?"]:
        assert not _is_meta_question(q), f"should NOT flag content: {q}"
