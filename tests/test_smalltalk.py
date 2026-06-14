"""Small-talk detection — multi-part greetings must route to chat, not book search.

Regression for the live bug where "السلام عليكم، شلونك اليوم؟" hit book retrieval
because the anchored regex only matched single-phrase messages.
"""
from backend.core.smalltalk import is_small_talk, is_library_query

_SMALL_TALK = [
    "السلام عليكم، شلونك اليوم؟",
    "لا شكرا، خلص",
    "هلا شلونك",
    "صباح الخير",
    "تمام شكرا",
    "اهلا صاحبي شلونك",
    "مرحبا، كيف حالك اليوم",
    "شكو ماكو",
    "مع السلامة",
    "hello",
    "thanks bye",
]

_REAL_QUESTIONS = [
    "عرّف الذكاء الاصطناعي",
    "شنو هي المتغيرات بالبرمجة",
    "مرحبا اشرح لي الذكاء الاصطناعي",      # greeting + real question
    "اهلا شنو هي البرمجة",                  # greeting + real question
    "What is Freakonomics about",
    "شلونك تگدر تشرح الكسور الجزئية؟",
]


def test_multipart_greetings_are_small_talk():
    for msg in _SMALL_TALK:
        assert is_small_talk(msg), f"should be small talk: {msg}"


def test_real_questions_are_not_small_talk():
    for msg in _REAL_QUESTIONS:
        assert not is_small_talk(msg), f"should NOT be small talk: {msg}"


def test_empty_and_trivial():
    assert is_small_talk("")
    assert is_small_talk("هل")        # <=3 chars, no question mark


def test_library_query_detection():
    assert is_library_query("شنو الكتب المتاحة عندك؟")
    assert is_library_query("what books are available")
    assert not is_library_query("عرّف الذكاء الاصطناعي")
