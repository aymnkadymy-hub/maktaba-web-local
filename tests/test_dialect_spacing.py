"""Dialect spacing must never split a valid Arabic word.

Regression for the live bug where _fix_arabic_spacing inserted spaces inside
common words (الاقتصاد→"الا قتصاد", المعلومات→"ال معلومات", الفهم→"الف هم"),
corrupting ~80% of answers.
"""
import pytest

pytest.importorskip("langdetect")  # dialect_processor pulls langdetect indirectly

from backend.dialect.dialect_processor import _fix_arabic_spacing

# Common words that previously got corrupted by particle-splitting rules
_VALID = [
    "الاصطناعي", "الاقتصاد", "الأساليب", "الفهم", "المعلومات", "المعرفة",
    "الانترنت", "الاستخدام", "الاعتماد", "المفاهيم", "الفيزياء", "الاختبار",
    "المعادلة", "الانتقال", "الخوارزمية", "الذكاء", "التعلم", "البيانات",
    "هواية", "أنابيب", "لائحة", "لاعبين", "رحلة", "منزل", "علامة",
]


@pytest.mark.parametrize("word", _VALID)
def test_valid_word_not_split(word):
    assert _fix_arabic_spacing(word) == word, f"corrupted valid word: {word}"


def test_punctuation_spacing_still_applied():
    assert _fix_arabic_spacing("أكيد،شلونك") == "أكيد، شلونك"


def test_double_space_collapsed():
    assert _fix_arabic_spacing("كلمة   كلمة") == "كلمة كلمة"


def test_clean_sentence_unchanged():
    s = "الذكاء الاصطناعي يعتمد على الخوارزميات في الاقتصاد"
    assert _fix_arabic_spacing(s) == s


# ── Future-tense morph must not corrupt nouns/names starting with س ───────────
from backend.dialect.dialect_processor import dialectize

_NOUNS_WITH_SEEN = ["سيارة", "سياسة", "سينما", "ستيفن", "سنوات", "سيدة", "سنغافورة"]


@pytest.mark.parametrize("word", _NOUNS_WITH_SEEN)
def test_seen_nouns_not_turned_into_future(word):
    # Previously "سيارة"→"راح يارة" because the attached future prefix rules
    # fired on any word starting with سي/ست/سن/سأ.
    assert "راح" not in dialectize(word), f"corrupted noun: {word}"


def test_standalone_future_particle_still_converts():
    assert "راح" in dialectize("سوف يذهب")      # سوف → راح
    assert "راح" in dialectize("سيتم العمل")     # سيتم → راح يصير (whole word)
