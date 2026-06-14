"""Arabic normalization and safe error messages."""
from backend.core.messages import _MSG
from backend.utils.arabic_normalizer import is_arabic, normalize_arabic, smart_normalize


def test_is_arabic_detection():
    assert is_arabic("ما هو الجهد الكهربائي؟")
    assert not is_arabic("What is voltage?")
    assert not is_arabic("")


def test_normalize_strips_tashkeel_and_tatweel():
    assert normalize_arabic("كِتَــــابٌ") == "كتاب"


def test_normalize_unifies_alef_and_ya():
    assert normalize_arabic("أحمد إلى آخر") == "احمد الي اخر"
    assert normalize_arabic("مستشفى") == "مستشفي"


def test_smart_normalize_keeps_english_content():
    assert smart_normalize("hello   world") == "hello world"


def test_error_messages_do_not_leak_exception_details():
    secret = Exception("C:\\Users\\Ayman\\secret\\path leaked")
    for fn in (_MSG.LLM_ERROR, _MSG.INGEST_FAILED, _MSG.DELETE_FAILED, _MSG.IMG_FAIL):
        msg = fn(secret)
        assert "secret" not in msg and "Ayman" not in msg
        assert fn() == msg, "must also work with no argument"
