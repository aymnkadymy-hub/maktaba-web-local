import re

SUPPORTED_LANGS = ['ar', 'en']
APOLOGY_MSG = "اعتذر انا لا افهم هذه اللغة انا اتحدث بالانجليزيه و العربيه فقط"

_ARABIC = re.compile(r'[؀-ۿ]')
_LATIN  = re.compile(r'[A-Za-z]')
# Letters that are NEITHER Arabic nor Latin (Cyrillic, Greek, CJK, Hebrew,
# Thai, Hangul …) — these mark a genuinely unsupported language.
_OTHER_SCRIPT = re.compile(r'[Ͱ-ϿЀ-ӿ֐-׿฀-๿぀-ヿ一-鿿가-힯]')


def validate_language(text: str) -> bool:
    """Accept Arabic or English; reject only genuinely other-script text.

    Script-based, not langdetect-based: langdetect misclassifies short/technical
    English (e.g. "How do you decompose a fraction into partial fractions?" was
    detected as Spanish) and wrongly triggered the apology. This app supports
    only Arabic + English, so the reliable test is which script dominates.
    """
    clean = text.strip()
    if len(clean) <= 8:
        return True

    arabic = len(_ARABIC.findall(clean))
    latin  = len(_LATIN.findall(clean))
    other  = len(_OTHER_SCRIPT.findall(clean))

    # Meaningful Arabic or Latin content → supported (covers mixed text such as
    # an Arabic question containing an English term).
    if arabic >= 2 or latin >= 3:
        return True

    # Reject only when another script clearly dominates the letters.
    if other >= 2 and (arabic + latin) < other:
        return False

    # No real letters (digits/symbols/emoji) or too short to judge → accept.
    return True

def enforce_output_rule(response_text: str) -> str:
    """
    يضمن أن الرد الخارج للمستخدم لا يخالف القيد اللغوي.
    """
    if validate_language(response_text):
        return response_text
    return APOLOGY_MSG

if __name__ == "__main__":
    # اختبار سريع للملف
    test_texts = ["Hello Aymen", "مرحباً أيمن", "Bonjour"]
    for t in test_texts:
        print(f"Text: {t} | Valid: {validate_language(t)}")
