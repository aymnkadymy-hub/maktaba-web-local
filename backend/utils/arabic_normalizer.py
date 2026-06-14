"""
تطبيع النص العربي قبل التقطيع والتضمين.
يستخدم pyarabic إن وُجدت، وإلا regex مدمج.
"""
import re

try:
    import pyarabic.araby as araby
    _HAS_PYARABIC = True
except ImportError:
    _HAS_PYARABIC = False

# نطاق أحرف Unicode العربية
_ARABIC_RE = re.compile(r'[؀-ۿ]')


def is_arabic(text: str) -> bool:
    """يُحدد إذا كان النص يحتوي عربياً بنسبة 30% على الأقل."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    arabic_count = sum(1 for c in chars if _ARABIC_RE.match(c))
    return arabic_count / len(chars) > 0.3


def _normalize_regex(text: str) -> str:
    """تطبيع عربي بالـ regex — يعمل بدون مكتبات خارجية."""
    # إزالة التشكيل والشدة والمدة
    text = re.sub(r'[ً-ٰٟ]', '', text)
    # إزالة التطويل (tatweel ـ)
    text = re.sub(r'ـ', '', text)
    # توحيد الألف (أ إ آ ٱ → ا)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # توحيد الياء (ى → ي)
    text = re.sub(r'ى', 'ي', text)
    # توحيد لام-ألف
    text = re.sub(r'لآ|لأ|لإ', 'لا', text)
    # تنظيف المسافات
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_arabic(text: str) -> str:
    """يُطبّع النص العربي باستخدام pyarabic أو regex."""
    if _HAS_PYARABIC:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        text = araby.normalize_alef(text)
        text = araby.normalize_ligature(text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    return _normalize_regex(text)


def smart_normalize(text: str) -> str:
    """يُطبّع النص إذا كان عربياً، ويُرجعه كما هو إذا كان إنجليزياً."""
    if is_arabic(text):
        return normalize_arabic(text)
    # تنظيف بسيط للنص الإنجليزي
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
