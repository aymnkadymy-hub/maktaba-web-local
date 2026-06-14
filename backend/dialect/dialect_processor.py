"""
Iraqi dialect post-processor.

Loads dialect_map.json and dialect_examples.json (built from real Iraqi
datasets: IADD 135K texts + IA2D 1,673 tweets + iraqi_dialect_llm training data).

Provides:
  - dialectize(text)           → apply word substitutions to LLM output
  - get_system_prompt_block()  → few-shot Iraqi examples for system prompt
  - get_response_prefix(type)  → natural Iraqi opener for each response type
"""
import json
import re
import os
import random
from typing import Optional

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load resources ────────────────────────────────────────────────────────────
def _load_json(name):
    path = os.path.join(_DIR, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

_MAP      = _load_json("dialect_map.json")    # str → str substitution
_EXAMPLES = _load_json("dialect_examples.json")  # curated + real sentences

# Sort by length descending so longer phrases match first (e.g. "لا يوجد" before "يوجد")
_MAP_SORTED = sorted(_MAP.items(), key=lambda x: len(x[0]), reverse=True)

# Pre-compile once at module load — avoids re.compile() on every dialectize() call
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?<![ء-ي])' + re.escape(src) + r'(?![ء-ي])'), dst)
    for src, dst in _MAP_SORTED
]

# ── Arabic word-merge spacing fixer ──────────────────────────────────────────
# The previous version tried to split LLM word-merge artifacts (a qwen2.5:3b
# problem) with patterns like "split on لا / على / مع / pronouns". Those fired
# INSIDE valid words — measured live, they corrupted 17/21 common words
# (الاقتصاد→"الا قتصاد", المعلومات→"ال معلومات", الفهم→"الف هم", …) because
# particles like "لا"/"مع"/"في" appear as substrings of ordinary words and no
# regex can tell "لاأعرف" (glued) from "الاقتصاد" (one word) without a lexicon.
# The deployed model is now 7b, which rarely merges words, so the safe trade is
# to keep ONLY the two rules that cannot touch letters inside a word.
_SPACING_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Space after Arabic/Latin punctuation: "أكيد،شلونك" → "أكيد، شلونك"
    (re.compile(r'([،,؛;:])([^\s\d\)\]\}])'), r'\1 \2'),
    # Collapse doubled spaces
    (re.compile(r'  +'), ' '),
]


def _fix_arabic_spacing(text: str) -> str:
    """Normalise spacing without ever splitting a valid word.

    Only punctuation spacing + double-space collapse — verified zero corruption
    across common Arabic vocabulary (see the note above)."""
    for pattern, repl in _SPACING_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


# ── Morphological dialect patterns (applied BEFORE word substitution) ─────────
# These handle verb-tense and structural conversions that can't be done
# word-by-word, ordered from most specific to most general.
_MORPH_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Future tense → "راح + verb".
    # NOTE: the attached سأ/سي/ست/سن prefixes were REMOVED — they can't be told
    # apart from nouns/names that merely start with those letters, so they
    # corrupted 14/18 common words (سيارة→"راح يارة", سياسة→"راح ياسة",
    # ستيفن→"راح تيفن", سنوات→"راح نوات", …). Only the unambiguous standalone
    # particle "سوف" is converted; "سيتم" is handled as a whole word below.
    (re.compile(r'(?<![ء-ي])سوف ([أيتن])'), r'راح \1'),

    # يتم/تتم/يُتم → يصير
    (re.compile(r'(?<![ء-ي])يتم(?![ء-ي])'), 'يصير'),
    (re.compile(r'(?<![ء-ي])تتم(?![ء-ي])'), 'تصير'),
    (re.compile(r'(?<![ء-ي])يُتم(?![ء-ي])'), 'يصير'),
    # تم (passive past) → صار
    (re.compile(r'(?<![ء-ي])تم(?![ء-ي])'), 'صار'),
    # سيتم → راح يصير
    (re.compile(r'(?<![ء-ي])سيتم(?![ء-ي])'), 'راح يصير'),

    # دعنا/دعني/دعه/دعها → خلنا/خليني/خله/خليها
    (re.compile(r'(?<![ء-ي])دعنا(?![ء-ي])'), 'خلنا'),
    (re.compile(r'(?<![ء-ي])دعني(?![ء-ي])'), 'خليني'),
    (re.compile(r'(?<![ء-ي])دعه(?![ء-ي])'),  'خله'),
    (re.compile(r'(?<![ء-ي])دعها(?![ء-ي])'), 'خليها'),
    (re.compile(r'(?<![ء-ي])دعهم(?![ء-ي])'), 'خليهم'),

    # يتحدث/تتحدث/يتكلم → يحچي
    (re.compile(r'(?<![ء-ي])يتحدث(?![ء-ي])'), 'يحچي'),
    (re.compile(r'(?<![ء-ي])تتحدث(?![ء-ي])'), 'تحچي'),
    (re.compile(r'(?<![ء-ي])يتكلم(?![ء-ي])'),  'يحچي'),
    (re.compile(r'(?<![ء-ي])تتكلم(?![ء-ي])'),  'تحچي'),
    (re.compile(r'(?<![ء-ي])تحدثنا(?![ء-ي])'), 'حچينا'),
    (re.compile(r'(?<![ء-ي])تحدث(?![ء-ي])'),   'حچى'),

    # يُشير/يشير → يگول / تشير → تگول
    (re.compile(r'(?<![ء-ي])يُشير(?![ء-ي])'), 'يگول'),
    (re.compile(r'(?<![ء-ي])يشير(?![ء-ي])'),  'يگول'),
    (re.compile(r'(?<![ء-ي])تشير(?![ء-ي])'),  'تگول'),

    # يُعدّ/يُعدّ من → يعتبر
    (re.compile(r'(?<![ء-ي])يُعدّ(?![ء-ي])'), 'يعتبر'),
    (re.compile(r'(?<![ء-ي])يعدّ(?![ء-ي])'),  'يعتبر'),
]


def _apply_morph(text: str) -> str:
    """Apply morphological dialect transformations (future tense, يتم, etc.)."""
    for pat, repl in _MORPH_PATTERNS:
        text = pat.sub(repl, text)
    return text


# ── Word substitution ─────────────────────────────────────────────────────────
def dialectize(text: str) -> str:
    """
    Apply Iraqi dialect transformations to text.
    1. Morphological patterns (future tense سـ, يتم, دعنا, etc.)
    2. Word substitution map (كثير→هواية, الآن→هسه, etc.)
    3. Spacing fixer (remove BPE merge artifacts)
    Skips English text and markdown code blocks.
    """
    if not text:
        return text

    # Preserve code blocks unchanged
    code_blocks = []
    def stash_code(m):
        code_blocks.append(m.group(0))
        return f"\x00CODE{len(code_blocks)-1}\x00"
    text = re.sub(r'```[\s\S]*?```|`[^`]+`', stash_code, text)

    # Step 1 — morphological transformations (future tense, يتم, دعنا…)
    text = _apply_morph(text)

    # Step 2 — word substitution map
    if _COMPILED_PATTERNS:
        for pattern, dst in _COMPILED_PATTERNS:
            text = pattern.sub(dst, text)

    # Step 3 — fix any spacing artifacts left behind
    text = _fix_arabic_spacing(text)

    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODE{i}\x00", block)

    return text


# ── Curated response openers ──────────────────────────────────────────────────
_CURATED = _EXAMPLES.get("curated", {})

def get_response_prefix(response_type: str) -> str:
    """
    Return a natural Iraqi opener for the given response type.
    Types: greeting, how_are_you, shakhbarak, closing, no_context, book_found, web_found
    """
    key = response_type + "_response"
    options = _CURATED.get(key, [])
    if options:
        return random.choice(options)
    return ""


# ── System prompt block ───────────────────────────────────────────────────────
_PROMPT_CACHE: Optional[str] = None


def get_system_prompt_block() -> str:
    """
    Return the few-shot Iraqi examples block to append to SYSTEM_PROMPT.
    Cached after first call.
    Only uses curated examples — real_sentences are excluded because social-media
    posts contain garbage (hashtags, transliteration, latin chars) that causes the
    model to hallucinate similar garbage in its responses.
    """
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE

    curated = _EXAMPLES.get("curated", {})

    # The greeting examples are scoped to greeting messages ONLY. Without that
    # scoping (and with the old "follow literally" wording) the model injected
    # "شلونك؟ شخبارك؟" into the middle of technical answers, because it copied the
    # greeting few-shots into every reply.
    lines = [
        "",
        "**أمثلة الردود العراقية للتحيات فقط (استخدمها فقط لو كانت رسالة المستخدم تحية أو سؤال عن الحال):**",
        f'- تحية: "{curated.get("greeting_response",["هلا بيك، تفضل."])[0]}"',
        f'- كيف حالك: "{curated.get("how_are_you_response",["الحمدلله، كلشي تمام. وأنت؟"])[0]}"',
        f'- شخبارك: "{curated.get("shakhbarak_response",["الحمدلله ماكو غير الخير. وأنت شلونك؟"])[0]}"',
        f'- إنهاء: "{curated.get("closing_response",["زين، في أمان الله."])[0]}"',
        '- شكو ماكو: "كلشي ماكو، الحمدلله. وأنت شلونك؟"',
        "",
        "**للأسئلة التقنية أو المعرفية: أجب مباشرة بالمحتوى بلهجة عراقية، "
        "وممنوع منعاً باتاً إدراج أي تحية (شلونك/شخبارك/هلا) داخل الإجابة.**",
        '- مثال سؤال تقني: المستخدم: "شنو المتغير بالبرمجة؟" → '
        '"المتغير مكان بالذاكرة تخزن بيه قيمة تگدر تغيّرها. مثلاً x=5 يخزن الرقم 5."',
    ]

    _PROMPT_CACHE = "\n".join(lines)
    return _PROMPT_CACHE


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test = "أنا الآن لا يوجد كثير من الوقت، لكن كل شيء جيد."
    print("Original :", test)
    print("Dialectized:", dialectize(test))
    print()
    print("System block:")
    print(get_system_prompt_block())
    print()
    print("Greeting prefix:", get_response_prefix("greeting"))
