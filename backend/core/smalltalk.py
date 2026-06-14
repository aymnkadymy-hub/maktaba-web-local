"""
Small-talk and library-query detection — pure functions, no imports from other backend modules.
Extracted here so context.py stays focused on RAG resolution.
"""
import re

_SMALL_TALK_RE = re.compile(
    r'^(?:'
    r'مرحب[اًا]?|هلا|(?:السلام|سلام)\s*(?:عليكم)?|أهلاً?|هيا?|هي'
    r'|hi|hello|hey|howdy|yo|sup|whats?\s*up'
    r'|كيف\s*(?:حالك|الحال|أنت|يومك|أخبارك|حالكم|حالكِ)'
    r'|كيفك|كيفكم|كيفكِ'
    r'|ايش\s*(?:أخبارك|الأخبار)|شو\s*أخبارك'
    r'|شخبارك|شخباركم|شلونك|شلونكم|شلون\s*حالك|شو\s*حالك|شو\s*أخبارك'
    r'|أهلين|يا\s*هلا|مرحبتين'
    r'|كلش\s*منيح|منيح|زين|باشر|عيني'
    r'|how\s*(?:are\s*you|r\s*u|you\s*doing|is\s*it\s*going)'
    r'|شكراً?\s*(?:جزيلاً?|لك)?|ألف\s*شكر|يسلموا|بارك\s*الله|جزاك\s*الله'
    r'|thanks?|thank\s*you|ty|thx'
    r'|مع\s*السلامة|وداعاً?|باي|bye|goodbye|cya|see\s*you'
    r'|صباح\s*(?:الخير|النور|الورد)|مساء\s*(?:الخير|النور|الورد)'
    r'|good\s*(?:morning|evening|night|afternoon|day)'
    r'|(?:أنا|اسمي|اسمي\s*هو|my\s+name\s+is)\s+\w+(?:\s+\w+)?'
    r'|تمام|أوكي|okay|ok|حسناً?|موافق|ماشي|طيب|عال|عالي'
    r'|نعم|لا|أيوه?|إي|ايوه?'
    r'|لا\s*شكراً?|شكراً?\s*لا|خلاص|بس|كافي|وداعاً?|يلا\s*وداع|يلا\s*باي'
    r'|مو\s*محتاج|مو\s*بحاجة|ماكو\s*شي|ماكو\s*سؤال|بسيط'
    r'|i\s*want\s*to\s*(?:talk|chat|speak|converse)'
    r'|(?:can|could)\s*(?:we|i)\s*(?:talk|chat|speak)'
    r'|fine|good|great|alright|sure|yep|yup|nope|yeah'
    r'|شكو\s*ماكو|شكو\s*ما\s*كو|لا\s*شكو|ماكو\s*شكو'
    r'|ولك\s*كافي|يكفي\s*هذا|كافيني|بس\s*هيچي|بس\s*اكو'
    r'|هاي|هاي\s*والله|تعبت|ملّيت|خلّها|ابد'
    r'|(?:هلا|مرحب[اًا]?|يا\s*هلا|أهلاً?)\s+(?:شلونك?|شخبارك?|كيف\s*(?:حالك?|أنت)|وينك)'
    r'|(?:شلونك?|شخبارك?)\s+(?:هلا|صاحبي|أخوي|حبيبي|والله)'
    r'|هاي\s*والله\s*(?:شلونك?|كيف\s*حالك?)'
    r')[\s!؟?،,\.]*$',
    re.IGNORECASE,
)

_DIALECT_META_RE = re.compile(
    r'(?:'
    r'شلون\s*(?:ترد|تجاوب|تقول|تحچي|تحكي|تتكلم)'
    r'|شت(?:جاوب|گول|قول|رد|حچي|حكي|كلم)[هوها]?'
    r'|شو\s*(?:تگول|تقول|تجاوب|ترد)'
    r'|كيف\s*(?:ترد|تجاوب|تتكلم|تحكي)'
    r'|احچي\s*عراقي|احكي\s*عراقي|تكلم\s*عراقي'
    r'|لهجة\s*عراقية|اللهجة\s*العراقية'
    r'|(?:إذا|اذا|لو)\s+(?:(?:واحد|حد|أحد|شخص)\s+)?(?:گالك|قالك|گال\s*لك|قال\s*لك)'
    r')',
    re.IGNORECASE,
)

_LIBRARY_KW = frozenset([
    "الكتب", "كتاب", "كتب", "مكتبة", "عندك", "عندكم", "موجود", "موجوده",
    "المتاحة", "المتاح", "اي كتب", "أي كتب", "ماكو كتاب",
    "book", "books", "library", "available",
])

# Social vocabulary — greetings, well-being, farewells, fillers. The anchored
# regex above only matches a message that is ENTIRELY one phrase, so multi-part
# greetings ("السلام عليكم، شلونك اليوم؟") slipped through and hit book search.
# The token check below catches them: if EVERY word is social it's small talk —
# but a single content word ("اشرح", "الذكاء") keeps it a real question.
_ST_TOKENS = frozenset("""
مرحبا مرحبتين هلا هلو اهلا أهلا أهلين اهلين السلام سلام عليكم وعليكم ورحمة الله وبركاته
كيف حالك الحال حالكم حالكي يومك اخبارك أخبارك كيفك كيفكم شلونك شلونكم شخبارك شخباركم وينك
شكو ماكو شكرا شكراً مشكور يسلموا تسلم تسلمو السلامة وداعا وداعاً باي صباح مساء الخير النور الورد
تمام اوكي اوك حسنا حسناً موافق ماشي طيب عال زين منيح كلش نعم لا ايوه أيوه اي خلص خلاص بس كافي يكفي
صاحبي اخوي أخوي حبيبي والله يا اليوم هاي تعبت مليت ملّيت عيني عزيزي جيد بخير الحمدلله حمدلله مع
hi hello hey thanks thank you bye goodbye ok okay fine good great yeah yep nope sure yo
""".split())
_PUNCT_SPLIT = re.compile(r'[،؛,\.!؟?:\-…]+')


def is_small_talk(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t) <= 3 and '?' not in t and '؟' not in t:
        return True
    if len(t) > 120:
        return False
    if _DIALECT_META_RE.search(t):
        return True
    if _SMALL_TALK_RE.match(t):
        return True
    # Multi-part social messages: small talk only if EVERY word is social
    words = [w for w in _PUNCT_SPLIT.sub(' ', t).split() if w]
    if words and len(words) <= 8 and all(w.lower() in _ST_TOKENS for w in words):
        return True
    return False


def is_library_query(q: str) -> bool:
    q_low = q.lower()
    return any(kw in q_low for kw in _LIBRARY_KW)
