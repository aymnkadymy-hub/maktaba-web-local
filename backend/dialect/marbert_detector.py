"""
Iraqi dialect detector using MARBERT (UBC-NLP/MARBERT).

MARBERT is a BERT-large model pretrained on 1B+ Arabic tweets from 21 Arab
countries. Its embeddings naturally cluster by dialect — we exploit this for
zero-shot Iraqi detection via cosine similarity with reference sentences.

Two detection modes:
1. Fast keyword check  (instant, no model needed)
2. MARBERT CLS-embedding similarity  (model downloaded ~1.2 GB on first use)

The model is loaded in a background thread on first Arabic message — subsequent
calls reuse the cached model with no latency.
"""
import re
import threading
import logging

logger = logging.getLogger("backend")

# ── Iraqi dialect keyword markers (high-precision) ────────────────────────────
# Includes root forms AND common suffixed variants (pronoun suffixes attach to roots
# in Arabic so regex extracts "شلونك" not "شلون" — we need both).
_IRAQI_KEYWORDS = frozenset([
    # Time / place
    "هسه", "هسة", "وين",
    # Existence
    "ماكو", "أكو", "اكو", "ماكوش",
    # Intensifiers (root + suffixed)
    "كلش", "كلشي", "كلشيا",
    "هواية", "هوايات",
    "گاع",
    # Pronouns
    "احنا", "اني",
    # تقدر variants (Iraqi ق→گ)
    "تگدر", "أگدر", "يگدر", "نگدر", "تگدرون",
    # قال variants
    "گال", "گلت", "يگول", "گلنا", "گالوا", "تگول",
    # كيف question root + variants
    "شلون", "شلونك", "شلونكم", "شلونهم", "شلوني",
    # Other question words
    "شنو", "شگد", "شصار", "شصاير", "منو",
    # Discourse markers
    "تره", "هيج", "هيجي", "ذول",
    # هذه → هاي
    "هاي", "هايچي",
    # Dialect words
    "حچي", "يحچي", "حچيت",
    "باوع", "يباوع", "باوعت",
    "مو",
    "شخبار", "شخباركم", "شخبارك",
    "شكو", "شكوماكو",
    "خوش", "بلكي",
    "وياه", "وياي", "وياك", "وياها", "وياهم", "وياناّ",
    "يمشي", "زبط", "زبطت",
    "چاي", "چان", "چانت", "چنت", "چنا",
    "صاير", "صارت", "شصار",
])

# ── Reference sentences for MARBERT similarity ────────────────────────────────
_IRAQI_REFS = [
    "شلونك؟ كلشي تمام؟",
    "هسه وين رايح؟",
    "ماكو مشكلة كلش",
    "باوعت هذا الموضوع هسه؟",
    "لازم تشوف هذا الشغل",
    "هواية ناس ما يعرفون هذا",
    "يعني شنو قصدك بالضبط؟",
    "أكيد أگدر أساعدك",
    "تره هذا الموضوع مهم كلش",
    "كلش زين هاي الفكرة",
    "گال يروح وياهم",
    "چان وياه هسه",
]

_MSA_REFS = [
    "كيف حالك؟ هل أنت بخير؟",
    "الآن إلى أين تذهب؟",
    "لا توجد مشكلة على الإطلاق",
    "هل فهمت هذا الموضوع الآن؟",
    "يجب عليك مشاهدة هذا العمل",
    "كثير من الناس لا يعرفون هذا",
    "ماذا تقصد بذلك بالضبط؟",
    "بالتأكيد أستطيع مساعدتك",
    "ينبغي الانتباه لهذا الأمر",
    "هذه فكرة ممتازة جداً",
    "قال إنه سيذهب معهم",
    "كان معه في تلك اللحظة",
]

# ── Thread-safe lazy model state ──────────────────────────────────────────────
_lock = threading.Lock()
_model = None
_tokenizer = None
_iraqi_emb = None
_msa_emb = None
_load_attempted = False


def _load_marbert() -> bool:
    """Download + load MARBERT and pre-compute reference embeddings."""
    global _model, _tokenizer, _iraqi_emb, _msa_emb, _load_attempted
    with _lock:
        if _load_attempted:
            return _model is not None
        _load_attempted = True
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        logger.info("MARBERT: جارٍ تحميل UBC-NLP/MARBERT (~1.2 GB)…")
        tok = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
        mdl = AutoModel.from_pretrained("UBC-NLP/MARBERT")
        mdl.eval()

        def _embed(texts):
            import torch.nn.functional as F
            vecs = []
            for t in texts:
                enc = tok(t, return_tensors="pt",
                          truncation=True, max_length=64, padding=True)
                with torch.no_grad():
                    out = mdl(**enc)
                vec = out.last_hidden_state.mean(dim=1)  # mean pooling
                vecs.append(F.normalize(vec, dim=-1))
            return torch.stack(vecs).squeeze(1)         # (N, hidden_dim)

        iq_emb  = _embed(_IRAQI_REFS)
        msa_emb = _embed(_MSA_REFS)

        with _lock:
            _tokenizer  = tok
            _model      = mdl
            _iraqi_emb  = iq_emb
            _msa_emb    = msa_emb
        logger.info("MARBERT: جاهز — كشف اللهجة العراقية نشط ✓")
        return True
    except Exception as exc:
        logger.warning(f"MARBERT: فشل التحميل ({exc}) — الكشف بالكلمات المفتاحية فعّال")
        return False


def _start_background_load():
    """Kick off MARBERT download in a daemon thread (non-blocking)."""
    t = threading.Thread(target=_load_marbert, daemon=True, name="marbert-loader")
    t.start()


# ── Public API ────────────────────────────────────────────────────────────────

def is_iraqi_dialect(text: str) -> bool:
    """
    Return True if *text* is likely Iraqi Arabic.

    Detection priority:
      1. Keyword fast-path  (≥2 Iraqi markers → True immediately)
      2. MARBERT similarity  (if model loaded)
      3. Keyword fallback   (1 marker → True)
    """
    if not text:
        return False

    # Extract Arabic words (including ڤ چ گ پ for Iraqi letters)
    words = set(re.findall(r'[ء-يپچگڤ]+', text))
    kw_hits = len(words & _IRAQI_KEYWORDS)

    if kw_hits >= 2:
        return True                      # strong keyword signal — skip model

    # Trigger model download on first Arabic message (runs in background)
    if not _load_attempted:
        _start_background_load()

    # Use MARBERT similarity if model is ready
    if _model is not None:
        try:
            import torch
            import torch.nn.functional as F
            enc = _tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=64, padding=True)
            with torch.no_grad():
                out = _model(**enc)
            vec = F.normalize(out.last_hidden_state.mean(dim=1), dim=-1)
            sim_iq  = (vec @ _iraqi_emb.T).mean().item()
            sim_msa = (vec @ _msa_emb.T).mean().item()
            return (sim_iq - sim_msa) > 0.01
        except Exception as exc:
            logger.debug(f"MARBERT inference error: {exc}")

    # Final fallback
    return kw_hits >= 1


def preload():
    """Call this at server startup to begin background model download."""
    if not _load_attempted:
        _start_background_load()
