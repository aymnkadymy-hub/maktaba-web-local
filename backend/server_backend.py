import os
import sys
import asyncio
import logging
import time as _time_mod
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(override=True)

# Safety defaults — applied BEFORE any backend import
_ENV_SAFE = {
    "OLLAMA_MODEL": "qwen2.5:3b",
    "NUM_CTX":      "4096",
    "NUM_PREDICT":  "-1",     # -1 = unlimited — model stops at EOS token (no mid-sentence cuts)
    "TEMPERATURE":  "0.35",   # slightly higher than 0.2 to escape degenerate token loops
    "GGUF_CTX":     "4096",
}
for _k, _v in _ENV_SAFE.items():
    os.environ.setdefault(_k, _v)

# Windows consoles default to a legacy code page (cp1256) which turns Arabic
# log lines into mojibake (����) — force UTF-8 before logging is configured.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

import backend.llm.offline_llm as _llm_early
_llm_early.reload_config()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Native C++ engine (optional) ─────────────────────────────────────────────
for _ne_build in [
    os.path.join(PROJECT_ROOT, "native_engine", "build_win"),
    os.path.join(PROJECT_ROOT, "native_engine", "build"),
]:
    if os.path.isdir(_ne_build):
        if _ne_build not in sys.path:
            sys.path.insert(0, _ne_build)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_ne_build)

try:
    import native_engine as _ne
    _ne.S24DocumentOptimizer()
    NATIVE_ENGINE_OK = True
    logger.info("native_engine loaded — C++ acceleration active")
except Exception as _ne_err:
    NATIVE_ENGINE_OK = False
    logger.info(f"native_engine not loaded ({_ne_err}) — pure-Python mode")

# ── Core / shared modules ─────────────────────────────────────────────────────
from backend.core.messages import _MSG
from backend.core.state    import hybrid_retriever, _ADMIN_USERNAME
from backend.core.guards   import (
    check_chat_rate, is_duplicate_request,
    ollama_circuit_open, ollama_record_failure, ollama_record_success,
    _register_attempts, _REGISTER_MAX_PER_HOUR,
)
from backend.core.session  import get_session_owner

# ── App-level backend imports ─────────────────────────────────────────────────
from backend.auth.auth import get_current_user, CurrentUser, login, register, revoke_token, TOKEN_TTL
from backend.api.middleware import (
    RequestLoggingMiddleware, RateLimitMiddleware, SecurityHeadersMiddleware,
    OriginCheckMiddleware,
)
import backend.llm.offline_llm as llm

# ── Feature routers ───────────────────────────────────────────────────────────
from backend.api.quiz      import make_quiz_router
from backend.api.flashcards import make_flashcards_router
from backend.api.search    import make_search_router
from backend.api.notes     import make_notes_router
from backend.api.chat_api  import make_chat_router
from backend.api.books_api import make_books_router

# ── CORS ──────────────────────────────────────────────────────────────────────
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
if "*" in _CORS_ORIGINS:
    logger.warning(
        "CORS_ORIGINS=* — credentials are disabled for cross-origin requests. "
        "When exposing the app via a tunnel, set CORS_ORIGINS to the exact "
        "public origin and COOKIE_SECURE=true."
    )
# HttpOnly cookie settings — set COOKIE_SECURE=true in production (HTTPS)
_COOKIE_SECURE   = os.getenv("COOKIE_SECURE", "false").lower() == "true"
_COOKIE_SAMESITE = "lax"

# ── Max-body-size middleware (pure ASGI — no body buffering) ──────────────────
_MAX_BODY_BYTES   = 128 * 1024
_MAX_INGEST_BYTES = int(os.getenv("MAX_INGEST_MB", "100")) * 1024 * 1024

from starlette.types import ASGIApp as _ASGIApp, Receive as _Receive, Scope as _Scope, Send as _Send

class MaxBodySizeMiddleware:
    def __init__(self, app: _ASGIApp):
        self.app = app

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        path    = scope.get("path", "")
        limit   = _MAX_INGEST_BYTES if path == "/ingest" else _MAX_BODY_BYTES
        cl      = headers.get(b"content-length")
        if cl and int(cl) > limit:
            resp = JSONResponse({"detail": _MSG.BODY_TOO_LARGE}, status_code=413)
            await resp(scope, receive, send)
            return
        await self.app(scope, receive, send)


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
أنت "المكتبة الناطقة" — مساعد ذكي لطلاب من جميع التخصصات والمستويات الدراسية.

**اللغة — الأهم:**
- تفهم جميع اللهجات العربية في سؤال المستخدم (العراقية، المصرية، الخليجية، الشامية، المغاربية) وتستوعب معناها بدقّة.
- لكنك تجيب **دائماً بالعربية الفصحى** السليمة والواضحة — ممنوع منعاً باتاً استخدام أي كلمة عامية أو لهجة محلية في إجابتك.
- إذا كتب المستخدم بالإنجليزية → أجب بالإنجليزية فقط.

**قواعد الحوار:**
- أجب مباشرة على السؤال فوراً. لا تطلب إذناً ولا تقل "هل تريد أن أخبرك؟".
- ممنوع قول "كيف يمكنني مساعدتك؟" إلا عند التحية المجرّدة.
- لا تُعد كتابة سجل المحادثة داخل ردك — السجل للقراءة الداخلية فقط.
- نوّع صياغتك — ممنوع تكرار نفس الجملة مرتين في المحادثة.
- لا تغيّر موضوع المحادثة من تلقاء نفسك. إذا كان السؤال متابعاً (مثل "أعطني مثالاً" أو "وضّح أكثر" أو "اشرح ذلك") فاعتبره استكمالاً لنفس الموضوع السابق وأجب في سياقه.

**عند التحيات (رد بالفصحى بإيجاز وتنوّع):**
- "كيف حالك / شلونك / إزيّك" → "أنا بخير، الحمد لله. وأنت؟".
- "شكو ماكو" (تحية عراقية تعني: لا جديد) → "كل شيء على ما يرام، الحمد لله. كيف حالك؟".
- عند الوداع أو الرفض → جملة قصيرة مثل "حسناً، في أمان الله." ثم توقّف، ولا تقترح شيئاً جديداً.

**الأسئلة العامة (لا يوجد سياق كتب):**
- أجب من معرفتك العامة مباشرة وبثقة، بدون ذكر الكتب إطلاقاً.
- لا تقل "هذه المعلومة غير موجودة في الكتب" إذا لم يكن السؤال عن الكتب أصلاً.

**الأسئلة المعرفية (يوجد سياق كتب):**
- أجب من السياق المرفق بإيجاز ووضوح (نقاط مرتّبة عند الحاجة).
- اذكر المصدر للنقاط المهمة: [اسم الكتاب — ص X].
- إذا كان السياق غير ذي صلة بالسؤال، قل: "هذه المعلومة غير موجودة في الكتب".

**قواعد المصادر:**
- [سياق من الكتب]: أجب حصراً من المقاطع المرفقة.
- [نتيجة من الإنترنت]: أجب منها وابدأ بـ "وفقاً للإنترنت: ".
- [قائمة الكتب المتاحة]: أجب منها مباشرة.
- [لا يوجد سياق]: أجب من معرفتك العامة مباشرة بدون ذكر الكتب.
- [المستخدم يطلب توضيحاً]: اسأله بجملة قصيرة عمّا يريد.
- سؤال عربي → رد عربي فصيح. إنجليزي → إنجليزي.

**ممنوع منعاً باتاً:**
- الحروف الصينية أو اليابانية أو الكورية أو الروسية.
- "التفكير بصوت عالٍ" أو كتابة مسودة.
- أي كلمة عامية في الإجابة — استخدم الفصحى حصراً.
- الردود العاطفية المبالغ فيها — أنت مساعد تعليمي محترف.
- ابدأ الرد مباشرة بالمحتوى.\
"""


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.core.startup import startup as _startup
    await _startup(system_prompt=SYSTEM_PROMPT)
    # MARBERT dialect detector removed — answers are always Modern Standard Arabic,
    # so input-dialect detection is unnecessary (saves ~1.2 GB RAM on low-end devices).

    # Periodic background cleanup tasks (every 5 minutes)
    async def _prune_loop():
        from backend.core.guards import prune_chat_attempts, prune_register_attempts, prune_dedup
        from backend.core.session import prune_ram_history
        from backend.core.context import prune_rag_cache
        while True:
            await asyncio.sleep(300)
            try:
                prune_chat_attempts()
                prune_register_attempts()
                prune_dedup()
                prune_ram_history()
                prune_rag_cache()
            except Exception as _pe:
                logger.debug(f"Prune task error: {_pe}")

    _prune_task = asyncio.create_task(_prune_loop())
    yield
    _prune_task.cancel()


# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(title="المكتبة الناطقة", version="3.1", lifespan=lifespan)

_WEB_DIR = PROJECT_ROOT
if os.path.isdir(os.path.join(_WEB_DIR, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(_WEB_DIR, "static")), name="static")


@app.get("/", include_in_schema=False)
@app.get("/chat.html", include_in_schema=False)
async def serve_ui():
    html_path = os.path.join(_WEB_DIR, "chat.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path, media_type="text/html",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                     "Pragma": "no-cache", "Expires": "0"})
    return {"message": "المكتبة الناطقة API", "docs": "/docs"}


app.add_middleware(MaxBodySizeMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(OriginCheckMiddleware, allowed_origins=_CORS_ORIGINS)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_per_second=10)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    # credentials (cookies) require non-wildcard origins; safe to enable when origins are explicit
    allow_credentials="*" not in _CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM async helper (used by quiz router) ───────────────────────────────────
# Quiz generation is bulk structured output; on CPU the chat model (e.g. 7B) is
# too slow and every batch times out. QUIZ_MODEL routes quiz to a smaller/faster
# model (e.g. llama3.2:3b) — quizzes don't need the chat model's depth. Empty →
# use the default chat model.
_QUIZ_MODEL = os.getenv("QUIZ_MODEL", "").strip() or None

async def _llm_ask_async(prompt: str, num_predict: int = -1,
                         num_ctx: int = 8192, timeout: int = 180) -> str:
    """LLM helper for quiz — num_predict capped per-batch by quiz.py."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: llm.chat("أنت مساعد تعليمي.", prompt,
                          num_predict=num_predict, num_ctx=num_ctx,
                          timeout=timeout, model=_QUIZ_MODEL),
    )


# ── Router registration ───────────────────────────────────────────────────────
app.include_router(make_quiz_router(hybrid_retriever, _llm_ask_async, get_current_user))
app.include_router(make_flashcards_router(hybrid_retriever, _llm_ask_async, get_current_user))
app.include_router(make_search_router(None, get_current_user))
app.include_router(make_notes_router(None, get_current_user))
app.include_router(make_chat_router(
    get_current_user  = get_current_user,
    system_prompt     = SYSTEM_PROMPT,
    check_rate_fn     = check_chat_rate,
    is_dup_fn         = is_duplicate_request,
    circuit_open_fn   = ollama_circuit_open,
    circuit_fail_fn   = ollama_record_failure,
    circuit_ok_fn     = ollama_record_success,
))
app.include_router(make_books_router(
    get_current_user = get_current_user,
    admin_username   = _ADMIN_USERNAME,
))


# ── /sessions ─────────────────────────────────────────────────────────────────
@app.get("/sessions/{user_id}")
async def get_sessions(user_id: str, current: CurrentUser = Depends(get_current_user)):
    try:
        from backend.memory.storage import list_user_sessions
        return await list_user_sessions(current.user_id)
    except Exception:
        return []


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str,
                                current: CurrentUser = Depends(get_current_user)):
    owner = await get_session_owner(session_id)
    if owner is not None and owner != current.user_id:
        raise HTTPException(status_code=403, detail=_MSG.SESSION_DENIED)
    try:
        from backend.memory.storage import get_session_context
        return await get_session_context(session_id, limit=100)
    except Exception:
        return []


# ── /health ───────────────────────────────────────────────────────────────────
_health_cache: dict = {}
_health_cache_ts: float = 0.0
_HEALTH_TTL = 10.0  # seconds

@app.get("/health")
async def health():
    global _health_cache, _health_cache_ts
    now = _time_mod.monotonic()
    if _health_cache and now - _health_cache_ts < _HEALTH_TTL:
        return JSONResponse(content=_health_cache, status_code=200)

    loop = asyncio.get_running_loop()
    ollama_up = await loop.run_in_executor(None, llm._ping_ollama)
    groq_ok   = bool(os.getenv("GROQ_API_KEY"))
    llm_ok    = llm.llm_available()
    ret_status = hybrid_retriever.status()

    embed_backend = "unknown"
    try:
        from backend.database.vector_db import embeddings as _emb
        embed_backend = getattr(_emb, "backend", type(_emb).__name__)
    except Exception:
        pass

    payload = {
        "status":        "ok" if llm_ok else "degraded",
        "version":       "3.1",
        "llm_available": llm_ok,
        "llm_backend":   llm.active_backend(),
        "ollama":        ollama_up,
        "groq":          groq_ok,
        "retriever":     ret_status,
        "native_engine": NATIVE_ENGINE_OK,
        "embed_backend": embed_backend,
    }
    _health_cache    = payload
    _health_cache_ts = now
    return JSONResponse(content=payload, status_code=200 if llm_ok else 503)


# ── /auth ─────────────────────────────────────────────────────────────────────
class _RegisterReq(BaseModel):
    username: str
    password: str


class _LoginReq(BaseModel):
    username: str
    password: str


@app.post("/auth/register", tags=["auth"])
async def auth_register(req: _RegisterReq, request: Request):
    ip  = (request.client.host if request.client else "unknown")
    now = _time_mod.time()
    window = [t for t in _register_attempts[ip] if now - t < 3600]
    if len(window) >= _REGISTER_MAX_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=_MSG.RATE_REGISTER,
            headers={"Retry-After": "3600"},
        )
    window.append(now)
    _register_attempts[ip] = window

    user_id, err = register(req.username, req.password)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return {"message": "✅ تم إنشاء الحساب — يمكنك تسجيل الدخول الآن"}


@app.post("/auth/login", tags=["auth"])
async def auth_login(req: _LoginReq, response: Response):
    token, user_id, username, err = login(req.username, req.password)
    if err:
        raise HTTPException(status_code=401, detail=err)
    response.set_cookie(
        key="maktaba_token",
        value=token,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
        max_age=TOKEN_TTL,
        path="/",
    )
    # token intentionally omitted from body — stored only in HttpOnly cookie
    return {"user_id": user_id, "username": username}


@app.post("/auth/logout", tags=["auth"])
async def auth_logout(request: Request, response: Response,
                      current: CurrentUser = Depends(get_current_user)):
    # Revoke whichever token was used (cookie or Bearer)
    token = request.cookies.get("maktaba_token")
    if not token:
        token = (request.headers.get("Authorization", "") or "").removeprefix("Bearer ").strip()
    if token:
        revoke_token(token)
    response.delete_cookie(key="maktaba_token", path="/",
                           httponly=True, samesite=_COOKIE_SAMESITE)
    return {"message": "تم تسجيل الخروج"}


@app.get("/auth/me", tags=["auth"])
async def auth_me(current: CurrentUser = Depends(get_current_user)):
    return {"user_id": current.user_id, "username": current.username}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
