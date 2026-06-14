"""
Multi-provider LLM client — offline-first, cloud-ready.

Provider selection via LLM_PROVIDER env var:
  auto       → llamacpp → ollama → groq (existing priority chain — default)
  llamacpp   → local .gguf via llama-cpp-python (100% offline)
  ollama     → Ollama localhost service (100% offline)
  groq       → Groq API (internet required, ~300ms, free tier)
  openai     → OpenAI API (internet required, GPT-4o)
  anthropic  → Anthropic API (internet required, Claude)

Quick setup per provider:
  llamacpp:  pip install llama-cpp-python && set LOCAL_GGUF_PATH=C:/models/qwen2.5-3b-instruct-q4_k_m.gguf
  ollama:    winget install Ollama.Ollama && ollama pull qwen2.5:3b
  groq:      set GROQ_API_KEY=sk-...
  openai:    pip install openai && set OPENAI_API_KEY=sk-...
  anthropic: pip install anthropic && set ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import re
import json
import logging
import threading
import requests
from typing import Iterator

logger = logging.getLogger("offline_llm")

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER",  "auto")   # auto|llamacpp|ollama|groq|openai|anthropic
OLLAMA_URL     = os.getenv("OLLAMA_URL",    "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",  "qwen2.5:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # seconds; quiz batches can take ~2 min
VISION_MODEL  = os.getenv("VISION_MODEL",  "llava:7b")
GROQ_MODEL    = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.2"))
NUM_CTX       = int(os.getenv("NUM_CTX",    "8192"))
NUM_PREDICT   = int(os.getenv("NUM_PREDICT", "-1"))
LOCAL_GGUF    = os.getenv("LOCAL_GGUF_PATH", "")
GGUF_CTX      = int(os.getenv("GGUF_CTX",    "8192"))
GGUF_GPU      = int(os.getenv("GGUF_GPU_LAYERS", "-1"))

_ollama_ok: bool | None = None
_ollama_ok_ts: float = 0.0
_PING_TTL = 60.0
_ping_lock = threading.Lock()
_llama_cpp_model = None
_llama_cpp_lock  = threading.Lock()
_http_session: "requests.Session | None" = None

# ── Token filtering ───────────────────────────────────────────────────────────
_THINK_RE   = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
_CHINESE_RE = re.compile(r'[一-鿿㐀-䶿　-〿＀-￯]')


def _strip_think(text: str) -> str:
    return _THINK_RE.sub('', text).lstrip()


def _strip_chinese_preamble(text: str) -> str:
    if not _CHINESE_RE.search(text):
        return text
    if "\n\n" in text:
        before, after = text.split("\n\n", 1)
        after = after.strip()
        if _CHINESE_RE.search(before) and after and not _CHINESE_RE.search(after[:80]):
            return after
    cleaned = re.sub(
        r'^[一-鿿㐀-䶿　-〿＀-￯'
        r'！-／、-。\n\r\s：。，？！、；：""'']+',
        '', text
    ).strip()
    return cleaned if cleaned else text


def _filter_think_stream(tokens: Iterator[str]) -> Iterator[str]:
    in_think = False
    buf = ""
    for tok in tokens:
        buf += tok
        while True:
            if in_think:
                end = buf.find("</think>")
                if end >= 0:
                    buf = buf[end + 8:].lstrip("\n")
                    in_think = False
                else:
                    buf = ""
                    break
            else:
                start = buf.find("<think>")
                if start >= 0:
                    if start > 0:
                        yield buf[:start]
                    buf = buf[start + 7:]
                    in_think = True
                else:
                    safe = max(0, len(buf) - 7)
                    if safe > 0:
                        yield buf[:safe]
                        buf = buf[safe:]
                    break
    if not in_think and buf:
        yield buf


def _filter_chinese_preamble_stream(tokens: Iterator[str]) -> Iterator[str]:
    content_started = False
    buf = ""
    for tok in tokens:
        if content_started:
            yield tok
            continue
        buf += tok
        if "\n\n" in buf:
            before, after = buf.split("\n\n", 1)
            if _CHINESE_RE.search(before):
                buf = after
                if after.strip():
                    content_started = True
                    yield after
            else:
                content_started = True
                yield buf
                buf = ""
        elif not _CHINESE_RE.search(buf) and buf.strip():
            content_started = True
            yield buf
            buf = ""
    if buf:
        clean = re.sub(r'^[一-鿿㐀-䶿　-〿＀-￯\n\r\s：。，？！、；]+', '', buf).strip()
        yield clean if clean else buf


# ── Utility ───────────────────────────────────────────────────────────────────

def _ping_ollama() -> bool:
    global _ollama_ok, _ollama_ok_ts
    import time as _t
    now = _t.monotonic()
    if _ollama_ok is not None and (now - _ollama_ok_ts) < _PING_TTL:
        return _ollama_ok
    with _ping_lock:
        if _ollama_ok is not None and (now - _ollama_ok_ts) < _PING_TTL:
            return _ollama_ok
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            _ollama_ok = r.status_code == 200
        except Exception:
            _ollama_ok = False
        _ollama_ok_ts = _t.monotonic()
    level = "info" if _ollama_ok else "debug"
    getattr(logger, level)(
        f"Ollama {'detected' if _ollama_ok else 'not running'} at {OLLAMA_URL}"
    )
    return _ollama_ok


def reset_cache():
    global _ollama_ok, _ollama_ok_ts
    _ollama_ok = None
    _ollama_ok_ts = 0.0


def reload_config():
    global OLLAMA_MODEL, TEMPERATURE, NUM_CTX, NUM_PREDICT, GGUF_CTX, LLM_PROVIDER
    global GROQ_MODEL, OPENAI_MODEL, ANTHROPIC_MODEL, _OLLAMA_OPTS
    LLM_PROVIDER    = os.getenv("LLM_PROVIDER",  "auto")
    OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",  "qwen2.5:3b")
    GROQ_MODEL      = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")
    OPENAI_MODEL    = os.getenv("OPENAI_MODEL",  "gpt-4o-mini")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.35"))
    NUM_CTX         = int(os.getenv("NUM_CTX",      "4096"))
    NUM_PREDICT     = int(os.getenv("NUM_PREDICT",  "-1"))    # -1 = generate until EOS
    GGUF_CTX        = int(os.getenv("GGUF_CTX",    "4096"))
    _OLLAMA_OPTS = {
        "temperature":    TEMPERATURE,
        "num_predict":    NUM_PREDICT,
        "num_ctx":        NUM_CTX,
        "num_batch":      512,
        "num_keep":       64,   # keep first 64 tokens (core identity) during context eviction
        "num_thread":     0,     # 0 = auto-detect optimal thread count
        # 1.3 was too aggressive for Arabic: the language naturally repeats short
        # words/letters, so a high penalty pushed the model toward rare FOREIGN
        # tokens to avoid repeating (observed live: Russian "память"/"мест" and
        # Latin fragments bleeding into Arabic). 1.1 removed the bleed but let
        # some repetition loops back in; 1.15 is the sweet spot found live — no
        # foreign-token bleed and no repetition loops.
        "repeat_penalty": float(os.getenv("REPEAT_PENALTY", "1.15")),
        "top_k":          40,    # limits vocab at each step — tighter than default (0)
        "top_p":          0.9,   # nucleus sampling — avoids low-probability garbage tokens
    }
    logger.info(
        f"[LLM config] provider={LLM_PROVIDER} model={OLLAMA_MODEL} "
        f"ctx={NUM_CTX} predict={NUM_PREDICT} temp={TEMPERATURE}"
    )


def _get_http_session() -> "requests.Session":
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.headers.update({"Connection": "keep-alive"})
    return _http_session


# ── Provider: llama-cpp-python ────────────────────────────────────────────────

def _load_llama_cpp():
    global _llama_cpp_model
    if _llama_cpp_model is not None:
        return _llama_cpp_model
    if not LOCAL_GGUF or not os.path.isfile(LOCAL_GGUF):
        return None
    with _llama_cpp_lock:
        if _llama_cpp_model is not None:
            return _llama_cpp_model
        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF: {LOCAL_GGUF} (ctx={GGUF_CTX}, gpu={GGUF_GPU})")
            _llama_cpp_model = Llama(
                model_path=LOCAL_GGUF,
                n_ctx=GGUF_CTX,
                n_gpu_layers=GGUF_GPU,
                verbose=False,
                chat_format="chatml",
            )
            logger.info("llama-cpp-python loaded — 100% offline")
        except Exception as e:
            logger.warning(f"llama-cpp-python failed: {e}")
    return _llama_cpp_model


def _llama_cpp_call(messages: list) -> str:
    model = _load_llama_cpp()
    resp = model.create_chat_completion(
        messages=messages, temperature=TEMPERATURE, max_tokens=-1, stream=False,
    )
    return _strip_chinese_preamble(_strip_think(resp["choices"][0]["message"]["content"]))


def _llama_cpp_stream(messages: list) -> Iterator[str]:
    model = _load_llama_cpp()

    def _raw():
        for chunk in model.create_chat_completion(
            messages=messages, temperature=TEMPERATURE, max_tokens=-1, stream=True,
        ):
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                yield content

    yield from _filter_chinese_preamble_stream(_filter_think_stream(_raw()))


# ── Provider: Ollama ──────────────────────────────────────────────────────────
# Initial opts — overwritten by reload_config() on startup (keeps single source of truth).
_OLLAMA_OPTS: dict = {}


def _ollama_call(messages: list, num_predict: int | None = None,
                 num_ctx: int | None = None, timeout: int | None = None,
                 model: str | None = None) -> str:
    import time as _t
    _t0 = _t.perf_counter()
    opts = dict(_OLLAMA_OPTS)
    if num_predict is not None:
        opts["num_predict"] = num_predict
    if num_ctx is not None:
        opts["num_ctx"] = num_ctx
    payload = {"model": model or OLLAMA_MODEL, "messages": messages, "stream": False,
               "options": opts, "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m")}
    sess = _get_http_session()
    r = sess.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout or OLLAMA_TIMEOUT)
    r.raise_for_status()
    d = r.json()
    _wall = (_t.perf_counter() - _t0) * 1000
    _load  = d.get("load_duration", 0) // 1_000_000
    _pfill = d.get("prompt_eval_duration", 0) // 1_000_000
    _gen   = d.get("eval_duration", 0) // 1_000_000
    _toks  = d.get("eval_count", 0)
    logger.info(
        f"[OLLAMA] load={_load}ms pfill={_pfill}ms gen={_gen}ms "
        f"tokens={_toks} tok/s={round(_toks/max(_gen/1000,0.001),1)} wall={_wall:.0f}ms"
    )
    return _strip_chinese_preamble(_strip_think(d["message"]["content"]))


def _ollama_stream(messages: list, num_predict: int | None = None) -> Iterator[str]:
    opts = _OLLAMA_OPTS
    if num_predict is not None:
        opts = dict(_OLLAMA_OPTS)
        opts["num_predict"] = num_predict
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": True,
               "options": opts, "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m")}

    def _raw_stream():
        sess = _get_http_session()
        with sess.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

    yield from _filter_chinese_preamble_stream(_filter_think_stream(_raw_stream()))


def _ollama_vision_call(user_text: str, image_b64: str) -> str:
    payload = {
        "model":    VISION_MODEL,
        "messages": [{"role": "user", "content": user_text or "ماذا في هذه الصورة؟", "images": [image_b64]}],
        "stream":   False,
        "options":  {"temperature": TEMPERATURE, "num_predict": 1024},
    }
    sess = _get_http_session()
    r = sess.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ── Provider: Groq ────────────────────────────────────────────────────────────

def _groq_call(messages: list) -> str:
    from groq import Groq
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY غير موجود.")
    client = Groq(api_key=key)
    # max_tokens=None → API decides (full completion); only cap if explicitly set
    _max = NUM_PREDICT if NUM_PREDICT > 0 else None
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=TEMPERATURE,
        max_tokens=_max,
    )
    return _strip_chinese_preamble(_strip_think(resp.choices[0].message.content))


def _groq_stream(messages: list) -> Iterator[str]:
    from groq import Groq
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY غير موجود.")
    client = Groq(api_key=key)
    _max = NUM_PREDICT if NUM_PREDICT > 0 else None
    stream = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=TEMPERATURE, stream=True,
        max_tokens=_max,
    )

    def _raw():
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    yield from _filter_chinese_preamble_stream(_filter_think_stream(_raw()))


# ── Provider: OpenAI ──────────────────────────────────────────────────────────

def _openai_call(messages: list) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai مطلوب.")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY غير موجود.")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=int(os.getenv("NUM_PREDICT", "600")) if os.getenv("NUM_PREDICT", "-1") != "-1" else None,
    )
    return _strip_think(resp.choices[0].message.content or "")


def _openai_stream(messages: list) -> Iterator[str]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai مطلوب.")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY غير موجود.")
    client = OpenAI(api_key=key)
    stream = client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, temperature=TEMPERATURE, stream=True,
    )

    def _raw():
        for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                yield token

    yield from _filter_think_stream(_raw())


# ── Provider: Anthropic ───────────────────────────────────────────────────────

def _anthropic_call(messages: list) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic مطلوب.")
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY غير موجود.")
    client = anthropic.Anthropic(api_key=key)

    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs  = [m for m in messages if m["role"] != "system"]

    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=NUM_PREDICT if NUM_PREDICT > 0 else 4096,
        system=system_msg,
        messages=user_msgs,
        temperature=TEMPERATURE,
    )
    return _strip_think(resp.content[0].text)


def _anthropic_stream(messages: list) -> Iterator[str]:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic مطلوب.")
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY غير موجود.")
    client = anthropic.Anthropic(api_key=key)

    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs  = [m for m in messages if m["role"] != "system"]

    def _raw():
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=int(os.getenv("NUM_PREDICT", "1024")) if os.getenv("NUM_PREDICT", "-1") != "-1" else 1024,
            system=system_msg,
            messages=user_msgs,
            temperature=TEMPERATURE,
        ) as stream:
            for token in stream.text_stream:
                yield token

    yield from _filter_think_stream(_raw())


# ── Provider router ───────────────────────────────────────────────────────────

# Dispatch tables — keyed by LLM_PROVIDER value
_CALL_MAP: dict = {}   # populated after all _*_call functions are defined
_STREAM_MAP: dict = {}


def _init_dispatch():
    global _CALL_MAP, _STREAM_MAP
    _CALL_MAP   = {"groq": _groq_call, "openai": _openai_call, "anthropic": _anthropic_call}
    _STREAM_MAP = {"groq": _groq_stream, "openai": _openai_stream,
                   "anthropic": _anthropic_stream, "ollama": _ollama_stream}

_init_dispatch()


def _auto_call(messages: list, num_predict, num_ctx, timeout) -> str:
    if _load_llama_cpp() is not None:
        return _llama_cpp_call(messages)
    if _ping_ollama():
        return _ollama_call(messages, num_predict=num_predict, num_ctx=num_ctx, timeout=timeout)
    return _groq_call(messages)


def _auto_stream(messages: list, num_predict: int | None = None) -> Iterator[str]:
    if _load_llama_cpp() is not None:
        yield from _llama_cpp_stream(messages)
    elif _ping_ollama():
        yield from _ollama_stream(messages, num_predict=num_predict)
    else:
        yield from _groq_stream(messages)


def _route_call(messages: list, num_predict: int | None = None,
                num_ctx: int | None = None, timeout: int | None = None,
                model: str | None = None) -> str:
    p = LLM_PROVIDER.lower()
    if p == "llamacpp":
        if _load_llama_cpp() is None:
            raise RuntimeError(f"llama-cpp-python: ملف GGUF غير موجود: {LOCAL_GGUF}")
        return _llama_cpp_call(messages)
    if p == "ollama":
        return _ollama_call(messages, num_predict=num_predict, num_ctx=num_ctx,
                            timeout=timeout, model=model)
    if p in _CALL_MAP:
        return _CALL_MAP[p](messages)
    return _auto_call(messages, num_predict, num_ctx, timeout)


def _route_stream(messages: list, num_predict: int | None = None) -> Iterator[str]:
    p = LLM_PROVIDER.lower()
    if p == "llamacpp":
        if _load_llama_cpp() is None:
            raise RuntimeError(f"llama-cpp-python: ملف GGUF غير موجود: {LOCAL_GGUF}")
        yield from _llama_cpp_stream(messages)
        return
    if p in _STREAM_MAP:
        yield from _STREAM_MAP[p](messages)
        return
    yield from _auto_stream(messages, num_predict)


# ── Public API ────────────────────────────────────────────────────────────────

def chat(system: str, user: str, num_predict: int | None = None,
         num_ctx: int | None = None, timeout: int | None = None,
         model: str | None = None) -> str:
    """Single-turn chat. Returns complete response string.
    `model` overrides the default Ollama model (e.g. a smaller/faster model for
    quiz generation on CPU)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return _route_call(messages, num_predict=num_predict, num_ctx=num_ctx,
                       timeout=timeout, model=model)


def chat_stream(system: str, user: str, num_predict: int | None = None) -> Iterator[str]:
    """Streaming chat. Yields tokens one by one."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    yield from _route_stream(messages, num_predict=num_predict)


def chat_vision(user_text: str, image_b64: str) -> str:
    """Analyze an image. Requires Ollama + vision model."""
    if not _ping_ollama():
        raise RuntimeError(
            "تحليل الصور يتطلب Ollama مع نموذج مرئي — ollama pull llava:7b"
        )
    return _ollama_vision_call(user_text, image_b64)


def llm_available() -> bool:
    """Returns True if any LLM backend is reachable."""
    p = LLM_PROVIDER.lower()
    if p == "llamacpp":
        return _load_llama_cpp() is not None
    if p == "ollama":
        return _ping_ollama()
    if p in ("groq", "openai", "anthropic"):
        key_map = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        return bool(os.getenv(key_map[p]))
    # auto
    return (
        _load_llama_cpp() is not None
        or _ping_ollama()
        or bool(os.getenv("GROQ_API_KEY"))
        or bool(os.getenv("OPENAI_API_KEY"))
        or bool(os.getenv("ANTHROPIC_API_KEY"))
    )


def active_backend() -> str:
    """Returns name of the active backend for status reporting."""
    p = LLM_PROVIDER.lower()
    if p == "llamacpp":
        return f"llama-cpp [{os.path.basename(LOCAL_GGUF)}]"
    if p == "ollama":
        return f"ollama [{OLLAMA_MODEL}]"
    if p == "groq":
        return f"groq [{GROQ_MODEL}]"
    if p == "openai":
        return f"openai [{OPENAI_MODEL}]"
    if p == "anthropic":
        return f"anthropic [{ANTHROPIC_MODEL}]"
    # auto — report what's actually available
    if _load_llama_cpp() is not None:
        return f"llama-cpp [{os.path.basename(LOCAL_GGUF)}]"
    if _ping_ollama():
        return f"ollama [{OLLAMA_MODEL}]"
    if os.getenv("GROQ_API_KEY"):
        return f"groq [{GROQ_MODEL}]"
    if os.getenv("OPENAI_API_KEY"):
        return f"openai [{OPENAI_MODEL}]"
    if os.getenv("ANTHROPIC_API_KEY"):
        return f"anthropic [{ANTHROPIC_MODEL}]"
    return "none"
