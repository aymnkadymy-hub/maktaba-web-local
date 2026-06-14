import time
import uuid
import logging
import os
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("api.middleware")

_COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"


class SecurityHeadersMiddleware:
    """Injects security headers on every HTTP response."""

    # Content-Security-Policy: allow same-origin for scripts/styles,
    # blob: for worker streams, data: for images.
    _CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "   # inline JS in chat.html
        "style-src 'self' 'unsafe-inline'; "    # inline styles
        "img-src 'self' data: blob:; "
        "connect-src 'self' blob:; "
        "font-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-content-type-options",   b"nosniff"),
                    (b"x-frame-options",           b"DENY"),
                    (b"referrer-policy",           b"strict-origin-when-cross-origin"),
                    (b"permissions-policy",        b"camera=(), microphone=(), geolocation=()"),
                    (b"content-security-policy",   self._CSP.encode()),
                ]
                if _COOKIE_SECURE:
                    headers.append((b"strict-transport-security",
                                    b"max-age=31536000; includeSubDomains"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RequestLoggingMiddleware:
    """Pure ASGI middleware — zero body-buffering overhead vs BaseHTTPMiddleware."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:8]
        request = Request(scope, receive)
        logger.info(f"[{request_id}] ← {request.method} {request.url.path}")

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)
        elapsed = (time.perf_counter() - start) * 1000
        symbol  = "✓" if status_code < 400 else "✗"
        logger.info(f"[{request_id}] {symbol} {status_code} {elapsed:.1f}ms")


class RateLimitMiddleware:
    """Pure ASGI rate limiter — no body buffering."""

    _MAX_STORE_SIZE = 5000   # cap IPs tracked; evict oldest on overflow

    def __init__(self, app: ASGIApp, max_per_second: int = 10):
        self.app        = app
        self._max       = max_per_second
        self._store:    dict[str, list[float]] = {}
        self._last_seen: dict[str, float]       = {}
        self._req_count = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Static files and favicon are never rate-limited — browsers batch-load
        # many assets in parallel on page load and would otherwise hit the limit.
        path = scope.get("path", "")
        if path.startswith("/static/") or path == "/favicon.ico":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        ip = client[0] if client else None
        if ip is None:
            await self.app(scope, receive, send)
            return

        now    = time.monotonic()
        window = [t for t in self._store.get(ip, []) if now - t < 1.0]
        if not window:
            self._store.pop(ip, None)
            self._last_seen.pop(ip, None)

        if len(window) >= self._max:
            response = JSONResponse(
                status_code=429,
                content={"detail": "كثرة الطلبات، انتظر لحظة يا غالي"},
            )
            await response(scope, receive, send)
            return

        window.append(now)
        self._store[ip]     = window
        self._last_seen[ip] = now

        # Periodic cleanup every 500 requests — evict IPs with no recent activity
        self._req_count += 1
        if self._req_count % 500 == 0:
            cutoff  = now - 60
            stale   = [k for k, v in self._last_seen.items() if v < cutoff]
            for k in stale:
                self._store.pop(k, None)
                self._last_seen.pop(k, None)
            # Hard cap: if still too large, evict oldest by last_seen
            if len(self._store) > self._MAX_STORE_SIZE:
                excess = sorted(self._last_seen.items(), key=lambda x: x[1])
                for k, _ in excess[:len(self._store) - self._MAX_STORE_SIZE]:
                    self._store.pop(k, None)
                    self._last_seen.pop(k, None)

        await self.app(scope, receive, send)


class OriginCheckMiddleware:
    """CSRF defense-in-depth on top of SameSite=lax cookies.

    Browsers attach an Origin header to every state-changing request; when
    one is present it must match the request's own Host (same-origin) or an
    explicitly configured origin. Requests without an Origin header
    (mobile Bearer clients, curl) are unaffected — CSRF is a browser-only
    attack. With CORS_ORIGINS=* (the default) the check is disabled, so
    enforcement only activates once the operator pins explicit origins.
    """

    _STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, app: ASGIApp, allowed_origins=None):
        self.app = app
        origins = set(allowed_origins or [])
        self.allow_all = "*" in origins
        self.allowed = {o.lower().rstrip("/") for o in origins if o != "*"}

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (scope["type"] != "http" or self.allow_all
                or scope.get("method") not in self._STATE_CHANGING):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        origin = headers.get("origin", "")
        if origin:
            from urllib.parse import urlsplit
            origin_host = urlsplit(origin).netloc.lower()
            same_origin = origin_host == headers.get("host", "").lower()
            if not same_origin and origin.lower().rstrip("/") not in self.allowed:
                logger.warning(f"Blocked cross-origin {scope.get('method')} "
                               f"{scope.get('path')} from origin={origin}")
                resp = JSONResponse({"detail": "Origin غير مسموح"}, status_code=403)
                await resp(scope, receive, send)
                return

        await self.app(scope, receive, send)
