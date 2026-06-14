"""OriginCheckMiddleware — CSRF defense-in-depth behavior."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.middleware import OriginCheckMiddleware


def _make_client(origins):
    app = FastAPI()
    app.add_middleware(OriginCheckMiddleware, allowed_origins=origins)

    @app.post("/mutate")
    async def mutate():
        return {"ok": True}

    @app.get("/read")
    async def read():
        return {"ok": True}

    return TestClient(app)


def test_no_origin_header_passes():
    # Mobile Bearer clients and curl send no Origin — must be unaffected
    c = _make_client(["https://app.example.com"])
    assert c.post("/mutate").status_code == 200


def test_same_origin_passes():
    c = _make_client(["https://app.example.com"])
    r = c.post("/mutate", headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_allowed_listed_origin_passes():
    c = _make_client(["https://app.example.com"])
    r = c.post("/mutate", headers={"Origin": "https://app.example.com"})
    assert r.status_code == 200


def test_foreign_origin_blocked_on_mutation():
    c = _make_client(["https://app.example.com"])
    r = c.post("/mutate", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_foreign_origin_allowed_on_read():
    # Only state-changing methods are CSRF-relevant
    c = _make_client(["https://app.example.com"])
    r = c.get("/read", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200


def test_wildcard_disables_check():
    # Default config (CORS_ORIGINS=*) must not change behavior
    c = _make_client(["*"])
    r = c.post("/mutate", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200
