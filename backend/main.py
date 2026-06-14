"""
نقطة الدخول الوحيدة للـ backend.
server_backend.py هو التطبيق الكامل؛ هذا الملف يشغّله فقط.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.server_backend import app  # noqa: F401 — يُستخدم بـ uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[os.path.join(PROJECT_ROOT, "backend")],
    )
