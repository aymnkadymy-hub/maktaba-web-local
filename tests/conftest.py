"""Test bootstrap — must run before any backend import.

Points the auth and chat databases at a throwaway temp directory so tests
never touch the real auth.db / chat.db.
"""
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_TMP = tempfile.mkdtemp(prefix="maktaba_tests_")
os.environ["AUTH_DB"]      = os.path.join(_TMP, "auth.db")
os.environ["CHAT_DB_PATH"] = os.path.join(_TMP, "chat.db")
