"""Shared singletons — imported by all core/api modules to avoid circular imports."""
import os

from backend.database.vector_db import vector_store  # noqa: F401 (re-exported)
from backend.rag.hybrid_retriever import HybridRetriever

hybrid_retriever: HybridRetriever = HybridRetriever(vector_store)

BOOKS_DIR: str = os.getenv(
    "BOOKS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "books"),
)
ENABLE_FUNCTION_CALLING: bool = os.getenv("ENABLE_FUNCTION_CALLING", "true").lower() == "true"
_ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
