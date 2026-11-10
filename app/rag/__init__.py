"""RAG module with PEP 562 lazy loading."""
from typing import Any

__all__ = ["QueryEngine"]


def __getattr__(name: str) -> Any:
    if name == "QueryEngine":
        from app.rag.engine import QueryEngine
        return QueryEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
