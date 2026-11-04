"""Backward-compatibility module forwarding to app.api.documents."""

from app.api.documents import (
    delete_document,
    get_document,
    get_knowledge_base_documents,
    router,
    upload_multiple_files,
)

__all__ = [
    "router",
    "upload_multiple_files",
    "delete_document",
    "get_document",
    "get_knowledge_base_documents",
]
