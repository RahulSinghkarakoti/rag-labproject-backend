"""Services package with PEP 562 lazy loading to avoid eager import cascade."""
from typing import Any

__all__ = [
    "AuthService",
    "ChatService",
    "DocumentService",
    "KnowledgeBaseService",
    "PipelineService",
    "ProjectService",
    "RunService",
]

_SERVICE_MAP = {
    "AuthService": ("app.services.auth_service", "AuthService"),
    "ChatService": ("app.services.chat_service", "ChatService"),
    "DocumentService": ("app.services.document_service", "DocumentService"),
    "KnowledgeBaseService": ("app.services.knowledge_base_service", "KnowledgeBaseService"),
    "PipelineService": ("app.services.pipeline_service", "PipelineService"),
    "ProjectService": ("app.services.project_service", "ProjectService"),
    "RunService": ("app.services.run_service", "RunService"),
}


def __getattr__(name: str) -> Any:
    if name in _SERVICE_MAP:
        module_path, class_name = _SERVICE_MAP[name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
