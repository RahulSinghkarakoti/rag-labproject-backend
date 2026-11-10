"""RAG pipeline orchestration layer with PEP 562 lazy exports."""
from typing import Any

__all__ = [
    "QueryPipeline",
    "IngestionPipeline",
    "PipelineContext",
    "PipelineStepExecutor",
    "PipelineState",
]

_PIPELINE_MAP = {
    "QueryPipeline": ("app.rag.pipeline.query", "QueryPipeline"),
    "IngestionPipeline": ("app.rag.pipeline.ingestion", "IngestionPipeline"),
    "PipelineContext": ("app.rag.pipeline.context", "PipelineContext"),
    "PipelineStepExecutor": ("app.rag.pipeline.executor", "PipelineStepExecutor"),
    "PipelineState": ("app.rag.pipeline.schemas", "PipelineState"),
}


def __getattr__(name: str) -> Any:
    if name in _PIPELINE_MAP:
        module_path, class_name = _PIPELINE_MAP[name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
