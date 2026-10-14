"""
RAG pipeline orchestration layer.

Exports:
    QueryPipeline      — full query RAG pipeline (transform → retrieve → rerank → generate).
    IngestionPipeline  — document ingestion pipeline (load → chunk → embed → store).
    PipelineContext    — immutable run-scoped identity record.
    PipelineStepExecutor — wraps pipeline stages with timing and structured logging.
    PipelineState      — typed snapshot of prepare-phase intermediate results.
"""
from app.rag.pipeline.query import QueryPipeline
from app.rag.pipeline.ingestion import IngestionPipeline
from app.rag.pipeline.context import PipelineContext
from app.rag.pipeline.executor import PipelineStepExecutor
from app.rag.pipeline.schemas import PipelineState

__all__ = [
    "QueryPipeline",
    "IngestionPipeline",
    "PipelineContext",
    "PipelineStepExecutor",
    "PipelineState",
]
