# app/rag/config.py
from app.schemas.thread import (
    GenerationConfig,
    QueryTransformationConfig,
    RerankingConfig,
    RetrievalConfig,
    ThreadConfig,
)

# Semantic aliases for the RAG query pipeline
QueryConfig = QueryTransformationConfig
RAGConfig = ThreadConfig

__all__ = [
    "QueryTransformationConfig",
    "QueryConfig",
    "RetrievalConfig",
    "RerankingConfig",
    "GenerationConfig",
    "ThreadConfig",
    "RAGConfig",
]
