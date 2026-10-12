# app/rag/config.py
from app.schemas.thread import (
    QueryTransformationConfig,
    RetrievalConfig,
    RerankingConfig,
    GenerationConfig,
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
