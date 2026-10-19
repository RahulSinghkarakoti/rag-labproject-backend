"""
Pipeline-level data structures for the RAG query pipeline.

``PipelineState`` carries the intermediate results produced during the
*prepare* phase (memory → transform → retrieve → rerank → context) so
they can be passed cleanly between pipeline methods and into ``_build_result``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """
    Immutable snapshot of all intermediate results produced by the
    RAG pipeline *prepare* phase.

    Attributes:
        memory:               Prior conversation turns loaded by MemoryManager.
        transformed_queries:  Queries produced by the query-transformation step.
        retrieved_documents:  Raw candidates returned by the retrieval workflow.
        reranked_documents:   Candidates after reranking (may equal retrieved_documents
                              when reranking is disabled).
        context:              Assembled RAG context string ready for the generation step.
    """

    memory: list[dict[str, Any]] = field(default_factory=list)
    transformed_queries: list[str] = field(default_factory=list)
    retrieved_documents: list[Any] = field(default_factory=list)
    reranked_documents: list[Any] = field(default_factory=list)
    context: str = ""
    is_greeting: bool = False
    greeting_response: str = ""

