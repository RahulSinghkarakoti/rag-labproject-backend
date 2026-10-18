"""
Context and prompt builders for the RAG generation stage.

Separates the concern of *assembling the context string* (from retrieved
documents + memory + transformed queries) from the rest of QueryPipeline,
making it easy to customise without touching pipeline orchestration logic.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Default generation system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT: str = (
    "Answer the user query using only the provided retrieved context and "
    "conversation memory. If the context is insufficient, say so clearly. "
    "Do not invent facts."
)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_rag_context(
    queries: list[str],
    documents: list[Any],
    memory: list[dict[str, Any]],
) -> str:
    """
    Assemble the RAG context block that is injected into the generation prompt.

    Args:
        queries:   Transformed query strings produced by the query-transformation step.
        documents: Reranked (or retrieved) document objects with a ``.content`` attribute.
        memory:    Prior conversation turns as ``{"role": ..., "content": ...}`` dicts.

    Returns:
        A formatted multi-section string ready to be embedded in the generation prompt.
    """
    query_context = "\n".join(f"- {q}" for q in queries) or "- None"

    document_context = "\n\n".join(
        f"[{idx}] {doc.content}"
        for idx, doc in enumerate(documents, start=1)
        if doc.content
    ) or "None"

    memory_context = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in memory
        if turn.get("content")
    ) or "None"

    return (
        f"Transformed queries:\n{query_context}\n\n"
        f"Conversation memory:\n{memory_context}\n\n"
        f"Retrieved context:\n{document_context}"
    )


# ---------------------------------------------------------------------------
# Generation prompt builder
# ---------------------------------------------------------------------------

def build_generation_prompt(
    query: str,
    context: str,
    system_prompt: str | None = None,
    citation_enforced: bool = False,
) -> str:
    """
    Build the final prompt that is sent to the LLM provider.

    Args:
        query:             The original (or primary) user query.
        context:           The assembled RAG context string from :func:`build_rag_context`.
        system_prompt:     Optional override for the system instructions.
                           Falls back to :data:`DEFAULT_SYSTEM_PROMPT` when *None*.
        citation_enforced: Whether to append strict inline citation rules.

    Returns:
        A ready-to-send prompt string.
    """
    instructions = system_prompt or DEFAULT_SYSTEM_PROMPT
    if citation_enforced:
        instructions += (
            " Strictly enforce citations: Every statement or claim based on retrieved context MUST include inline citations corresponding to document numbers (e.g. [1], [2])."
        )
    return f"{instructions}\n\nQuery: {query}\n\nContext: {context}"

