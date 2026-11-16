"""
Context and prompt builders for the RAG generation stage.

Separates the concern of *assembling the context string* (from retrieved
documents + memory + transformed queries) from the rest of QueryPipeline,
making it easy to customise without touching pipeline orchestration logic.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Generation system prompts
# ---------------------------------------------------------------------------

STRICT_SYSTEM_PROMPT: str = (
    "Answer the user query using only the provided retrieved context and "
    "conversation memory. If the context is insufficient, say so clearly. "
    "Do not invent facts."
)

CONVERSATIONAL_SYSTEM_PROMPT: str = (
    "You are an insightful and helpful AI research partner in a conversational chat thread. "
    "Your goal is to provide rich, comprehensive, and engaging answers grounded in the provided documents and conversation history.\n\n"
    "Guidelines:\n"
    "1. Directly address the user's question first in a clear, natural, and helpful conversational tone.\n"
    "2. If the retrieved context contains relevant supplementary facts, insights, or nuance that the user didn't explicitly ask for but are helpful, share them naturally.\n"
    "3. Base all factual claims strictly on the provided retrieved context and conversation memory. Do not invent external facts or speculate beyond the sources.\n"
    "4. If the retrieved context is insufficient or silent on certain aspects, politely clarify what is known from the documents and what is not covered."
)

FOLLOWUP_SUGGESTIONS_PROMPT: str = (
    "\n\nAt the end of your response, provide a brief 'Suggested Follow-ups' section "
    "with 2-3 logical next questions or exploration topics the user might find valuable based on the context."
)

DEFAULT_SYSTEM_PROMPT: str = CONVERSATIONAL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_rag_context(
    queries: list[str],
    documents: list[Any],
    memory: list[dict[str, Any]],
    summary: str | None = None,
) -> str:
    """
    Assemble the RAG context block that is injected into the generation prompt.

    Args:
        queries:   Transformed query strings produced by the query-transformation step.
        documents: Reranked (or retrieved) document objects with a ``.content`` attribute.
        memory:    Prior conversation turns as ``{"role": ..., "content": ...}`` dicts.
        summary:   Optional progressive summary of earlier compacted conversation turns.

    Returns:
        A formatted multi-section string ready to be embedded in the generation prompt.
    """
    query_context = "\n".join(f"- {q}" for q in queries) or "- None"

    document_context = "\n\n".join(
        f"[{doc.document.name if hasattr(doc, 'document') and doc.document else getattr(doc, 'name', idx)}] {doc.content}"
        for idx, doc in enumerate(documents, start=1)
        if doc.content
    ) or "None"

    memory_context = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in memory
        if turn.get("content")
    ) or "None"

    sections = [f"Transformed queries:\n{query_context}"]
    if summary and summary.strip():
        sections.append(f"Conversation summary (older turns):\n{summary.strip()}")
    sections.append(f"Recent conversation memory:\n{memory_context}")
    sections.append(f"Retrieved context:\n{document_context}")

    return "\n\n".join(sections)


def estimate_context_tokens(text: str) -> int:
    """
    Heuristic estimation of token count for prompt/context text (~4 chars per token).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_context_metrics(
    prompt: str,
    is_summarized: bool = False,
    model_limit: int = 8192,
) -> dict[str, Any]:
    """
    Calculate context usage metrics and threshold warnings.
    """
    tokens = estimate_context_tokens(prompt)
    percentage = min(100, int((tokens / model_limit) * 100)) if model_limit > 0 else 0
    warning = None
    if percentage >= 80:
        warning = "near_limit"
    return {
        "tokens": tokens,
        "limit": model_limit,
        "percentage": percentage,
        "summarized": is_summarized,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Generation prompt builder
# ---------------------------------------------------------------------------

def build_generation_prompt(
    query: str,
    context: str,
    system_prompt: str | None = None,
    citation_enforced: bool = False,
    chat_mode: str = "conversational",
    suggest_followups: bool = True,
) -> str:
    """
    Build the final prompt that is sent to the LLM provider.

    Args:
        query:              The original (or primary) user query.
        context:            The assembled RAG context string from :func:`build_rag_context`.
        system_prompt:      Optional override for the system instructions.
                            Falls back to mode default when *None*.
        citation_enforced:  Whether to append strict inline citation rules.
        chat_mode:          Generation style ("conversational" or "strict").
        suggest_followups:  Whether to include follow-up suggestion prompt in conversational mode.

    Returns:
        A ready-to-send prompt string.
    """
    if system_prompt:
        instructions = system_prompt
    elif chat_mode == "strict":
        instructions = STRICT_SYSTEM_PROMPT
    else:
        instructions = CONVERSATIONAL_SYSTEM_PROMPT

    if suggest_followups and chat_mode != "strict":
        instructions += FOLLOWUP_SUGGESTIONS_PROMPT

    if citation_enforced:
        instructions += (
            " Strictly enforce citations: Every statement or claim based on retrieved context MUST include inline citations corresponding to document names (e.g. [document_name.pdf], [other_document.txt])."
        )
    return f"{instructions}\n\nQuery: {query}\n\nContext: {context}"

