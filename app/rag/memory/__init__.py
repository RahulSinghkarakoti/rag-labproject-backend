"""
Conversation memory management for the RAG pipeline.

Exports:
    MemoryManager  — loads prior thread messages from the database.
    build_rag_context — assembles the RAG context block for the generation prompt.
    build_generation_prompt — builds the final LLM prompt from query + context.
"""
from app.rag.memory.manager import MemoryManager
from app.rag.memory.context import build_rag_context, build_generation_prompt

__all__ = [
    "MemoryManager",
    "build_rag_context",
    "build_generation_prompt",
]
