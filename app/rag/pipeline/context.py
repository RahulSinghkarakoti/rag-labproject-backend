"""
Runtime context for a single RAG pipeline execution.

``PipelineContext`` is a lightweight, read-only record that identifies
the current run and is threaded through the pipeline infrastructure
(e.g. ``PipelineStepExecutor``) so that every log line and trace event
carries consistent run-scoped metadata without relying on shared state.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineContext:
    """
    Immutable identity record for one pipeline execution.

    Attributes:
        run_id:        Unique ID of the active :class:`~app.models.run.Run`.
        thread_id:     ID of the parent :class:`~app.models.thread.Thread`.
        message_id:    ID of the triggering user :class:`~app.models.message.Message`.
        provider_name: Human-readable name of the LLM provider (e.g. ``"groq"``).
        model_name:    Model identifier used for generation (e.g. ``"llama3-8b-8192"``).
    """

    run_id: str
    thread_id: str
    message_id: str
    provider_name: str
    model_name: str
