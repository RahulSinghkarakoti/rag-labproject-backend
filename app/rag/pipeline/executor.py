"""
Pipeline step executor with structured logging and timing.

``PipelineStepExecutor`` wraps every async pipeline stage (memory, transform,
retrieval, reranking, generation) in a uniform try/finally block that emits
structured log lines and elapsed-time measurements without the pipeline
orchestrator having to repeat this boilerplate.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Coroutine, TypeVar

from app.rag.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PipelineStepExecutor:
    """
    Executes a single awaitable pipeline step with consistent structured logging.

    Each call to :meth:`run` emits two log lines:
    * ``state=waiting``  — before the coroutine starts
    * ``state=completed`` or ``state=failed`` — after it resolves

    Every log line includes the step name, elapsed time, and the run-scoped
    metadata from :class:`~app.rag.pipeline.context.PipelineContext`.

    Usage::

        executor = PipelineStepExecutor(pipeline_context)
        memory = await executor.run("memory", memory_manager.load(...))
        queries = await executor.run("transformation", process_query(...))
    """

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    async def run(self, step_name: str, coro: Coroutine[Any, Any, T] | Awaitable[T]) -> T:
        """
        Await *coro* while logging timing and status for *step_name*.

        Args:
            step_name: Short label used in log output (e.g. ``"memory"``, ``"retrieval"``).
            coro:      The coroutine to execute.

        Returns:
            Whatever *coro* returns.

        Raises:
            Re-raises any exception from *coro* after logging ``state=failed``.
        """
        logger.info(
            "pipeline_step step=%s state=waiting run_id=%s provider=%s model=%s",
            step_name,
            self.context.run_id,
            self.context.provider_name,
            self.context.model_name,
        )
        start = time.monotonic()
        try:
            result = await coro
            elapsed = time.monotonic() - start
            count_str = (
                f" count={len(result)}"  # type: ignore[arg-type]
                if hasattr(result, "__len__")
                else ""
            )
            logger.info(
                "pipeline_step step=%s state=completed elapsed=%.3fs%s run_id=%s",
                step_name,
                elapsed,
                count_str,
                self.context.run_id,
            )
            return result
        except Exception:
            elapsed = time.monotonic() - start
            logger.error(
                "pipeline_step step=%s state=failed elapsed=%.3fs run_id=%s",
                step_name,
                elapsed,
                self.context.run_id,
                exc_info=True,
            )
            raise
