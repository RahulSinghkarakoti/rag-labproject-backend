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

    def __init__(self, context: PipelineContext, run_manager: Any = None) -> None:
        self.context = context
        self.run_manager = run_manager

    def _format_output_for_db(self, step_name: str, result: Any) -> Any:
        try:
            from fastapi.encoders import jsonable_encoder

            from app.models.document import Document
        except ImportError:
            jsonable_encoder = lambda x: x  # noqa
        if isinstance(result, list):
            if not result:
                return []
            # If the output is a list of objects that look like documents, truncate them
            formatted_list = []
            for item in result:
                if hasattr(item, "id") and hasattr(item, "content"):
                    formatted_list.append(
                        {
                            "id": str(item.id),
                            "snippet": item.content[:200] + "..."
                            if item.content and len(item.content) > 200
                            else item.content,
                            "score": getattr(item, "score", None),
                            "rerank_score": getattr(item, "rerank_score", None),
                        }
                    )
                else:
                    try:
                        formatted_list.append(jsonable_encoder(item))
                    except Exception:
                        formatted_list.append(str(item))
            return formatted_list

        elif isinstance(result, dict):
            try:
                return jsonable_encoder(result)
            except Exception:
                return {"count": len(result)}
        else:
            try:
                return jsonable_encoder(result)
            except Exception:
                return str(result)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def stream_step(
        self,
        step_name: str,
        input_data: dict | None = None,
        config: dict | None = None,
    ):
        """Context manager for streaming steps that yield tokens."""
        logger.info(
            "pipeline_step step=%s state=waiting run_id=%s provider=%s model=%s",
            step_name,
            self.context.run_id,
            self.context.provider_name,
            self.context.model_name,
        )

        import asyncio

        step = None
        if self.run_manager:
            step = await self.run_manager.start_step(
                step_name, input_data=input_data, config=config
            )
        start = time.monotonic()
        
        result_container = {"response": ""}
        
        try:
            yield result_container
            elapsed = time.monotonic() - start
            logger.info(
                "pipeline_step step=%s state=completed elapsed=%.3fs run_id=%s",
                step_name,
                elapsed,
                self.context.run_id,
            )
            if self.run_manager and step:
                await self.run_manager.complete_step(
                    step,
                    output_data={"response": result_container["response"]},
                    metadata={"elapsed_seconds": round(elapsed, 4)},
                )
        except asyncio.CancelledError:
            elapsed = time.monotonic() - start
            logger.warning(
                "pipeline_step step=%s state=cancelled elapsed=%.3fs run_id=%s",
                step_name,
                elapsed,
                self.context.run_id,
            )
            if self.run_manager and step:
                await self.run_manager.fail_step(step, "Request cancelled by user")
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "pipeline_step step=%s state=failed elapsed=%.3fs run_id=%s",
                step_name,
                elapsed,
                self.context.run_id,
                exc_info=True,
            )
            if self.run_manager and step:
                await self.run_manager.fail_step(step, str(exc))
            raise

    async def run(
        self,
        step_name: str,
        coro: Coroutine[Any, Any, T] | Awaitable[T],
        input_data: dict | None = None,
        config: dict | None = None,
    ) -> T:
        """
        Await *coro* while logging timing and status for *step_name*, and persisting
        step lifecycle via *run_manager* if provided.
        """
        logger.info(
            "pipeline_step step=%s state=waiting run_id=%s provider=%s model=%s",
            step_name,
            self.context.run_id,
            self.context.provider_name,
            self.context.model_name,
        )

        if step_name == "memory":
            # Skip persisting memory step in DB to reduce noise
            start = time.monotonic()
            try:
                result = await coro
                elapsed = time.monotonic() - start
                logger.info(
                    "pipeline_step step=%s state=completed elapsed=%.3fs run_id=%s",
                    step_name,
                    elapsed,
                    self.context.run_id,
                )
                return result
            except Exception as exc:
                elapsed = time.monotonic() - start
                logger.error(
                    "pipeline_step step=%s state=failed elapsed=%.3fs run_id=%s",
                    step_name,
                    elapsed,
                    self.context.run_id,
                    exc_info=True,
                )
                raise

        import asyncio

        step = None
        if self.run_manager:
            step = await self.run_manager.start_step(
                step_name, input_data=input_data, config=config
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
            if self.run_manager and step:
                out = self._format_output_for_db(step_name, result)
                await self.run_manager.complete_step(
                    step,
                    output_data=out,
                    metadata={"elapsed_seconds": round(elapsed, 4)},
                )
            return result
        except asyncio.CancelledError as exc:
            elapsed = time.monotonic() - start
            logger.warning(
                "pipeline_step step=%s state=cancelled elapsed=%.3fs run_id=%s",
                step_name,
                elapsed,
                self.context.run_id,
            )
            if self.run_manager and step:
                await self.run_manager.fail_step(step, "Request cancelled by user")
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "pipeline_step step=%s state=failed elapsed=%.3fs run_id=%s",
                step_name,
                elapsed,
                self.context.run_id,
                exc_info=True,
            )
            if self.run_manager and step:
                await self.run_manager.fail_step(step, str(exc))
            raise
