import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message
from app.models.run import ModelCall, RetrievalResult, RetrievalRun, Run, RunStep

logger = logging.getLogger(__name__)


class RunManager:
    """Manages lifecycle, stage auto-commits, and telemetry persistence for RAG runs."""

    def __init__(self, db: AsyncSession, run: Run | None = None) -> None:
        self.db = db
        self.run = run
        self._step_sequence = 0
        self.current_retrieval_run_id: uuid.UUID | None = None
        logger.debug(
            "[RunManager] Initialized | active_run_id=%s",
            getattr(run, "id", None),
        )

    async def start_run(
        self,
        *,
        thread_id: uuid.UUID,
        input_message_id: uuid.UUID | None = None,
        pipeline_version_id: uuid.UUID | None = None,
        run_type: str = "chat",
        metadata: dict | None = None,
    ) -> Run:
        logger.info(
            "[RunManager] Starting run | thread_id=%s | run_type=%s | "
            "input_message_id=%s | pipeline_version_id=%s",
            thread_id,
            run_type,
            input_message_id,
            pipeline_version_id,
        )
        self.run = Run(
            thread_id=thread_id,
            input_message_id=input_message_id,
            pipeline_version_id=pipeline_version_id,
            run_type=run_type,
            status="running",
            started_at=datetime.now(UTC),
            metadata_=metadata or {},
        )
        self.db.add(self.run)
        logger.debug("[RunManager] Committing new run to database")
        await self.db.commit()
        await self.db.refresh(self.run)
        logger.info(
            "[RunManager] Run started | run_id=%s | status=%s",
            self.run.id,
            self.run.status,
        )
        return self.run

    async def start_step(self,step_type: str,*,input_data: dict | None = None,metadata: dict | None = None,config: dict | None = None) -> RunStep:
        if not self.run:
            logger.error(
                "[RunManager] Cannot start step '%s': no active run",
                step_type,
            )
            raise RuntimeError("RunManager has no active run")
        self._step_sequence += 1
        logger.info(
            "[RunManager] Starting step | run_id=%s | sequence=%s | step_type=%s",
            self.run.id,
            self._step_sequence,
            step_type,
        )
        logger.debug(
            "[RunManager] Step input | step_type=%s | input=%s | metadata=%s",
            step_type,
            input_data,
            metadata,
        )
        step = RunStep(
            run_id=self.run.id,
            step_type=step_type,
            sequence=self._step_sequence,
            status="running",
            started_at=datetime.now(UTC),
            input=input_data,
            config=config,
            metadata_=metadata or {},
        )
        self.db.add(step)
        await self.db.commit()
        await self.db.refresh(step)
        logger.info(
            "[RunManager] Step started | step_id=%s | sequence=%s | status=%s",
            step.id,
            step.sequence,
            step.status,
        )
        return step

    async def complete_step(
        self,
        step: RunStep,
        *,
        output_data: dict | None = None,
        metadata: dict | None = None,
    ) -> RunStep:
        logger.info(
            "[RunManager] Completing step | step_id=%s | type=%s | sequence=%s",
            step.id,
            step.step_type,
            step.sequence,
        )
        logger.debug(
            "[RunManager] Step output | step_id=%s | output=%s | metadata=%s",
            step.id,
            output_data,
            metadata,
        )
        step.status = "completed"
        step.completed_at = datetime.now(UTC)
        if output_data is not None:
            step.output = output_data
        if metadata:
            step.metadata_ = {**(step.metadata_ or {}), **metadata}
        await self.db.commit()
        logger.info(
            "[RunManager] Step completed | step_id=%s | status=%s",
            step.id,
            step.status,
        )
        return step

    async def fail_step(
        self,
        step: RunStep,
        error: Exception | dict | str,
    ) -> RunStep:
        logger.error(
            "[RunManager] Step failed | step_id=%s | type=%s | error=%s",
            step.id,
            step.step_type,
            error,
            exc_info=error if isinstance(error, Exception) else None,
        )
        step.status = "failed"
        step.completed_at = datetime.now(UTC)
        step.error = (
            {"message": str(error)} if isinstance(error, (Exception, str)) else error
        )
        await self.db.commit()
        logger.info("[RunManager] Failed step persisted | step_id=%s", step.id)
        return step

    async def record_retrieval(
        self,
        *,
        query: str,
        retriever_type: str,
        top_k: int,
        documents: list[Any],
    ) -> RetrievalRun:
        if not self.run:
            logger.error("[RunManager] Cannot record retrieval: no active run")
            raise RuntimeError("RunManager has no active run")
        logger.info(
            "[RunManager] Recording retrieval | run_id=%s | retriever=%s | "
            "top_k=%s | document_count=%s",
            self.run.id,
            retriever_type,
            top_k,
            len(documents),
        )
        logger.debug("[RunManager] Retrieval query: %s", query)
        retrieval_run = RetrievalRun(
            run_id=self.run.id,
            query=query,
            retriever_type=retriever_type,
            top_k=top_k,
        )
        self.db.add(retrieval_run)
        await self.db.flush()
        self.current_retrieval_run_id = retrieval_run.id
        linked_count = 0
        skipped_count = 0
        # Existing behavior: only UUID-compatible chunk IDs are linked.
        for rank, doc in enumerate(documents, start=1):
            # Try multiple fallback attributes for robust ID extraction
            raw_id = getattr(doc, "id", None) or getattr(doc, "chunk_id", None)
            logger.debug(
                "[RunManager] Retrieval result | rank=%s | chunk_id=%s | score=%s",
                rank,
                raw_id,
                getattr(doc, "score", None),
            )
            if raw_id:
                try:
                    chunk_uuid = uuid.UUID(str(raw_id))
                except (ValueError, TypeError):
                    logger.warning(
                        "[RunManager] Invalid UUID format for chunk ID: %s", raw_id
                    )
                    chunk_uuid = None
                if chunk_uuid:
                    result = RetrievalResult(
                        retrieval_run_id=retrieval_run.id,
                        chunk_id=chunk_uuid,
                        rank=rank,
                        score=getattr(doc, "score", None),
                        metadata_={"preview": getattr(doc, "content", "")[:200]}
                        if hasattr(doc, "content")
                        else {},
                    )
                    self.db.add(result)
                    linked_count += 1
                else:
                    skipped_count += 1
                    logger.debug(
                        "[RunManager] Skipping non-UUID chunk ID | rank=%s | id=%s",
                        rank,
                        raw_id,
                    )
            else:
                skipped_count += 1
                logger.debug(
                    "[RunManager] Skipping document without ID | rank=%s",
                    rank,
                )
        await self.db.commit()
        await self.db.refresh(retrieval_run)
        logger.info(
            "[RunManager] Retrieval recorded | retrieval_run_id=%s | "
            "linked_results=%s | skipped_results=%s",
            retrieval_run.id,
            linked_count,
            skipped_count,
        )
        return retrieval_run

    async def record_rerank(
        self,
        documents: list[Any],
    ) -> None:
        if not self.current_retrieval_run_id:
            logger.warning(
                "[RunManager] Cannot record rerank: no current retrieval run"
            )
            return
        logger.info(
            "[RunManager] Recording rerank | retrieval_run_id=%s | document_count=%s",
            self.current_retrieval_run_id,
            len(documents),
        )
        existing_results = (
            await self.db.scalars(
                select(RetrievalResult).where(
                    RetrievalResult.retrieval_run_id == self.current_retrieval_run_id
                )
            )
        ).all()
        results_by_chunk_id = {
            res.chunk_id: res for res in existing_results if res.chunk_id is not None
        }

        updated_count = 0
        unmatched_count = 0
        for rank, doc in enumerate(documents, start=1):
            raw_id = getattr(doc, "id", None) or getattr(doc, "chunk_id", None)
            rerank_score = getattr(doc, "rerank_score", None)
            logger.debug(
                "[RunManager] Rerank result | rank=%s | chunk_id=%s | rerank_score=%s",
                rank,
                raw_id,
                rerank_score,
            )
            if raw_id and rerank_score is not None:
                try:
                    chunk_uuid = uuid.UUID(str(raw_id))
                except (ValueError, TypeError):
                    logger.debug(
                        "[RunManager] Skipping non-UUID rerank ID | id=%s",
                        raw_id,
                    )
                    continue
                res = results_by_chunk_id.get(chunk_uuid)
                if res:
                    res.rerank_score = float(rerank_score)
                    res.metadata_ = {
                        **(res.metadata_ or {}),
                        "rerank_rank": rank,
                    }
                    updated_count += 1
                else:
                    unmatched_count += 1
                    logger.debug(
                        "[RunManager] No retrieval result found for rerank "
                        "chunk | chunk_id=%s",
                        chunk_uuid,
                    )
        await self.db.commit()
        logger.info(
            "[RunManager] Rerank recorded | updated=%s | unmatched=%s",
            updated_count,
            unmatched_count,
        )

    async def record_model_call(
        self,
        *,
        provider: str,
        model: str,
        request: dict | None = None,
        response: dict | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: int | None = None,
        status: str = "completed",
        error: dict | None = None,
    ) -> ModelCall:
        if not self.run:
            logger.error("[RunManager] Cannot record model call: no active run")
            raise RuntimeError("RunManager has no active run")
        logger.info(
            "[RunManager] Recording model call | run_id=%s | provider=%s | "
            "model=%s | status=%s | latency_ms=%s",
            self.run.id,
            provider,
            model,
            status,
            latency_ms,
        )
        logger.debug(
            "[RunManager] Model usage | input_tokens=%s | output_tokens=%s | "
            "total_tokens=%s",
            input_tokens,
            output_tokens,
            total_tokens,
        )
        if error:
            logger.warning(
                "[RunManager] Model call error | provider=%s | model=%s | error=%s",
                provider,
                model,
                error,
            )
        model_call = ModelCall(
            run_id=self.run.id,
            provider=provider,
            model=model,
            request=request,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )
        self.db.add(model_call)
        await self.db.commit()
        logger.info("[RunManager] Model call persisted | run_id=%s", self.run.id)
        return model_call

    async def complete_run(
        self,
        *,
        output_message: Message | None = None,
        metadata: dict | None = None,
    ) -> Run:
        if not self.run:
            logger.error("[RunManager] Cannot complete run: no active run")
            raise RuntimeError("RunManager has no active run")
        logger.info("[RunManager] Completing run | run_id=%s", self.run.id)
        if output_message:
            self.run.output_message_id = output_message.id
        self.run.status = "completed"
        self.run.completed_at = datetime.now(UTC)
        if metadata:
            self.run.metadata_ = {**(self.run.metadata_ or {}), **metadata}
        await self.db.commit()
        logger.info(
            "[RunManager] Run completed | run_id=%s | output_message_id=%s",
            self.run.id,
            self.run.output_message_id,
        )
        return self.run

    async def cancel_run(
        self,
        *,
        cancellation_message: Message | None = None,
        error: dict | None = None,
    ) -> Run:
        if not self.run:
            logger.error("[RunManager] Cannot cancel run: no active run")
            raise RuntimeError("RunManager has no active run")
        logger.warning("[RunManager] Cancelling run | run_id=%s", self.run.id)
        if cancellation_message:
            self.run.output_message_id = cancellation_message.id
        self.run.status = "cancelled"
        self.run.completed_at = datetime.now(UTC)
        self.run.error = error or {
            "code": "CANCELLED",
            "message": "User stopped the retrieval or generation.",
        }
        await self.db.commit()
        logger.info("[RunManager] Run cancelled | run_id=%s", self.run.id)
        return self.run

    async def fail_run(
        self,
        error: dict | Exception | str,
    ) -> Run:
        if not self.run:
            logger.error("[RunManager] Cannot fail run: no active run")
            raise RuntimeError("RunManager has no active run")
        logger.error(
            "[RunManager] Run failed | run_id=%s | error=%s",
            self.run.id,
            error,
            exc_info=error if isinstance(error, Exception) else None,
        )
        self.run.status = "failed"
        self.run.completed_at = datetime.now(UTC)
        if isinstance(error, dict):
            self.run.error = error
        elif isinstance(error, Exception):
            self.run.error = {
                "code": getattr(error, "code", "RUN_ERROR"),
                "message": getattr(error, "message", str(error)),
                "details": getattr(error, "details", None),
            }
        else:
            self.run.error = {"code": "RUN_ERROR", "message": str(error)}
        await self.db.commit()
        logger.info("[RunManager] Failed run persisted | run_id=%s", self.run.id)
        return self.run
