from typing import Any
import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppException, sanitize_model_error
from app.models.pipeline import PipelineVersion
from app.models.message import Message
from app.models.run import ModelCall, Run, RunStep
from app.models.thread import Thread
from app.rag.generation.base import BaseGenerationProvider
from app.rag.generation.factory import ProviderModelFactory, provider_model_factory
from app.rag.pipeline.query import QueryPipeline


class QueryEngine:
    def __init__(
        self,
        db: AsyncSession,
        provider_factory: ProviderModelFactory = provider_model_factory,
        run_manager: Any = None,
    ):
        self.db = db
        self.provider_factory = provider_factory
        self.run_manager = run_manager

    async def run(
        self,
        *,
        run: Run,
        thread: Thread,
        message: Message,
        pipeline_version: PipelineVersion,
    ):
        """
        Execute one RAG/chat run using the pipeline version config.
        Tracks run status (running -> completed / failed / cancelled).
        """
        config = pipeline_version.config or {}
        now = datetime.now(timezone.utc)
        run.started_at = now
        run.status = "running"

        try:
            provider: BaseGenerationProvider = self.provider_factory.create(config)
            pipeline = QueryPipeline(
                db=self.db,
                config=config,
                run=run,
                run_manager=self.run_manager,
                thread=thread,
                message=message,
                provider=provider,
            )

            result = await pipeline.execute()

            await self._persist_run_result(
                run=run,
                message=message,
                result=result,
                config=config,
                provider=provider,
            )

            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            await self.db.commit()
            return result

        except (asyncio.CancelledError, GeneratorExit):
            run.status = "cancelled"
            run.completed_at = datetime.now(timezone.utc)
            run.error = {
                "code": "CANCELLED",
                "message": "User stopped the retrieval or generation.",
            }
            await self.db.commit()
            raise

        except Exception as exc:
            # Handle and sanitize backend model provider errors safely
            if isinstance(exc, AppException):
                safe_exc = exc
            else:
                gen_cfg = config.get("generation", {})
                provider_name = gen_cfg.get("provider", "unknown")
                model_name = gen_cfg.get("llm_model") or gen_cfg.get("model", "unknown")
                safe_exc = sanitize_model_error(
                    exc, provider=provider_name, model=model_name
                )

            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.error = {
                "code": getattr(safe_exc, "code", "RUN_ERROR"),
                "message": getattr(safe_exc, "message", str(safe_exc)),
                "details": getattr(safe_exc, "details", None),
            }
            await self.db.commit()
            raise safe_exc

    async def run_stream(
        self,
        *,
        run: Run,
        thread: Thread,
        message: Message,
        pipeline_version: PipelineVersion,
    ):
        """Stream generation tokens and handle cancellation/disconnection gracefully."""
        config = pipeline_version.config or {}
        run.started_at = datetime.now(timezone.utc)
        run.status = "running"
        accumulated_tokens: list[str] = []

        try:
            provider: BaseGenerationProvider = self.provider_factory.create(config)
            pipeline = QueryPipeline(
                db=self.db,
                config=config,
                run=run,
                run_manager=self.run_manager,
                thread=thread,
                message=message,
                provider=provider,
            )
            async for event in pipeline.execute_stream():
                if event["type"] == "token":
                    accumulated_tokens.append(event["content"])
                    yield event
                    continue

                result = event["result"]
                await self._persist_run_result(
                run=run,
                message=message,
                    result=result,
                    config=config,
                    provider=provider,
                )
                run.status = "completed"
                run.completed_at = datetime.now(timezone.utc)
                await self.db.commit()
                yield {
                    "type": "complete",
                    "run_id": str(run.id),
                    "output_message_id": result["output_message_id"],
                    "response": result["response"],
                }
        except (asyncio.CancelledError, GeneratorExit, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "CancelledError" not in str(exc):
                raise exc
            
            from app.database.database import AsyncSessionLocal
            import asyncio
            
            async def _mark_cancelled(r_id, m_id, thread_id, seq, p_text):
                try:
                    async with AsyncSessionLocal() as session:
                        from app.models.run import Run
                        from app.models.message import Message
                        c_run = await session.get(Run, r_id)
                        if not c_run: return
                        c_run.status = "cancelled"
                        c_run.completed_at = datetime.now(timezone.utc)
                        c_run.error = {"code": "CANCELLED", "message": "User stopped generation."}
                        c_msg = Message(
                            thread_id=thread_id,
                            parent_message_id=m_id,
                            role="assistant",
                            content=f"{p_text}\n\n[Generation stopped by user]" if p_text else "[Stopped by user]",
                            sequence=(seq or 0) + 1,
                            status="cancelled",
                            metadata_={"run_id": str(r_id)},
                        )
                        session.add(c_msg)
                        await session.flush()
                        c_run.output_message_id = c_msg.id
                        await session.commit()
                except Exception:
                    pass

            p_text = "".join(accumulated_tokens).strip()
            asyncio.create_task(_mark_cancelled(run.id, message.id, message.thread_id, message.sequence, p_text))
            raise

        except Exception as exc:
            safe_exc = (
                exc
                if isinstance(exc, AppException)
                else sanitize_model_error(
                    exc,
                    provider=config.get("generation", {}).get("provider", "unknown"),
                    model=config.get("generation", {}).get("llm_model")
                    or config.get("generation", {}).get("model", "unknown"),
                )
            )
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.error = {
                "code": getattr(safe_exc, "code", "RUN_ERROR"),
                "message": getattr(safe_exc, "message", str(safe_exc)),
                "details": getattr(safe_exc, "details", None),
            }
            await self.db.commit()
            yield {"type": "error", "error": run.error}

    async def _persist_run_result(
        self,
        *,
        run: Run,
        message: Message,
        result: dict,
        config: dict,
        provider: BaseGenerationProvider,
    ) -> None:
        """Persist the assistant message and the durable stages of a run."""
        now = datetime.now(timezone.utc)
        docs = result.get("reranked_documents", []) or result.get("retrieved_documents", [])
        citations = []
        for i, doc in enumerate(docs, start=1):
            citations.append({
                "index": i,
                "name": doc.get("name", "Unknown"),
                "mime_type": doc.get("mime_type")
            })

        assistant_message = Message(
            thread_id=message.thread_id,
            parent_message_id=message.id,
            role="assistant",
            content=result.get("response", ""),
            sequence=(message.sequence or 0) + 1,
            status="completed",
            metadata_={
                "run_id": str(run.id),
                "citations": citations
            },
        )
        self.db.add(assistant_message)
        await self.db.flush()
        run.output_message_id = assistant_message.id
        result["output_message_id"] = str(assistant_message.id)

        if self.run_manager:
            generation = config.get("generation", {})
            provider_config = getattr(provider, "config", None)
            
            transformed_queries = result.get("transformed_queries", [])
            reranked_documents = result.get("reranked_documents", [])
            
            await self.run_manager.record_model_call(
                provider=getattr(provider, "provider_name", generation.get("provider", "unknown")),
                model=getattr(provider_config, "model", generation.get("model", "unknown")),
                request={
                    "query": message.content,
                    "transformed_queries": transformed_queries,
                    "document_ids": [doc.get("id") for doc in reranked_documents],
                    "temperature": generation.get("temperature"),
                    "max_tokens": generation.get("max_tokens"),
                },
                response={"content": result.get("response", "")},
                status="completed",
            )
