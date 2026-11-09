import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import List, Tuple
from uuid import UUID

from fastapi import HTTPException, Request, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.cancellation import cancellation_manager
from app.core.sse import create_sse_response

logger = logging.getLogger(__name__)
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.pipeline import Pipeline, PipelineVersion
from app.models.project import Project
from app.models.thread import Thread
from app.rag import QueryEngine
from app.rag.run_manager import RunManager


class ChatService:
    """Service handling thread lifecycle, pipeline version syncing,

    message branching/truncation, and chat execution streaming.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_thread(self, thread_id: UUID, user_id: int) -> Thread:
        """Fetch a thread by ID verifying user ownership, preloading its knowledge bases."""
        thread = await self.db.scalar(
            select(Thread)
            .options(selectinload(Thread.knowledge_bases))
            .where(Thread.id == thread_id, Thread.user_id == user_id)
        )
        if not thread:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        return thread

    async def validate_knowledge_bases(
        self, project_id: UUID, user_id: int, kb_ids: List[UUID]
    ) -> Tuple[List[UUID], List[KnowledgeBase]]:
        """Validate that all supplied knowledge bases exist, belong to the user/project,

        and contain at least one document.
        """
        unique_ids = list(dict.fromkeys(kb_ids))
        kbs = list(
            (
                await self.db.scalars(
                    select(KnowledgeBase).where(
                        KnowledgeBase.id.in_(unique_ids),
                        KnowledgeBase.project_id == project_id,
                        KnowledgeBase.user_id == user_id,
                    )
                )
            ).all()
        )
        if len(kbs) != len(unique_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more knowledge bases not found",
            )
        if not unique_ids or not await self.db.scalar(
            select(Document.id).where(Document.knowledge_base_id.in_(unique_ids))
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Select a knowledge base with at least one document",
            )
        return unique_ids, kbs

    async def create_thread(
        self,
        project_id: UUID,
        user_id: int,
        title: str,
        kb_ids: List[UUID],
        config: dict | None,
    ) -> Thread:
        """Create a new thread along with its associated pipeline and version."""
        if not await self.db.scalar(
            select(Project.id).where(Project.id == project_id, Project.user_id == user_id)
        ):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        _, kbs = await self.validate_knowledge_bases(project_id, user_id, kb_ids)
        pipeline_config = config or {}

        pipeline = Pipeline(
            name=title,
            description="Thread pipeline",
            project_id=project_id,
            user_id=user_id,
        )
        self.db.add(pipeline)
        await self.db.flush()

        version = PipelineVersion(version=1, config=pipeline_config, pipeline_id=pipeline.id)
        self.db.add(version)
        await self.db.flush()

        thread = Thread(
            user_id=user_id,
            project_id=project_id,
            title=title,
            status="active",
            pipeline_id=pipeline.id,
            pipeline_version_id=version.id,
            metadata_=pipeline_config,
        )
        thread.knowledge_bases = kbs
        self.db.add(thread)
        await self.db.commit()

        refreshed = await self.db.scalar(
            select(Thread)
            .options(selectinload(Thread.knowledge_bases))
            .where(Thread.id == thread.id)
        )
        return refreshed

    async def sync_thread_pipeline_config(
        self, thread: Thread, incoming_config: dict | None
    ) -> None:
        """Sync incoming thread configuration with pipeline versions.

        If the configuration differs from the latest version, creates a new PipelineVersion.
        """
        if incoming_config is None:
            return

        current = dict(thread.metadata_ or {})
        latest = await self.db.scalar(
            select(PipelineVersion)
            .where(PipelineVersion.pipeline_id == thread.pipeline_id)
            .order_by(PipelineVersion.version.desc())
        )
        config_changed = incoming_config != (latest.config if latest else {})
        if config_changed:
            version = PipelineVersion(
                version=(latest.version + 1 if latest else 1),
                config=incoming_config,
                pipeline_id=thread.pipeline_id,
            )
            self.db.add(version)
            await self.db.flush()
            thread.pipeline_version_id = version.id
            current.update(incoming_config)

        thread.metadata_ = current

    async def get_latest_pipeline_version(self, thread: Thread) -> PipelineVersion:
        """Retrieve the latest pipeline version associated with the thread."""
        if not thread.pipeline_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Thread has no active pipeline",
            )
        pipeline_version = await self.db.scalar(
            select(PipelineVersion)
            .where(PipelineVersion.pipeline_id == thread.pipeline_id)
            .order_by(PipelineVersion.version.desc())
            .limit(1)
        )
        if not pipeline_version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Thread has no active pipeline",
            )
        return pipeline_version

    async def truncate_thread_after_sequence(self, thread_id: UUID, sequence: int) -> None:
        """Deprecated: preserved for interface compatibility.
        Multi-branch conversation history preserves all runs and messages.
        """
        return

    async def prepare_chat_message(
        self,
        thread: Thread,
        content: str | None,
        target_message_id: UUID | str | None = None,
    ) -> Message:
        """Prepare a message for execution. Supports creating a new message or
        branching from an existing message target.
        """
        # 1. Edit / re-process an existing message if target_message_id is provided
        if target_message_id:
            try:
                target_uuid = UUID(str(target_message_id))
            except (ValueError, TypeError):
                target_uuid = None

            if target_uuid:
                target_message = await self.db.scalar(
                    select(Message).where(
                        Message.id == target_uuid,
                        Message.thread_id == thread.id,
                    )
                )
            if target_message:
                if target_message.role == "assistant":
                    user_message = None
                    if target_message.parent_message_id:
                        user_message = await self.db.scalar(
                            select(Message).where(
                                Message.id == target_message.parent_message_id,
                                Message.thread_id == thread.id,
                            )
                        )
                    if not user_message:
                        user_message = await self.db.scalar(
                            select(Message)
                            .where(
                                Message.thread_id == thread.id,
                                Message.sequence < target_message.sequence,
                                Message.role == "human",
                            )
                            .order_by(Message.sequence.desc())
                            .limit(1)
                        )
                    if not user_message:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No preceding user message to reprocess",
                        )
                    return user_message
                else:
                    # Target is a human message: create a new branch message
                    now = datetime.now(timezone.utc)
                    user_message = Message(
                        thread_id=thread.id,
                        role="human",
                        content=content or target_message.content,
                        parent_message_id=target_message.parent_message_id,
                        sequence=target_message.sequence,
                        status="completed",
                        metadata_={"created_at": now.isoformat()},
                    )
                    self.db.add(user_message)
                    await self.db.commit()
                    await self.db.refresh(user_message)
                    return user_message

        # 2. Standard new message flow
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message content is required",
            )

        last_sequence = await self.db.scalar(
            select(func.max(Message.sequence)).where(Message.thread_id == thread.id)
        )
        next_sequence = (last_sequence or 0) + 1

        last_assistant_message = await self.db.scalar(
            select(Message)
            .where(
                Message.thread_id == thread.id,
                Message.role == "assistant",
            )
            .order_by(
                Message.sequence.desc(),
                Message.metadata_["created_at"].astext.desc().nulls_last(),
            )
            .limit(1)
        )
        parent_id = last_assistant_message.id if last_assistant_message else None

        now = datetime.now(timezone.utc)
        user_message = Message(
            thread_id=thread.id,
            role="human",
            content=content,
            parent_message_id=parent_id,
            sequence=next_sequence,
            status="completed",
            metadata_={"created_at": now.isoformat()},
        )
        self.db.add(user_message)
        await self.db.commit()
        await self.db.refresh(user_message)
        return user_message

    async def edit_message(
        self, thread: Thread, message_id: UUID, new_content: str | None
    ) -> Message:
        """Edit a user message by branching into a new message with identical parent."""
        target_message = await self.db.scalar(
            select(Message).where(
                Message.id == message_id,
                Message.thread_id == thread.id,
            )
        )
        if not target_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found",
            )
        if target_message.role != "human":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only user messages can be edited",
            )

        now = datetime.now(timezone.utc)
        user_message = Message(
            thread_id=thread.id,
            role="human",
            content=new_content or target_message.content,
            parent_message_id=target_message.parent_message_id,
            sequence=target_message.sequence,
            status="completed",
            metadata_={"created_at": now.isoformat()},
        )
        self.db.add(user_message)
        await self.db.commit()
        await self.db.refresh(user_message)
        return user_message

    async def resolve_retry_message(
        self, thread: Thread, message_id: UUID | None = None
    ) -> Message:
        """Resolve which human message to retry, truncating downstream messages."""
        if message_id:
            target_message = await self.db.scalar(
                select(Message).where(
                    Message.id == message_id,
                    Message.thread_id == thread.id,
                )
            )
            if not target_message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Message not found",
                )

            if target_message.role == "assistant":
                user_message = None
                if target_message.parent_message_id:
                    user_message = await self.db.scalar(
                        select(Message).where(
                            Message.id == target_message.parent_message_id,
                            Message.thread_id == thread.id,
                        )
                    )
                if not user_message:
                    user_message = await self.db.scalar(
                        select(Message)
                        .where(
                            Message.thread_id == thread.id,
                            Message.sequence < target_message.sequence,
                            Message.role == "human",
                        )
                        .order_by(Message.sequence.desc())
                        .limit(1)
                    )
                if not user_message:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No preceding user message found to retry",
                    )
            else:
                user_message = target_message
        else:
            user_message = await self.db.scalar(
                select(Message)
                .where(
                    Message.thread_id == thread.id,
                    Message.role == "human",
                )
                .order_by(
                    Message.sequence.desc(),
                    Message.metadata_["created_at"].astext.desc().nulls_last(),
                )
                .limit(1)
            )
            if not user_message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No user message found to retry",
                )

        # Preserve prior assistant responses on retry so that the frontend can detect
        # multiple alternative responses for the same turn via identical parent_message_id.
        return user_message

    async def stream_chat_run(
        self,
        thread: Thread,
        user_message: Message,
        pipeline_version: PipelineVersion,
        request: Request | None = None,
    ):
        """Execute a streaming query run via QueryEngine and RunManager, returning an SSE response."""
        run_manager = RunManager(self.db)
        run = await run_manager.start_run(
            thread_id=thread.id,
            input_message_id=user_message.id,
            pipeline_version_id=pipeline_version.id,
            run_type="chat",
        )

        cancellation_manager.register(run.id)
        engine = QueryEngine(self.db, run_manager=run_manager)

        async def event_generator():
            disconnect_task = None
            if request:
                async def _monitor_disconnect():
                    try:
                        while True:
                            if await request.is_disconnected():
                                logger.info("[stream_chat_run] Client disconnected for run %s", run.id)
                                await cancellation_manager.cancel(run.id)
                                break
                            await asyncio.sleep(0.5)
                    except asyncio.CancelledError:
                        pass
                disconnect_task = asyncio.create_task(_monitor_disconnect())

            try:
                yield {
                    "event": "started",
                    "data": json.dumps({"run_id": str(run.id), "message_id": str(user_message.id)}),
                }
                async for event in engine.run_stream(
                    message=user_message,
                    thread=thread,
                    pipeline_version=pipeline_version,
                    run=run,
                ):
                    yield {"event": event["type"], "data": json.dumps(event)}

                if not cancellation_manager.is_cancelled(run.id) and run_manager.run and run_manager.run.status not in ("completed", "failed", "cancelled"):
                    await run_manager.complete_run()
            except asyncio.CancelledError:
                logger.info("[stream_chat_run] Handling cancellation for run %s", run.id)
                # Shield cancellation cleanup so database records are reliably persisted
                async def _cleanup_cancellation():
                    from app.database.database import AsyncSessionLocal
                    p_text = getattr(run, "_partial_text", "").strip()
                    async with AsyncSessionLocal() as session:
                        from app.models.run import Run as RunModel
                        from app.models.message import Message as MessageModel
                        c_run = await session.get(RunModel, run.id)
                        if c_run and c_run.status != "completed":
                            c_run.status = "cancelled"
                            c_run.completed_at = datetime.now(timezone.utc)
                            c_run.error = {"code": "CANCELLED", "message": "User stopped generation."}
                            content = f"{p_text}\n\n[Generation stopped by user]" if p_text else "[Generation stopped by user]"
                            c_msg = MessageModel(
                                thread_id=thread.id,
                                parent_message_id=user_message.id,
                                role="assistant",
                                content=content,
                                sequence=(user_message.sequence or 0) + 1,
                                status="cancelled",
                                metadata_={"run_id": str(run.id)},
                            )
                            session.add(c_msg)
                            await session.flush()
                            c_run.output_message_id = c_msg.id
                            await session.commit()
                            return c_run, c_msg
                    return None, None

                _, c_msg = await asyncio.shield(_cleanup_cancellation())
                yield {
                    "event": "cancelled",
                    "data": json.dumps({
                        "type": "cancelled",
                        "run_id": str(run.id),
                        "output_message_id": str(c_msg.id) if c_msg else None,
                        "response": c_msg.content if c_msg else "[Generation stopped by user]",
                        "message": "Generation stopped by user",
                    }),
                }
                raise
            except Exception as e:
                if run_manager.run and run_manager.run.status not in ("completed", "failed", "cancelled"):
                    await run_manager.fail_run(str(e))
                raise
            finally:
                if disconnect_task and not disconnect_task.done():
                    disconnect_task.cancel()
                cancellation_manager.unregister(run.id)

        return create_sse_response(event_generator())

    async def get_project_threads(
        self, project_id: UUID, user_id: int
    ) -> List[Thread]:
        """Fetch all threads for a project after verifying project ownership."""
        if not await self.db.scalar(
            select(Project.id).where(
                Project.id == project_id, Project.user_id == user_id
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )

        threads = (
            await self.db.scalars(
                select(Thread)
                .options(selectinload(Thread.knowledge_bases))
                .where(
                    Thread.project_id == project_id,
                    Thread.user_id == user_id,
                )
                .order_by(Thread.updated_at.desc())
            )
        ).all()
        return list(threads)

    async def update_thread(
        self,
        thread_id: UUID,
        user_id: int,
        title: str | None = None,
        knowledge_base_ids: List[UUID] | None = None,
        config: dict | None = None,
    ) -> Thread:
        """Update thread metadata, associated knowledge bases, and pipeline config."""
        thread = await self.get_thread(thread_id, user_id)

        if title is not None:
            thread.title = title

        if knowledge_base_ids is not None:
            _, kbs = await self.validate_knowledge_bases(
                thread.project_id, user_id, knowledge_base_ids
            )
            thread.knowledge_bases = kbs

        if config is not None:
            await self.sync_thread_pipeline_config(thread, config)

        await self.db.commit()
        await self.db.refresh(thread)
        return thread

    async def delete_thread(self, thread_id: UUID, user_id: int) -> None:
        """Delete thread after verifying user ownership."""
        thread = await self.get_thread(thread_id, user_id)
        await self.db.delete(thread)
        await self.db.commit()

    async def get_thread_messages(
        self, thread_id: UUID, user_id: int
    ) -> List[Message]:
        """Fetch all messages for a thread ordered by sequence and creation."""
        await self.get_thread(thread_id, user_id)
        messages = (
            await self.db.scalars(
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(
                    Message.sequence.asc(),
                    Message.metadata_["created_at"].astext.asc().nulls_first(),
                )
            )
        ).all()
        return list(messages)

