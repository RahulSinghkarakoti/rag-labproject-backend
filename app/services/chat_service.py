import asyncio
from datetime import datetime, timezone
import json
from typing import List, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.sse import create_sse_response
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
    ):
        """Execute a streaming query run via QueryEngine and RunManager, returning an SSE response."""
        run_manager = RunManager(self.db)
        run = await run_manager.start_run(
            thread_id=thread.id,
            input_message_id=user_message.id,
            pipeline_version_id=pipeline_version.id,
            run_type="chat",
        )

        engine = QueryEngine(self.db, run_manager=run_manager)

        async def event_generator():
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

                if run_manager.run and run_manager.run.status not in ("completed", "failed", "cancelled"):
                    await run_manager.complete_run()
            except asyncio.CancelledError:
                # Handle client disconnection
                if run_manager.run and run_manager.run.status not in ("completed", "failed", "cancelled"):
                    await run_manager.cancel_run()
                raise
            except Exception as e:
                if run_manager.run and run_manager.run.status not in ("completed", "failed", "cancelled"):
                    await run_manager.fail_run(str(e))
                raise

        return create_sse_response(event_generator())
