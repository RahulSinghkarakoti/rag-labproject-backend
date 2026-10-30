import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_current_user
from app.core.sse import create_sse_response
from app.database.database import get_db
from app.models import Message, Run, Thread
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.pipeline import Pipeline, PipelineVersion
from app.models.project import Project
from app.rag import QueryEngine
from app.schemas.common import APIResponse
from app.schemas.pipeline import PipelineVersionResponse
from app.schemas.thread import (
    MessageCreate,
    MessageResponse,
    ThreadCreate,
    ThreadResponse,
    ThreadUpdate,
)

router = APIRouter(prefix="/api", tags=["Threads"])


async def _get_thread(thread_id: UUID, user_id: int, db: AsyncSession) -> Thread:
    thread = await db.scalar(
        select(Thread)
        .options(selectinload(Thread.knowledge_bases))
        .where(Thread.id == thread_id, Thread.user_id == user_id)
    )
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


async def _get_kbs(project_id: UUID, user_id: int, ids: list[UUID], db: AsyncSession):
    ids = list(dict.fromkeys(ids))
    kbs = list((await db.scalars(select(KnowledgeBase).where(
        KnowledgeBase.id.in_(
            ids), KnowledgeBase.project_id == project_id, KnowledgeBase.user_id == user_id,
    ))).all())
    if len(kbs) != len(ids):
        raise HTTPException(
            status_code=404, detail="One or more knowledge bases not found")
    if not ids or not await db.scalar(select(Document.id).where(Document.knowledge_base_id.in_(ids))):
        raise HTTPException(
            status_code=400, detail="Select a knowledge base with at least one document")
    return ids, kbs


@router.get("/projects/{project_id}/threads")
async def get_project_threads(project_id: UUID, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):  # noqa: B008
    if not await db.scalar(select(Project.id).where(Project.id == project_id, Project.user_id == current_user.id)):
        raise HTTPException(status_code=404, detail="Project not found")
    threads = (await db.scalars(
        select(Thread)
        .options(selectinload(Thread.knowledge_bases))
        .where(Thread.project_id == project_id, Thread.user_id == current_user.id)
        .order_by(Thread.updated_at.desc())
    )).all()
    return APIResponse(success=True, message="Threads retrieved successfully", data=[ThreadResponse.model_validate(t) for t in threads])


@router.post("/projects/{project_id}/threads", status_code=status.HTTP_201_CREATED)
async def create_thread(project_id: UUID, thread_data: ThreadCreate, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if not await db.scalar(select(Project.id).where(Project.id == project_id, Project.user_id == current_user.id)):
        raise HTTPException(status_code=404, detail="Project not found")
    _, kbs = await _get_kbs(project_id, current_user.id, thread_data.knowledge_base_ids, db)
    config = thread_data.config.model_dump() if thread_data.config else {}
    pipeline = Pipeline(name=thread_data.title, description="Thread pipeline",
                        project_id=project_id, user_id=current_user.id)
    db.add(pipeline)
    await db.flush()
    version = PipelineVersion(
        version=1, config=config, pipeline_id=pipeline.id)
    db.add(version)
    await db.flush()
    metadata = config
    thread = Thread(user_id=current_user.id,
                    project_id=project_id,
                    title=thread_data.title,
                    status="active",
                    pipeline_id=pipeline.id,
                    pipeline_version_id=version.id,
                    metadata_=metadata)
    thread.knowledge_bases = kbs
    db.add(thread)
    await db.commit()
    thread = await db.scalar(
        select(Thread)
        .options(selectinload(Thread.knowledge_bases))
        .where(Thread.id == thread.id)
    )
    return APIResponse(success=True, message="Thread created successfully", data=ThreadResponse.model_validate(thread))


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: UUID, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    thread = await _get_thread(thread_id, current_user.id, db)
    return APIResponse(success=True, message="Thread retrieved successfully", data=ThreadResponse.model_validate(thread))


@router.get("/threads/{thread_id}/pipeline/latest", response_model=APIResponse[PipelineVersionResponse])
async def get_latest_thread_pipeline_version(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_thread(thread_id, current_user.id, db)
    if not thread.pipeline_id:
        raise HTTPException(status_code=404, detail="Thread has no pipeline")

    version = await db.scalar(
        select(PipelineVersion)
        .where(PipelineVersion.pipeline_id == thread.pipeline_id)
        .order_by(PipelineVersion.version.desc())
    )
    if not version:
        raise HTTPException(
            status_code=404, detail="Pipeline version not found")

    return APIResponse(
        success=True,
        message="Latest pipeline version retrieved successfully",
        data=PipelineVersionResponse.model_validate(version),
    )


@router.patch("/threads/{thread_id}")
async def update_thread(thread_id: UUID, thread_data: ThreadUpdate, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    thread = await _get_thread(thread_id, current_user.id, db)

    print("Updating thread with data:", thread_data)
    if thread_data.title is not None:
        thread.title = thread_data.title
    if thread_data.knowledge_base_ids is not None:
        kb_ids, kbs = await _get_kbs(thread.project_id, current_user.id, thread_data.knowledge_base_ids, db)
        thread.knowledge_bases = kbs
    # print("Updating thread metadata with config:", thread_data)
    incoming_config = thread_data.config.model_dump() if thread_data.config else None
    if incoming_config is not None:
        current = dict(thread.metadata_ or {})
        latest = await db.scalar(select(PipelineVersion).where(PipelineVersion.pipeline_id == thread.pipeline_id).order_by(PipelineVersion.version.desc()))
        config_changed = incoming_config != (latest.config if latest else {})
        if config_changed:
            # print("Config changed, creating new pipeline version...",current)
            pipeline_id = thread.pipeline_id
            version = PipelineVersion(version=(
                latest.version + 1 if latest else 1), config=incoming_config, pipeline_id=pipeline_id)
            db.add(version)
            await db.flush()
            thread.pipeline_version_id = version.id
            current.update(incoming_config)
        thread.metadata_ = current
    await db.commit()
    await db.refresh(thread)
    return APIResponse(success=True, message="Thread updated successfully", data=ThreadResponse.model_validate(thread))


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: UUID, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    thread = await _get_thread(thread_id, current_user.id, db)
    await db.delete(thread)
    await db.commit()
    return APIResponse(success=True, message="Thread deleted successfully", data=None)


@router.get("/threads/{thread_id}/messages", response_model=APIResponse[list[MessageResponse]])
async def get_thread_messages(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_thread(thread_id, current_user.id, db)
    messages = (
        await db.scalars(
            select(Message)
            .where(Message.thread_id == thread_id)
            .order_by(Message.sequence.asc())
        )
    ).all()
    return APIResponse(
        success=True,
        message="Thread messages retrieved successfully",
        data=[MessageResponse.model_validate(message) for message in messages],
    )


async def _get_pipeline_version(thread: Thread, db: AsyncSession) -> PipelineVersion:
    if not thread.pipeline_id:
        raise HTTPException(status_code=400, detail="Thread has no active pipeline")
    pipeline_version = await db.scalar(
        select(PipelineVersion)
        .where(PipelineVersion.pipeline_id == thread.pipeline_id)
        .order_by(PipelineVersion.version.desc())
        .limit(1)
    )
    if not pipeline_version:
        raise HTTPException(
            status_code=400,
            detail="Thread has no active pipeline"
        )
    return pipeline_version


async def _stream_chat_run(
    thread: Thread,
    user_message: Message,
    pipeline_version: PipelineVersion,
    db: AsyncSession,
):
    from app.rag.run_manager import RunManager
    import asyncio
    run_manager = RunManager(db)
    run = await run_manager.start_run(
        thread_id=thread.id,
        input_message_id=user_message.id,
        pipeline_version_id=pipeline_version.id,
        run_type="chat",
    )

    engine = QueryEngine(db, run_manager=run_manager)

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

    return create_sse_response(event_generator())


@router.post("/threads/{thread_id}/messages")
async def create_message(
    thread_id: UUID,
    message_data: MessageCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_thread(thread_id, current_user.id, db)
    pipeline_version = await _get_pipeline_version(thread, db)

    # 1. Edit / re-process an existing message if message_id is provided and valid
    if message_data.message_id:
        try:
            target_uuid = UUID(str(message_data.message_id))
        except (ValueError, TypeError):
            target_uuid = None

        if target_uuid:
            target_message = await db.scalar(
                select(Message).where(
                    Message.id == target_uuid,
                    Message.thread_id == thread.id,
                )
            )
            if target_message:
                if target_message.role == "assistant":
                    user_message = None
                    if target_message.parent_message_id:
                        user_message = await db.scalar(
                            select(Message).where(
                                Message.id == target_message.parent_message_id,
                                Message.thread_id == thread.id,
                            )
                        )
                    if not user_message:
                        user_message = await db.scalar(
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
                            status_code=400, detail="No preceding user message to reprocess")
                else:
                    user_message = target_message
                    if message_data.content:
                        user_message.content = message_data.content

                await db.execute(
                    delete(Message).where(
                        Message.thread_id == thread.id,
                        Message.sequence > user_message.sequence,
                    )
                )
                await db.commit()
                return await _stream_chat_run(thread, user_message, pipeline_version, db)

    # 2. Standard new message flow (new thread or resume conversation)
    if not message_data.content:
        raise HTTPException(
            status_code=400, detail="Message content is required")

    last_sequence = await db.scalar(
        select(func.max(Message.sequence)).where(
            Message.thread_id == thread.id)
    )
    next_sequence = (last_sequence or 0) + 1

    user_message = Message(
        thread_id=thread.id,
        role="human",
        content=message_data.content,
        sequence=next_sequence,
        status="completed",
    )
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    return await _stream_chat_run(thread, user_message, pipeline_version, db)


@router.put("/threads/{thread_id}/messages/{message_id}")
@router.post("/threads/{thread_id}/messages/{message_id}/edit")
async def edit_message(
    thread_id: UUID,
    message_id: UUID,
    message_data: MessageCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_thread(thread_id, current_user.id, db)
    pipeline_version = await _get_pipeline_version(thread, db)

    target_message = await db.scalar(
        select(Message).where(
            Message.id == message_id,
            Message.thread_id == thread.id,
        )
    )
    if not target_message:
        raise HTTPException(status_code=404, detail="Message not found")

    if target_message.role != "human":
        raise HTTPException(
            status_code=400, detail="Only user messages can be edited")

    if message_data.content:
        target_message.content = message_data.content

    # shortcut: linear thread truncation drops subsequent messages on edit/retry instead of storing tree branches, upgrade when UI needs multi-branch history navigation.
    await db.execute(
        delete(Message).where(
            Message.thread_id == thread.id,
            Message.sequence > target_message.sequence,
        )
    )
    await db.commit()

    return await _stream_chat_run(thread, target_message, pipeline_version, db)


@router.post("/threads/{thread_id}/retry")
@router.post("/threads/{thread_id}/messages/{message_id}/retry")
async def retry_message(
    thread_id: UUID,
    message_id: UUID | None = None,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_thread(thread_id, current_user.id, db)
    pipeline_version = await _get_pipeline_version(thread, db)

    if message_id:
        target_message = await db.scalar(
            select(Message).where(
                Message.id == message_id,
                Message.thread_id == thread.id,
            )
        )
        if not target_message:
            raise HTTPException(status_code=404, detail="Message not found")

        if target_message.role == "assistant":
            user_message = None
            if target_message.parent_message_id:
                user_message = await db.scalar(
                    select(Message).where(
                        Message.id == target_message.parent_message_id,
                        Message.thread_id == thread.id,
                    )
                )
            if not user_message:
                user_message = await db.scalar(
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
                    status_code=400, detail="No preceding user message found to retry")
        else:
            user_message = target_message
    else:
        user_message = await db.scalar(
            select(Message)
            .where(
                Message.thread_id == thread.id,
                Message.role == "human",
            )
            .order_by(Message.sequence.desc())
            .limit(1)
        )
        if not user_message:
            raise HTTPException(
                status_code=400, detail="No user message found to retry")

    await db.execute(
        delete(Message).where(
            Message.thread_id == thread.id,
            Message.sequence > user_message.sequence,
        )
    )
    await db.commit()

    return await _stream_chat_run(thread, user_message, pipeline_version, db)
