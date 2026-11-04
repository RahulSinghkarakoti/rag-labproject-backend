from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models import Message, Thread
from app.models.project import Project
from app.schemas.common import APIResponse
from app.schemas.pipeline import PipelineVersionResponse
from app.schemas.thread import (
    MessageCreate,
    MessageResponse,
    ThreadCreate,
    ThreadResponse,
    ThreadUpdate,
)
from app.services.chat_service import ChatService

router = APIRouter(tags=["Threads"])


@router.get("/projects/{project_id}/threads")
async def get_project_threads(
    project_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not await db.scalar(
        select(Project.id).where(Project.id == project_id, Project.user_id == current_user.id)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    threads = (
        await db.scalars(
            select(Thread)
            .options(selectinload(Thread.knowledge_bases))
            .where(Thread.project_id == project_id, Thread.user_id == current_user.id)
            .order_by(Thread.updated_at.desc())
        )
    ).all()
    return APIResponse(
        success=True,
        message="Threads retrieved successfully",
        data=[ThreadResponse.model_validate(t) for t in threads],
    )


@router.post("/projects/{project_id}/threads", status_code=status.HTTP_201_CREATED)
async def create_thread(
    project_id: UUID,
    thread_data: ThreadCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    config = thread_data.config.model_dump() if thread_data.config else {}
    thread = await service.create_thread(
        project_id=project_id,
        user_id=current_user.id,
        title=thread_data.title,
        kb_ids=thread_data.knowledge_base_ids,
        config=config,
    )
    return APIResponse(
        success=True,
        message="Thread created successfully",
        data=ThreadResponse.model_validate(thread),
    )


@router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    return APIResponse(
        success=True,
        message="Thread retrieved successfully",
        data=ThreadResponse.model_validate(thread),
    )


@router.get(
    "/threads/{thread_id}/pipeline/latest",
    response_model=APIResponse[PipelineVersionResponse],
)
async def get_latest_thread_pipeline_version(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    version = await service.get_latest_pipeline_version(thread)
    return APIResponse(
        success=True,
        message="Latest pipeline version retrieved successfully",
        data=PipelineVersionResponse.model_validate(version),
    )


@router.patch("/threads/{thread_id}")
async def update_thread(
    thread_id: UUID,
    thread_data: ThreadUpdate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)

    if thread_data.title is not None:
        thread.title = thread_data.title

    if thread_data.knowledge_base_ids is not None:
        _, kbs = await service.validate_knowledge_bases(
            thread.project_id, current_user.id, thread_data.knowledge_base_ids
        )
        thread.knowledge_bases = kbs

    incoming_config = thread_data.config.model_dump() if thread_data.config else None
    if incoming_config is not None:
        await service.sync_thread_pipeline_config(thread, incoming_config)

    await db.commit()
    await db.refresh(thread)
    return APIResponse(
        success=True,
        message="Thread updated successfully",
        data=ThreadResponse.model_validate(thread),
    )


@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    await db.delete(thread)
    await db.commit()
    return APIResponse(success=True, message="Thread deleted successfully", data=None)


@router.get(
    "/threads/{thread_id}/messages",
    response_model=APIResponse[list[MessageResponse]],
)
async def get_thread_messages(
    thread_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    await service.get_thread(thread_id, current_user.id)
    messages = (
        await db.scalars(
            select(Message)
            .where(Message.thread_id == thread_id)
            .order_by(
                Message.sequence.asc(),
                Message.metadata_["created_at"].astext.asc().nulls_first(),
            )
        )
    ).all()
    return APIResponse(
        success=True,
        message="Thread messages retrieved successfully",
        data=[MessageResponse.model_validate(message) for message in messages],
    )


@router.post("/threads/{thread_id}/messages")
async def create_message(
    thread_id: UUID,
    message_data: MessageCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    pipeline_version = await service.get_latest_pipeline_version(thread)
    user_message = await service.prepare_chat_message(
        thread,
        content=message_data.content,
        target_message_id=message_data.message_id,
    )
    return await service.stream_chat_run(thread, user_message, pipeline_version)


@router.put("/threads/{thread_id}/messages/{message_id}")
@router.post("/threads/{thread_id}/messages/{message_id}/edit")
async def edit_message(
    thread_id: UUID,
    message_id: UUID,
    message_data: MessageCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    pipeline_version = await service.get_latest_pipeline_version(thread)
    target_message = await service.edit_message(thread, message_id, message_data.content)
    return await service.stream_chat_run(thread, target_message, pipeline_version)


@router.post("/threads/{thread_id}/retry")
@router.post("/threads/{thread_id}/messages/{message_id}/retry")
async def retry_message(
    thread_id: UUID,
    message_id: UUID | None = None,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    pipeline_version = await service.get_latest_pipeline_version(thread)
    user_message = await service.resolve_retry_message(thread, message_id)
    return await service.stream_chat_run(thread, user_message, pipeline_version)
