from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
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


@router.get("/projects/{project_id}/threads", response_model=APIResponse[List[ThreadResponse]])
async def get_project_threads(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    threads = await service.get_project_threads(
        project_id=project_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Threads retrieved successfully",
        data=[ThreadResponse.model_validate(t) for t in threads],
    )


@router.post("/projects/{project_id}/threads", status_code=status.HTTP_201_CREATED)
async def create_thread(
    project_id: UUID,
    thread_data: ThreadCreate,
    current_user: User = Depends(get_current_user),
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


@router.get("/threads/{thread_id}", response_model=APIResponse[ThreadResponse])
async def get_thread(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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


@router.patch("/threads/{thread_id}", response_model=APIResponse[ThreadResponse])
async def update_thread(
    thread_id: UUID,
    thread_data: ThreadUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    config = thread_data.config.model_dump() if thread_data.config else None
    thread = await service.update_thread(
        thread_id=thread_id,
        user_id=current_user.id,
        title=thread_data.title,
        knowledge_base_ids=thread_data.knowledge_base_ids,
        config=config,
    )
    return APIResponse(
        success=True,
        message="Thread updated successfully",
        data=ThreadResponse.model_validate(thread),
    )


@router.delete("/threads/{thread_id}", response_model=APIResponse[None])
async def delete_thread(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    await service.delete_thread(thread_id=thread_id, user_id=current_user.id)
    return APIResponse(
        success=True, message="Thread deleted successfully", data=None
    )


@router.get(
    "/threads/{thread_id}/messages",
    response_model=APIResponse[List[MessageResponse]],
)
async def get_thread_messages(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    messages = await service.get_thread_messages(
        thread_id=thread_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Thread messages retrieved successfully",
        data=[MessageResponse.model_validate(message) for message in messages],
    )


@router.post("/threads/{thread_id}/messages")
async def create_message(
    thread_id: UUID,
    message_data: MessageCreate,
    request: Request = None,
    current_user: User = Depends(get_current_user),
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
    return await service.stream_chat_run(thread, user_message, pipeline_version, request=request)


@router.put("/threads/{thread_id}/messages/{message_id}")
@router.post("/threads/{thread_id}/messages/{message_id}/edit")
async def edit_message(
    thread_id: UUID,
    message_id: UUID,
    message_data: MessageCreate,
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    pipeline_version = await service.get_latest_pipeline_version(thread)
    target_message = await service.edit_message(
        thread, message_id, message_data.content
    )
    return await service.stream_chat_run(
        thread, target_message, pipeline_version, request=request
    )


@router.post("/threads/{thread_id}/retry")
@router.post("/threads/{thread_id}/messages/{message_id}/retry")
async def retry_message(
    thread_id: UUID,
    message_id: UUID | None = None,
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    thread = await service.get_thread(thread_id, current_user.id)
    pipeline_version = await service.get_latest_pipeline_version(thread)
    user_message = await service.resolve_retry_message(thread, message_id)
    return await service.stream_chat_run(thread, user_message, pipeline_version, request=request)
