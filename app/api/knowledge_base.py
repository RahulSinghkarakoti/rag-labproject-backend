from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.document import DocumentResponse
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdateRequest,
)
from app.services.knowledge_base_service import KnowledgeBaseService

router = APIRouter(
    tags=["Knowledge Bases"],
)


@router.get(
    "/projects/{project_id}/knowledge-bases",
    response_model=APIResponse[list[KnowledgeBaseResponse]],
)
async def get_project_knowledge_bases(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = KnowledgeBaseService(db)
    enriched = await service.get_project_knowledge_bases(
        project_id=project_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Knowledge bases retrieved successfully",
        data=enriched,
    )


@router.post(
    "/projects/{project_id}/knowledge-bases",
    response_model=APIResponse[KnowledgeBaseResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    project_id: UUID,
    payload: KnowledgeBaseCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = KnowledgeBaseService(db)
    response_dto = await service.create_knowledge_base(
        project_id=project_id, user_id=current_user.id, payload=payload
    )
    return APIResponse(
        success=True,
        message="Knowledge base created successfully",
        data=response_dto,
    )


@router.get(
    "/projects/{project_id}/knowledge-bases/{kb_id}",
    response_model=APIResponse[KnowledgeBaseResponse],
)
async def get_knowledge_base(
    project_id: UUID,
    kb_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = KnowledgeBaseService(db)
    response_dto = await service.get_knowledge_base(
        project_id=project_id, kb_id=kb_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Knowledge base retrieved successfully",
        data=response_dto,
    )


@router.patch(
    "/projects/{project_id}/knowledge-bases/{kb_id}",
    response_model=APIResponse[KnowledgeBaseResponse],
)
async def update_knowledge_base(
    project_id: UUID,
    kb_id: UUID,
    payload: KnowledgeBaseUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = KnowledgeBaseService(db)
    response_dto = await service.update_knowledge_base(
        project_id=project_id,
        kb_id=kb_id,
        payload=payload,
        user_id=current_user.id,
    )
    return APIResponse(
        success=True,
        message="Knowledge base updated successfully",
        data=response_dto,
    )


@router.delete(
    "/projects/{project_id}/knowledge-bases/{kb_id}",
    response_model=APIResponse[None],
)
async def delete_knowledge_base(
    project_id: UUID,
    kb_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = KnowledgeBaseService(db)
    await service.delete_knowledge_base(
        project_id=project_id, kb_id=kb_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Knowledge base deleted successfully",
        data=None,
    )


# Preserved for backward-compatibility with direct module imports (e.g., tests)
async def get_knowledge_base_documents(
    project_id: UUID,
    kb_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = KnowledgeBaseService(db)
    documents = await service.get_knowledge_base_documents(
        project_id=project_id, kb_id=kb_id, user_id=current_user.id
    )
    return APIResponse(
        success=True,
        message="Documents retrieved successfully",
        data=[DocumentResponse.model_validate(d) for d in documents],
    )
