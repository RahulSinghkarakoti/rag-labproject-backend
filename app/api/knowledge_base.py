from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.document import DocumentResponse
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdateRequest,
)

router = APIRouter(
    prefix="/api",
    tags=["Knowledge Bases"],
)


async def _verify_project_ownership(
    project_id: UUID, user_id: int, db: AsyncSession
) -> Project:
    stmt = select(Project).where(
        Project.id == project_id, Project.user_id == user_id
    )
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return project


@router.get("/projects/{project_id}/knowledge-bases",
    response_model=APIResponse[List[KnowledgeBaseResponse]],
)
async def get_project_knowledge_bases(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_project_ownership(project_id, current_user.id, db)

    stmt = select(KnowledgeBase).where(
        KnowledgeBase.project_id == project_id,
        KnowledgeBase.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    knowledge_bases = result.scalars().all()

    # Enrich document count per KB
    enriched: List[KnowledgeBaseResponse] = []
    for kb in knowledge_bases:
        count_stmt = select(func.count(Document.id)).where(
            Document.knowledge_base_id == kb.id
        )
        count_res = await db.execute(count_stmt)
        doc_count = count_res.scalar_one() or 0

        kb_dto = KnowledgeBaseResponse.model_validate(kb)
        kb_dto.document_count = doc_count
        enriched.append(kb_dto)

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
    await _verify_project_ownership(project_id, current_user.id, db)

    chunking_dict = (
        payload.chunking_config.model_dump()
        if payload.chunking_config
        else {
            "strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "all-MiniLM-L6-v2",
        }
    )
    embedding_dict = (
        payload.embedding_config.model_dump()
        if payload.embedding_config
        else {
            "model": "all-MiniLM-L6-v2",
            # "dimension": 384,
        }
    ) 

    new_kb = KnowledgeBase(
        project_id=project_id,
        user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        chunking_config=chunking_dict,
        embedding_config=embedding_dict,
        indexing_config={},
        metadata_={},
    )
    db.add(new_kb)
    await db.commit()
    await db.refresh(new_kb)

    response_dto = KnowledgeBaseResponse.model_validate(new_kb)
    response_dto.document_count = 0
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
    await _verify_project_ownership(project_id, current_user.id, db)

    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.project_id == project_id,
        KnowledgeBase.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    count_stmt = select(func.count(Document.id)).where(
        Document.knowledge_base_id == kb.id
    )
    count_res = await db.execute(count_stmt)
    doc_count = count_res.scalar_one() or 0

    response_dto = KnowledgeBaseResponse.model_validate(kb)
    response_dto.document_count = doc_count

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
    await _verify_project_ownership(project_id, current_user.id, db)

    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.project_id == project_id,
        KnowledgeBase.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    if payload.name is not None:
        kb.name = payload.name
    if payload.description is not None:
        kb.description = payload.description
    if payload.chunking_config is not None:
        kb.chunking_config = payload.chunking_config.model_dump()
    if payload.embedding_config is not None:
        kb.embedding_config = payload.embedding_config.model_dump()
    if payload.indexing_config is not None:
        kb.indexing_config = payload.indexing_config.model_dump()

    await db.commit()
    await db.refresh(kb)

    response_dto = KnowledgeBaseResponse.model_validate(kb)
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
    await _verify_project_ownership(project_id, current_user.id, db)

    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.project_id == project_id,
        KnowledgeBase.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    await db.delete(kb)
    await db.commit()

    return APIResponse(
        success=True,
        message="Knowledge base deleted successfully",
        data=None,
    )


@router.get(
    "/projects/{project_id}/knowledge-bases/{kb_id}/documents",
    response_model=APIResponse[List[DocumentResponse]],
)
async def get_knowledge_base_documents(
    project_id: UUID,
    kb_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_project_ownership(project_id, current_user.id, db)

    stmt = select(Document).where(Document.knowledge_base_id == kb_id)
    result = await db.execute(stmt)
    documents = result.scalars().all()

    return APIResponse(
        success=True,
        message="Documents retrieved successfully",
        data=[DocumentResponse.model_validate(d) for d in documents],
    )
