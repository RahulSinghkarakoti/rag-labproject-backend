from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdateRequest,
)


class KnowledgeBaseService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def verify_project_ownership(
        self, project_id: UUID, user_id: int
    ) -> Project:
        stmt = select(Project).where(
            Project.id == project_id, Project.user_id == user_id
        )
        result = await self.db.execute(stmt)
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )
        return project

    async def get_project_knowledge_bases(
        self, project_id: UUID, user_id: int
    ) -> list[KnowledgeBaseResponse]:
        await self.verify_project_ownership(project_id, user_id)

        stmt = (
            select(KnowledgeBase, func.count(Document.id))
            .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
            .where(
                KnowledgeBase.project_id == project_id,
                KnowledgeBase.user_id == user_id,
            )
            .group_by(KnowledgeBase.id)
        )
        result = await self.db.execute(stmt)

        enriched: list[KnowledgeBaseResponse] = []
        for kb, doc_count in result.all():
            kb_dto = KnowledgeBaseResponse.model_validate(kb)
            kb_dto.document_count = doc_count or 0
            enriched.append(kb_dto)

        return enriched

    async def create_knowledge_base(
        self, project_id: UUID, user_id: int, payload: KnowledgeBaseCreateRequest
    ) -> KnowledgeBaseResponse:
        await self.verify_project_ownership(project_id, user_id)

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
            }
        )

        new_kb = KnowledgeBase(
            project_id=project_id,
            user_id=user_id,
            name=payload.name,
            description=payload.description,
            chunking_config=chunking_dict,
            embedding_config=embedding_dict,
            indexing_config={},
            metadata_={},
        )
        self.db.add(new_kb)
        await self.db.commit()
        await self.db.refresh(new_kb)

        response_dto = KnowledgeBaseResponse.model_validate(new_kb)
        response_dto.document_count = 0
        return response_dto

    async def get_knowledge_base(
        self, project_id: UUID, kb_id: UUID, user_id: int
    ) -> KnowledgeBaseResponse:
        await self.verify_project_ownership(project_id, user_id)

        stmt = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.project_id == project_id,
            KnowledgeBase.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        kb = result.scalar_one_or_none()
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge base not found",
            )

        count_stmt = select(func.count(Document.id)).where(
            Document.knowledge_base_id == kb.id
        )
        count_res = await self.db.execute(count_stmt)
        doc_count = count_res.scalar_one() or 0

        response_dto = KnowledgeBaseResponse.model_validate(kb)
        response_dto.document_count = doc_count
        return response_dto

    async def update_knowledge_base(
        self,
        project_id: UUID,
        kb_id: UUID,
        payload: KnowledgeBaseUpdateRequest,
        user_id: int,
    ) -> KnowledgeBaseResponse:
        await self.verify_project_ownership(project_id, user_id)

        stmt = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.project_id == project_id,
            KnowledgeBase.user_id == user_id,
        )
        result = await self.db.execute(stmt)
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

        await self.db.commit()
        await self.db.refresh(kb)

        count_stmt = select(func.count(Document.id)).where(
            Document.knowledge_base_id == kb.id
        )
        count_res = await self.db.execute(count_stmt)
        doc_count = count_res.scalar_one() or 0

        response_dto = KnowledgeBaseResponse.model_validate(kb)
        response_dto.document_count = doc_count
        return response_dto

    async def delete_knowledge_base(
        self, project_id: UUID, kb_id: UUID, user_id: int
    ) -> None:
        await self.verify_project_ownership(project_id, user_id)

        stmt = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.project_id == project_id,
            KnowledgeBase.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        kb = result.scalar_one_or_none()
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge base not found",
            )

        await self.db.delete(kb)
        await self.db.commit()

    async def get_knowledge_base_documents(
        self, project_id: UUID, kb_id: UUID, user_id: int
    ) -> list[Document]:
        await self.verify_project_ownership(project_id, user_id)
        kb = await self.db.scalar(
            select(KnowledgeBase.id).where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.project_id == project_id,
                KnowledgeBase.user_id == user_id,
            )
        )
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge base not found",
            )
        stmt = select(Document).where(Document.knowledge_base_id == kb_id)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

