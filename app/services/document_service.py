import hashlib
from pathlib import Path
from typing import List
from uuid import UUID, uuid4

import anyio
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import get_redis_pool
from app.core.sse import publisher
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.user import User

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DocumentService:
    """Service handling file upload processing, checksum deduplication,

    file system persistence, and ingestion task queueing.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    async def calculate_checksum(file: UploadFile) -> str:
        """Calculate SHA-256 checksum in chunks without reading the whole file into RAM."""
        sha256_hash = hashlib.sha256()
        await file.seek(0)
        chunk_size = 64 * 1024
        while chunk := await file.read(chunk_size):
            sha256_hash.update(chunk)
        await file.seek(0)
        return sha256_hash.hexdigest()

    async def verify_kb_ownership(
        self, project_id: UUID, kb_id: UUID, user_id: int
    ) -> KnowledgeBase:
        """Verify knowledge base exists and belongs to the given project and user."""
        stmt = (
            select(KnowledgeBase)
            .join(Project, KnowledgeBase.project_id == Project.id)
            .where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.project_id == project_id,
                Project.user_id == user_id,
            )
        )
        result = await self.db.execute(stmt)
        kb = result.scalar_one_or_none()
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge base or project not found",
            )
        return kb

    async def process_uploads(
        self,
        project_id: UUID,
        kb_id: UUID,
        files: List[UploadFile],
        current_user: User,
    ) -> List[Document]:
        """Save uploaded files to disk, deduplicate by checksum, persist DB records,

        publish SSE status events, and enqueue ingestion tasks in Redis.
        """
        await self.verify_kb_ownership(project_id, kb_id, current_user.id)

        redis_pool = get_redis_pool()
        if redis_pool is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Background worker queue is not available",
            )

        processed_documents: List[Document] = []
        new_documents: List[Document] = []

        for file in files:
            checksum = await self.calculate_checksum(file)

            # Check if the file already exists in this Knowledge Base
            stmt = select(Document).where(
                Document.knowledge_base_id == kb_id,
                Document.checksum == checksum,
            )
            result = await self.db.execute(stmt)
            existing_doc = result.scalar_one_or_none()

            if existing_doc:
                processed_documents.append(existing_doc)
                continue

            # Save to disk under KB-scoped directory with UUID prefix to prevent traversal and collisions
            safe_filename = Path(file.filename or "document").name
            if not safe_filename or safe_filename in {".", ".."}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid filename",
                )
            kb_dir = UPLOAD_DIR / str(kb_id)
            kb_dir.mkdir(parents=True, exist_ok=True)
            file_path = kb_dir / f"{uuid4().hex}_{safe_filename}"
            await anyio.Path(file_path).write_bytes(await file.read())

            db_doc = Document(
                knowledge_base_id=kb_id,
                name=safe_filename,
                mime_type=file.content_type,
                size_bytes=file_path.stat().st_size,
                storage_path=str(file_path),
                checksum=checksum,
                status="uploaded",
                metadata_={},
            )
            self.db.add(db_doc)
            await self.db.flush()
            processed_documents.append(db_doc)
            new_documents.append(db_doc)

        await self.db.commit()
        for doc in new_documents:
            await self.db.refresh(doc)

        user_channel = f"ingestion_channel:{current_user.id}"
        for doc in new_documents:
            await publisher.publish(
                data={
                    "document_id": str(doc.id),
                    "knowledge_base_id": str(kb_id),
                    "filename": doc.name,
                    "status": "queued",
                    "message": "Document queued for ingestion",
                },
                event="ingestion_status",
                channel=user_channel,
            )
            await redis_pool.enqueue_job("ingest_document", str(doc.id))

        return processed_documents

    async def get_documents(
        self,
        project_id: UUID,
        kb_id: UUID,
        user_id: int,
    ) -> List[Document]:
        """Verify KB ownership and return all documents in the knowledge base."""
        await self.verify_kb_ownership(project_id, kb_id, user_id)
        stmt = select(Document).where(Document.knowledge_base_id == kb_id)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_document(
        self,
        project_id: UUID,
        kb_id: UUID,
        document_id: UUID,
        user_id: int,
    ) -> Document:
        """Verify KB ownership and return a single document by ID."""
        await self.verify_kb_ownership(project_id, kb_id, user_id)
        stmt = select(Document).where(
            Document.id == document_id,
            Document.knowledge_base_id == kb_id,
        )
        result = await self.db.execute(stmt)
        document = result.scalar_one_or_none()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        return document

    async def delete_document(
        self,
        project_id: UUID,
        kb_id: UUID,
        document_id: UUID,
        user_id: int,
    ) -> None:
        """Verify KB ownership, remove storage file from disk, and delete the DB record."""
        await self.verify_kb_ownership(project_id, kb_id, user_id)

        stmt = select(Document).where(
            Document.id == document_id,
            Document.knowledge_base_id == kb_id,
        )
        result = await self.db.execute(stmt)
        document = result.scalar_one_or_none()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        if document.storage_path:
            file_path = Path(document.storage_path).resolve()
            upload_root = UPLOAD_DIR.resolve()
            if file_path.is_relative_to(upload_root) and file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass

        await self.db.delete(document)
        await self.db.commit()
