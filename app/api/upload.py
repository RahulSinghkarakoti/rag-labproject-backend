import hashlib
from pathlib import Path
import shutil
from typing import List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Path as PathParam, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.user import User
from app.core.redis import get_redis_pool
from app.core.sse import publisher
from app.schemas.common import APIResponse
from app.schemas.document import DocumentResponse, UploadDocumentsResponse

router = APIRouter(
    prefix="/api",
    tags=["Uploads & Documents"],
)

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def calculate_checksum(file: UploadFile) -> str:
    """Calculates SHA-256 checksum in chunks without reading the entire file into RAM."""
    sha256_hash = hashlib.sha256()
    await file.seek(0)
    chunk_size = 64 * 1024
    while chunk := await file.read(chunk_size):
        sha256_hash.update(chunk)
    await file.seek(0)  # Reset pointer for subsequent disk reads
    return sha256_hash.hexdigest()


async def _verify_kb_ownership(
    project_id: UUID, kb_id: UUID, user_id: int, db: AsyncSession
) -> KnowledgeBase:
    stmt = (
        select(KnowledgeBase)
        .join(Project, KnowledgeBase.project_id == Project.id)
        .where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.project_id == project_id,
            Project.user_id == user_id,
        )
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base or project not found",
        )
    return kb


@router.post(
    "/projects/{project_id}/knowledge-bases/{kb_id}/upload-files",
    response_model=APIResponse[UploadDocumentsResponse],
)
async def upload_multiple_files(
    project_id: UUID,
    kb_id: UUID,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_kb_ownership(project_id, kb_id, current_user.id, db)

    processed_documents: List[Document] = []

    for file in files:
        checksum = await calculate_checksum(file)

        # Check if the file already exists in this Knowledge Base
        stmt = select(Document).where(
            Document.knowledge_base_id == kb_id,
            Document.checksum == checksum,
        )
        result = await db.execute(stmt)
        existing_doc = result.scalar_one_or_none()

        if existing_doc:
            processed_documents.append(existing_doc)
            continue

        # Save to disk under KB-scoped directory with UUID prefix to prevent traversal and collisions
        import anyio
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
        db.add(db_doc)
        await db.flush()
        processed_documents.append(db_doc)

    await db.commit()
    for doc in processed_documents:
        await db.refresh(doc)

    doc_dtos = [DocumentResponse.model_validate(doc) for doc in processed_documents]

    redis_pool = get_redis_pool()
    if redis_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background worker queue is not available",
        )

    user_channel = f"ingestion_channel:{current_user.id}"
    for doc in processed_documents:
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
        await redis_pool.enqueue_job(
            "ingest_document",
            str(doc.id),
        )

    return APIResponse(
        success=True,
        message=f"{len(doc_dtos)} file(s) processed successfully",
        data=UploadDocumentsResponse(
            project_id=project_id,
            knowledge_base_id=kb_id,
            documents=doc_dtos,
            total=len(files),
        ),
    )


@router.delete(
    "/projects/{project_id}/knowledge-bases/{kb_id}/documents/{document_id}",
    response_model=APIResponse[None],
)
async def delete_document(
    project_id: UUID,
    kb_id: UUID,
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_kb_ownership(project_id, kb_id, current_user.id, db)

    stmt = select(Document).where(
        Document.id == document_id,
        Document.knowledge_base_id == kb_id,
    )
    result = await db.execute(stmt)
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

    await db.delete(document)
    await db.commit()

    return APIResponse(
        success=True,
        message="Document deleted successfully",
        data=None,
    )