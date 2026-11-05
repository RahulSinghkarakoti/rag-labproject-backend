from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.document import DocumentResponse, UploadDocumentsResponse
from app.services.document_service import DocumentService

router = APIRouter(
    tags=["Documents"],
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
    service = DocumentService(db)
    documents = await service.get_documents(
        project_id=project_id,
        kb_id=kb_id,
        user_id=current_user.id,
    )
    return APIResponse(
        success=True,
        message="Documents retrieved successfully",
        data=[DocumentResponse.model_validate(d) for d in documents],
    )


@router.get(
    "/projects/{project_id}/knowledge-bases/{kb_id}/documents/{document_id}",
    response_model=APIResponse[DocumentResponse],
)
async def get_document(
    project_id: UUID,
    kb_id: UUID,
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = DocumentService(db)
    document = await service.get_document(
        project_id=project_id,
        kb_id=kb_id,
        document_id=document_id,
        user_id=current_user.id,
    )
    return APIResponse(
        success=True,
        message="Document retrieved successfully",
        data=DocumentResponse.model_validate(document),
    )


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
    service = DocumentService(db)
    processed_documents = await service.process_uploads(
        project_id=project_id,
        kb_id=kb_id,
        files=files,
        current_user=current_user,
    )
    doc_dtos = [DocumentResponse.model_validate(doc) for doc in processed_documents]
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
    service = DocumentService(db)
    await service.delete_document(
        project_id=project_id,
        kb_id=kb_id,
        document_id=document_id,
        user_id=current_user.id,
    )
    return APIResponse(
        success=True,
        message="Document deleted successfully",
        data=None,
    )
