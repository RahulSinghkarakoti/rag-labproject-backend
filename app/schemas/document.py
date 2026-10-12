from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: UUID
    knowledge_base_id: UUID
    name: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    status: str
    checksum: Optional[str] = None
    chunk_count: Optional[int] = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class UploadDocumentsResponse(BaseModel):
    project_id: UUID
    knowledge_base_id: UUID
    documents: list[DocumentResponse]
    total: int
    job_id: Optional[str] = None
