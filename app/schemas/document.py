from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: UUID
    knowledge_base_id: UUID
    name: str
    mime_type: str | None = None
    size_bytes: int | None = None
    status: str
    checksum: str | None = None
    chunk_count: int | None = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class UploadDocumentsResponse(BaseModel):
    project_id: UUID
    knowledge_base_id: UUID
    documents: list[DocumentResponse]
    total: int
    job_id: str | None = None
