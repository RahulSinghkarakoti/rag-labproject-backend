from pydantic import BaseModel, ConfigDict, Field
from uuid import UUID

class PipelineResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None  # Match database nullability
    project_id: UUID
    model_config = ConfigDict(from_attributes=True)  # Allows reading SQLAlchemy models directly

class PipelineRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class PipelineVersionResponse(BaseModel):
    id: UUID
    pipeline_id: UUID
    version: int
    config: dict
    model_config = ConfigDict(from_attributes=True)  # Allows reading SQLAlchemy models directly

class PipelineVersionRequest(BaseModel):
    version: int | None = None
    config: dict
    
