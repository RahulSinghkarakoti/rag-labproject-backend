from pydantic import BaseModel, ConfigDict
from uuid import UUID

class ProjectResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None  # Match database nullability
    model_config = ConfigDict(from_attributes=True)  # Allows reading SQLAlchemy models directly

class ProjectRequest(BaseModel):
    name: str
    description: str | None = None