from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from app.core.model_config import validate_embedding_model


class ChunkingConfig(BaseModel):
    strategy: Literal["recursive", "markdown", "semantic", "character"] = "recursive"
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    # dimension: Optional[int] = 384
    _validate_model = field_validator("model")(validate_embedding_model)



# class IndexingConfig(BaseModel):
#     retriever: Literal["similarity", "mmr", "similarity-score-threshold"] = "similarity"
#     top_k: int = Field(default=5, ge=1, le=20)
#     similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
#     reranking: bool = True
#     citations: bool = True


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    chunking_config: Optional[ChunkingConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    # indexing_config: Optional[IndexingConfig] = None


class KnowledgeBaseUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    chunking_config: Optional[ChunkingConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    # indexing_config: Optional[IndexingConfig] = None


class KnowledgeBaseResponse(BaseModel):
    id: UUID
    project_id: UUID
    name: str
    description: Optional[str] = None
    chunking_config: Dict[str, Any] = Field(default_factory=dict)
    embedding_config: Dict[str, Any] = Field(default_factory=dict)
    indexing_config: Dict[str, Any] = Field(default_factory=dict)
    document_count: Optional[int] = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
