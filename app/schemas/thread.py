# app/schemas/thread.py (or inside app/api/threads.py)
from typing import Any, Dict, Optional, List, Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from app.core.model_config import (
    validate_embedding_model,
    validate_generation_model,
    validate_reranking_model,
)
from app.schemas.knowledge_base import KnowledgeBaseResponse
from uuid import UUID
from datetime import datetime


class QueryTransformationConfig(BaseModel):
    mode: Literal[
        "none",
        "rewrite",
        "hyde",
        "multi_query",
        "step_back",
        "sub_queries",
    ] = "none"
    query_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of queries/hypotheses to generate for multi_query, sub_queries, or hyde",
    )
    include_original: bool = Field(
        default=True,
        description="Whether to include the original user query alongside transformed queries",
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Optional custom prompt instructions for the transformation technique",
    )

     

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_mode(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            mode = data.get("mode")
            hyde_enabled = data.get("hyde_enabled")
            multi_query_count = data.get("multi_query_count")

            # Resolve legacy boolean and count fields unambiguously
            if mode is None or mode == "none":
                if hyde_enabled:
                    data["mode"] = "hyde"
                elif multi_query_count and int(multi_query_count) > 0:
                    data["mode"] = "multi_query"

            if multi_query_count is not None and "query_count" not in data:
                try:
                    data["query_count"] = max(1, min(int(multi_query_count), 10))
                except (ValueError, TypeError):
                    pass

        return data


class RetrievalConfig(BaseModel):
    strategy: Literal["hybrid", "dense", "sparse", "mmr"] = "hybrid"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=8, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    similarity_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    sparse_model: str = Field(default="bm25", min_length=1, max_length=100)
    dense_model: str = Field(default="all-MiniLM-L6-v2", min_length=1, max_length=100)

    _validate_dense_model = field_validator("dense_model")(validate_embedding_model)

    model_config = ConfigDict(extra="forbid")


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: str = Field(default="BAAI/bge-reranker-base", min_length=1, max_length=200)
    top_n: int = Field(default=10, ge=1, le=200)
    top_k: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=256)
    score_threshold: float | None = None
    return_documents: bool = True

    _validate_model = field_validator("model")(validate_reranking_model)

    model_config = ConfigDict(extra="forbid")


class GenerationConfig(BaseModel):
    provider: Literal["openai", "groq", "gemini"] = "groq"
    llm_model: str = Field(
        default="gpt-oss-120b",
        min_length=1,
        max_length=200,
        validation_alias=AliasChoices("llm_model", "model"),
        description="Model name/identifier",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=32000)
    stream: bool = True
    citation_enforced: bool = False

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def model(self) -> str:
        return self.llm_model

    @model_validator(mode="before")
    @classmethod
    def resolve_provider_and_model(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            model_val = data.get("llm_model") or data.get("model") or "gpt-oss-120b"
            provider = data.get("provider")

            if not provider or provider == "none":
                model_str = str(model_val).lower()
                if "gemini" in model_str:
                    data["provider"] = "gemini"
                elif any(m in model_str for m in ["llama", "mixtral", "gemma", "deepseek", "gpt-oss", "groq"]):
                    data["provider"] = "groq"
                else:
                    data["provider"] = "openai"
        return data

    @model_validator(mode="after")
    def validate_configured_model(self):
        validate_generation_model(self.provider, self.llm_model)
        return self


class ThreadConfig(BaseModel):
    query_transformation: QueryTransformationConfig = Field(default_factory=QueryTransformationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    model_config = ConfigDict(extra="forbid")


class ThreadCreate(BaseModel):
    title: str
    knowledge_base_ids: List[UUID] = Field(default_factory=list)
    config: Optional[ThreadConfig] = None

class ThreadUpdate(BaseModel): 
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    knowledge_base_ids: Optional[List[UUID]] = None
    config: Optional[ThreadConfig] = None

class ThreadResponse(BaseModel):
    id: UUID
    user_id: int
    project_id: Optional[UUID]
    title: Optional[str]
    status: str
    knowledge_bases: List[KnowledgeBaseResponse] = Field(default_factory=list)
    metadata: Optional[dict] = None
    created_at: datetime
    updated_at: datetime  
    metadata: Optional[dict] = Field(default=None, validation_alias="metadata_")
    model_config = ConfigDict(from_attributes=True)

class MessageCreate(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    id: UUID
    thread_id: UUID
    parent_message_id: UUID | None = None
    role: str
    content: str | None = None
    content_json: Any | None = None
    sequence: int
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")

    model_config = ConfigDict(from_attributes=True)
