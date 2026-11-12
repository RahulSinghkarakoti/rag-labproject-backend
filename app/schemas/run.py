from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RunStepResponse(BaseModel):
    id: UUID
    run_id: UUID
    step_type: str
    sequence: int
    status: str
    input: Any | None = None
    output: Any | None = None
    config: Any | None = None
    error: Any | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")

    model_config = ConfigDict(from_attributes=True)


class RunDetailsResponse(BaseModel):
    id: UUID
    thread_id: UUID
    input_message_id: UUID | None = None
    output_message_id: UUID | None = None
    run_type: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: Any | None = None
    steps: list[RunStepResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
