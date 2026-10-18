from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class RunStepResponse(BaseModel):
    id: UUID
    run_id: UUID
    step_type: str
    sequence: int
    status: str
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")

    model_config = ConfigDict(from_attributes=True)


class RunDetailsResponse(BaseModel):
    id: UUID
    thread_id: UUID
    input_message_id: Optional[UUID] = None
    output_message_id: Optional[UUID] = None
    run_type: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[Dict[str, Any]] = None
    steps: List[RunStepResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
