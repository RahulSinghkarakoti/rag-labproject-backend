from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIError(BaseModel):
    code: str = Field(
        ...,
        description="Machine-readable error code",
        examples=["NOT_FOUND", "UNAUTHORIZED", "VALIDATION_ERROR"],
    )
    details: Any | None = Field(
        default=None,
        description="Optional error context, field validation errors, or provider details",
    )


class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: T | None = None


class APIErrorResponse(BaseModel):
    success: bool = False
    message: str = Field(..., description="Human-readable error message")
    error: APIError
