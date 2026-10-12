from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class APIError(BaseModel):
    code: str = Field(..., examples=["INVALID_TOKEN"])
    details: Any | None = None


class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: T | None = None


class APIErrorResponse(BaseModel):
    success: bool = False
    message: str
    error: APIError