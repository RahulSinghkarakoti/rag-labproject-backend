import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class OAuthLoginRequest(BaseModel):
    provider: str
    provider_user_id: str
    email: EmailStr
    name: str | None = None
    avatar_url: str | None = None


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    name: str | None
    avatar_url: str | None
    is_active: bool
    is_verified: bool
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str