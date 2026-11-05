from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
from app.schemas.auth import (
    AuthResponse,
    LogoutRequest,
    OAuthLoginRequest,
    RefreshRequest,
    UserResponse,
)
from app.schemas.common import APIResponse
from app.services.auth_service import AuthService

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post("/oauth", response_model=APIResponse[AuthResponse])
async def oauth_login(
    request: Request,
    data: OAuthLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    internal_auth = request.headers.get("X-Internal-Auth") if hasattr(request, "headers") else None
    client = getattr(request, "client", None)
    ip_address = client.host if client else None
    user_agent = request.headers.get("user-agent") if hasattr(request, "headers") else None

    service = AuthService(db)
    auth_data = await service.oauth_login(
        data=data,
        internal_auth=internal_auth,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    return APIResponse(
        message="Login successful.",
        data=auth_data,
    )


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_me(
    current_user: User = Depends(get_current_user),
):
    return APIResponse(
        message="User details retrieved.",
        data=current_user,
    )


@router.post("/refresh", response_model=APIResponse[AuthResponse])
async def refresh_access_token(
    request: Request,
    data: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    client = getattr(request, "client", None)
    ip_address = client.host if client else None
    user_agent = request.headers.get("user-agent") if hasattr(request, "headers") else None

    service = AuthService(db)
    auth_data = await service.refresh_access_token(
        data.refresh_token,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    return APIResponse(
        message="Access token refreshed.",
        data=auth_data,
    )


@router.post("/logout", response_model=APIResponse[dict])
async def logout(
    request: Request,
    data: LogoutRequest,
    db: AsyncSession = Depends(get_db),
):
    client = getattr(request, "client", None)
    ip_address = client.host if client else None
    user_agent = request.headers.get("user-agent") if hasattr(request, "headers") else None

    service = AuthService(db)
    await service.logout(
        data.refresh_token,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    return APIResponse(
        message="Logged out successfully",
        data={},
    )
