import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException,Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.api.dependencies import get_current_user
from app.database.database import get_db
from app.models.user import User
from app.models.auth import OAuthAccount, RefreshToken
from app.schemas.auth import (
    AuthResponse,
    OAuthLoginRequest,
    RefreshRequest,
    LogoutRequest,
    UserResponse,
)
 

from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
)
from app.schemas.common import APIResponse

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post("/oauth", response_model=APIResponse[AuthResponse])
async def oauth_login(
    request: Request,
    data: OAuthLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    if data.provider not in {"google", "github"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported authentication provider",
        )
    if (
        request.headers.get("X-Internal-Auth")
        != settings.NEXTAUTH_INTERNAL_SECRET
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )
    # 1. Find existing OAuth account
    result = await db.execute(
        select(OAuthAccount).where(
            OAuthAccount.provider == data.provider,
            OAuthAccount.provider_user_id == data.provider_user_id,
        )
    )
    oauth_account = result.scalar_one_or_none()
    # 2. Existing OAuth account
    if oauth_account:
        result = await db.execute(
            select(User).where(
                User.id == oauth_account.user_id
            )
        )
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(
                status_code=500,
                detail="OAuth account has no user",
            )
    else:
        # 3. Check whether email already exists
        result = await db.execute(
            select(User).where(
                User.email == data.email
            )
        )
        user = result.scalar_one_or_none()
        # 4. Create user if necessary
        if not user:
            user = User(
                # id=uuid.uuid4(),
                email=data.email,
                name=data.name,
                avatar_url=data.avatar_url,
                is_active=True,
                is_verified=True,
            )
            db.add(user)
            await db.flush()
        # 5. Create provider connection
        oauth_account = OAuthAccount(
            user_id=user.id,
            provider=data.provider,
            provider_user_id=data.provider_user_id,
        )
        db.add(oauth_account)
    # 6. Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token()
    refresh_token_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(refresh_token),
        expires_at=(
            datetime.now(timezone.utc)
            + timedelta(days=30)
        ),
    )
    db.add(refresh_token_record)
    await db.commit()
    await db.refresh(user)
    return APIResponse(
        message="Login Successfull.",
        data=AuthResponse(
        user=user,
        access_token=access_token,
        refresh_token=refresh_token,
    ))


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_me(
    current_user: User = Depends(get_current_user),
):
    return APIResponse(
        message="User details retrieved.",
        data=current_user
    )



@router.post("/refresh", response_model=APIResponse[AuthResponse])
async def refresh_access_token(
    data: RefreshRequest,
    db: AsyncSession = Depends(get_db),
): 
    token_hash = hash_refresh_token(
        data.refresh_token
    )
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash
        )
    )
    stored_token = result.scalar_one_or_none()
    if not stored_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token",
        )
    now = datetime.now(timezone.utc)
    if stored_token.revoked_at is not None:
        raise HTTPException(
            status_code=401,
            detail="Refresh token has been revoked",
        )
    if stored_token.expires_at <= now:
        raise HTTPException(
            status_code=401,
            detail="Refresh token has expired",
        )
    result = await db.execute(
        select(User).where(
            User.id == stored_token.user_id,
            User.is_active.is_(True),
        )
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found",
        )
    # Revoke old refresh token
    stored_token.revoked_at = now
    # Create new tokens
    access_token = create_access_token(user.id)
    new_refresh_token = create_refresh_token()
    new_refresh_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(
            new_refresh_token
        ),
        expires_at=(
            now + timedelta(days=30)
        ),
    )
    db.add(new_refresh_record)
    await db.commit()
    return APIResponse(
        message="Access token refreshed.",
        data=AuthResponse(
            user=user,
            access_token=access_token,
            refresh_token=new_refresh_token,
        )
    )


@router.post("/logout",response_model=APIResponse[dict])
async def logout(
    data: LogoutRequest,
    db: AsyncSession = Depends(get_db),
):
    token_hash = hash_refresh_token(
        data.refresh_token
    )
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash
        )
    )
    stored_token = result.scalar_one_or_none()
    if stored_token:
        stored_token.revoked_at = datetime.now(
            timezone.utc
        )
        await db.commit()
    return APIResponse(
        message="Logged out successfully",
        data={}
    )