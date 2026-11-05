import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
)
from app.models.auth import OAuthAccount, RefreshToken
from app.models.audit_log import AuthLog
from app.models.user import User
from app.schemas.auth import AuthResponse, OAuthLoginRequest


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _log_event(
        self,
        event: str,
        user_id: int | None = None,
        provider: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        log_metadata: dict | None = None,
    ) -> AuthLog:
        entry = AuthLog(
            user_id=user_id,
            event=event,
            provider=provider,
            ip_address=ip_address,
            user_agent=user_agent,
            log_metadata=log_metadata or {},
        )
        self.db.add(entry)
        return entry

    async def oauth_login(
        self,
        data: OAuthLoginRequest,
        internal_auth: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthResponse:
        if data.provider not in {"google", "github"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported authentication provider",
            )

        internal_auth_val = internal_auth or ""
        if not settings.NEXTAUTH_INTERNAL_SECRET or not secrets.compare_digest(
            internal_auth_val, settings.NEXTAUTH_INTERNAL_SECRET
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )

        # 1. Find existing OAuth account
        result = await self.db.execute(
            select(OAuthAccount).where(
                OAuthAccount.provider == data.provider,
                OAuthAccount.provider_user_id == data.provider_user_id,
            )
        )
        oauth_account = result.scalar_one_or_none()
        is_new_user = False

        # 2. Existing OAuth account
        if oauth_account:
            result = await self.db.execute(
                select(User).where(User.id == oauth_account.user_id)
            )
            user = result.scalar_one_or_none()
        else:
            # 3. Check whether email already exists
            result = await self.db.execute(
                select(User).where(User.email == data.email)
            )
            user = result.scalar_one_or_none()
            if user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="An account with this email already exists. Please log in with your original provider.",
                )

            # 4. Create user if necessary
            user = User(
                email=data.email,
                name=data.name,
                avatar_url=data.avatar_url,
                is_active=True,
                is_verified=True,
            )
            self.db.add(user)
            await self.db.flush()

            # 5. Create provider connection
            oauth_account = OAuthAccount(
                user_id=user.id,
                provider=data.provider,
                provider_user_id=data.provider_user_id,
            )
            self.db.add(oauth_account)
            is_new_user = True

        # 6. Generate tokens
        access_token = create_access_token(user.id)
        refresh_token = create_refresh_token()
        refresh_token_record = RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(refresh_token),
            expires_at=(
                datetime.now(timezone.utc)
                + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
            ),
        )
        self.db.add(refresh_token_record)

        # 7. Record auth audit log
        self._log_event(
            user_id=user.id,
            event="oauth_signup" if is_new_user else "oauth_login",
            provider=data.provider,
            ip_address=ip_address,
            user_agent=user_agent,
            log_metadata={
                "provider_user_id": data.provider_user_id,
                "email": data.email,
                "is_new_user": is_new_user,
            },
        )

        await self.db.commit()
        await self.db.refresh(user)

        return AuthResponse(
            user=user,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    async def refresh_access_token(
        self,
        refresh_token: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthResponse:
        token_hash = hash_refresh_token(refresh_token)
        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        stored_token = result.scalar_one_or_none()
        if not stored_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        now = datetime.now(timezone.utc)
        if stored_token.revoked_at is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked",
            )
        if stored_token.expires_at <= now:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired",
            )

        result = await self.db.execute(
            select(User).where(
                User.id == stored_token.user_id,
                User.is_active.is_(True),
            )
        )
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Revoke old refresh token
        stored_token.revoked_at = now

        # Create new tokens
        access_token = create_access_token(user.id)
        new_refresh_token = create_refresh_token()
        new_refresh_record = RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(new_refresh_token),
            expires_at=(now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)),
        )
        self.db.add(new_refresh_record)

        # Record token refresh in audit log
        self._log_event(
            user_id=user.id,
            event="token_refresh",
            provider=None,
            ip_address=ip_address,
            user_agent=user_agent,
            log_metadata={"action": "rotate_refresh_token"},
        )

        await self.db.commit()

        return AuthResponse(
            user=user,
            access_token=access_token,
            refresh_token=new_refresh_token,
        )

    async def logout(
        self,
        refresh_token: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        token_hash = hash_refresh_token(refresh_token)
        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        stored_token = result.scalar_one_or_none()
        if stored_token:
            stored_token.revoked_at = datetime.now(timezone.utc)
            self._log_event(
                user_id=stored_token.user_id,
                event="logout",
                provider=None,
                ip_address=ip_address,
                user_agent=user_agent,
                log_metadata={"action": "revoke_refresh_token"},
            )
            await self.db.commit()
