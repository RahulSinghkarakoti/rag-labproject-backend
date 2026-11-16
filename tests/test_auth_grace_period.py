from datetime import UTC, datetime, timedelta
import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.security import hash_refresh_token
from app.models.audit_log import AuthLog
from app.models.auth import RefreshToken
from app.models.user import User
from app.services.auth_service import AuthService
from tests.test_new_services import MockDB, MockResult


@pytest.mark.anyio
async def test_refresh_token_within_grace_period_succeeds():
    now = datetime.now(UTC)
    user = User(
        id=42,
        email="grace@example.com",
        name="Grace User",
        is_active=True,
        is_verified=True,
        created_at=now,
        updated_at=now,
    )

    token_str = "original-secret-refresh-token"
    # Token was revoked 5 seconds ago (well within 30s grace window)
    stored_rt = RefreshToken(
        id=10,
        user_id=42,
        token_hash=hash_refresh_token(token_str),
        expires_at=now + timedelta(days=7),
        revoked_at=now - timedelta(seconds=5),
        created_at=now - timedelta(minutes=15),
    )

    class GraceMockDB(MockDB):
        async def execute(self, stmt):
            stmt_str = str(stmt)
            if "refresh_tokens" in stmt_str:
                return MockResult(one=stored_rt)
            if "users" in stmt_str:
                return MockResult(one=user)
            return MockResult(one=None)

    db = GraceMockDB()
    service = AuthService(db)

    res = await service.refresh_access_token(
        refresh_token=token_str,
        ip_address="127.0.0.1",
        user_agent="concurrent-client",
    )

    # Should succeed!
    assert res.access_token is not None
    assert res.refresh_token is not None
    assert res.user.id == 42

    # Verify audit log was recorded with token_refresh_grace
    logs = [obj for obj in db.added if isinstance(obj, AuthLog)]
    assert len(logs) == 1
    assert logs[0].event == "token_refresh_grace"
    assert logs[0].log_metadata.get("action") == "rotate_refresh_token_grace"
    assert "grace_seconds_elapsed" in logs[0].log_metadata


@pytest.mark.anyio
async def test_refresh_token_outside_grace_period_fails():
    now = datetime.now(UTC)
    user = User(
        id=42,
        email="grace@example.com",
        name="Grace User",
        is_active=True,
        is_verified=True,
    )

    token_str = "expired-grace-refresh-token"
    # Token was revoked 45 seconds ago (outside 30s grace window)
    stored_rt = RefreshToken(
        id=11,
        user_id=42,
        token_hash=hash_refresh_token(token_str),
        expires_at=now + timedelta(days=7),
        revoked_at=now - timedelta(seconds=45),
        created_at=now - timedelta(minutes=16),
    )

    class ExpiredGraceMockDB(MockDB):
        async def execute(self, stmt):
            stmt_str = str(stmt)
            if "refresh_tokens" in stmt_str:
                return MockResult(one=stored_rt)
            if "users" in stmt_str:
                return MockResult(one=user)
            return MockResult(one=None)

    db = ExpiredGraceMockDB()
    service = AuthService(db)

    with pytest.raises(HTTPException) as exc_info:
        await service.refresh_access_token(
            refresh_token=token_str,
            ip_address="127.0.0.1",
            user_agent="late-client",
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Refresh token has been revoked"
