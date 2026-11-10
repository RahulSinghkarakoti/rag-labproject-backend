from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.models.auth import RefreshToken
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.pipeline import Pipeline, PipelineVersion
from app.models.project import Project
from app.models.run import Run, RunStep
from app.models.thread import Thread
from app.models.user import User
from app.schemas.auth import OAuthLoginRequest
from app.schemas.knowledge_base import KnowledgeBaseCreateRequest, KnowledgeBaseUpdateRequest
from app.schemas.pipeline import PipelineRequest, PipelineVersionRequest
from app.schemas.project import ProjectRequest
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.pipeline_service import PipelineService
from app.services.project_service import ProjectService
from app.services.run_service import RunService


class MockResult:
    def __init__(self, one=None, all_vals=None, scalar_val=None):
        self._one = one
        self._all_vals = all_vals or []
        self._scalar = scalar_val

    def scalar_one_or_none(self):
        return self._one

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._all_vals


class MockDB:
    def __init__(self, scalar_return=None, execute_return=None):
        self.scalar_return = scalar_return
        self.execute_return = execute_return
        self.added = []
        self.deleted = []
        self.committed = False
        self.flushed = False

    async def scalar(self, stmt):
        if callable(self.scalar_return):
            return self.scalar_return(stmt)
        return self.scalar_return

    async def scalars(self, stmt):
        if callable(self.scalar_return):
            vals = self.scalar_return(stmt)
            return MockResult(all_vals=vals if isinstance(vals, list) else [])
        return MockResult(all_vals=self.scalar_return if isinstance(self.scalar_return, list) else [])

    async def execute(self, stmt):
        if callable(self.execute_return):
            return self.execute_return(stmt)
        if isinstance(self.execute_return, MockResult):
            return self.execute_return
        return MockResult(one=self.execute_return)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushed = True

    async def commit(self):
        self.committed = True

    async def refresh(self, obj):
        pass

    async def delete(self, obj):
        self.deleted.append(obj)


# --- ProjectService Tests ---
@pytest.mark.anyio
async def test_project_service_crud():
    project_id = uuid4()
    user_id = 1
    project = Project(id=project_id, user_id=user_id, name="Test", description="Desc")

    db = MockDB(execute_return=MockResult(one=project, all_vals=[project]))
    service = ProjectService(db)

    # get_projects
    projects = await service.get_projects(user_id=user_id)
    assert len(projects) == 1
    assert projects[0].id == project_id

    # get_project
    p = await service.get_project(project_id, user_id)
    assert p.id == project_id

    # create_project
    new_p = await service.create_project(user_id, ProjectRequest(name="New", description="New Desc"))
    assert new_p.name == "New"
    assert db.committed is True

    # update_project
    updated_p = await service.update_project(project_id, user_id, ProjectRequest(name="Updated"))
    assert updated_p.name == "Updated"

    # delete_project
    await service.delete_project(project_id, user_id)
    assert len(db.deleted) == 1


@pytest.mark.anyio
async def test_project_service_not_found():
    db = MockDB(execute_return=MockResult(one=None))
    service = ProjectService(db)
    with pytest.raises(HTTPException) as exc_info:
        await service.get_project(uuid4(), user_id=1)
    assert exc_info.value.status_code == 404


# --- PipelineService Tests ---
@pytest.mark.anyio
async def test_pipeline_service_crud():
    project_id = uuid4()
    pipeline_id = uuid4()
    version_id = uuid4()
    user_id = 1

    project = Project(id=project_id, user_id=user_id, name="Proj")
    pipeline = Pipeline(id=pipeline_id, project_id=project_id, user_id=user_id, name="Pipe")
    version = PipelineVersion(id=version_id, pipeline_id=pipeline_id, version=1, config={})

    db = MockDB(
        scalar_return=version,
        execute_return=MockResult(one=pipeline, all_vals=[pipeline]),
    )
    service = PipelineService(db)

    # get_project_pipelines
    pipes = await service.get_project_pipelines(project_id, user_id)
    assert len(pipes) == 1

    # create_pipeline
    new_pipe = await service.create_pipeline(project_id, user_id, PipelineRequest(name="P2"))
    assert new_pipe.name == "P2"

    # get_pipeline
    db.scalar_return = pipeline
    p = await service.get_pipeline(pipeline_id, user_id)
    assert p.id == pipeline_id

    # create_pipeline_version
    db.scalar_return = version
    new_v = await service.create_pipeline_version(pipeline_id, user_id, PipelineVersionRequest(config={"k": "v"}))
    assert new_v.version == 2
    assert new_v.config == {"k": "v"}


# --- RunService Tests ---
@pytest.mark.anyio
async def test_run_service_get_and_cancel():
    run_id = uuid4()
    user_id = 1
    step = RunStep(id=uuid4(), run_id=run_id, sequence=1, step_type="retrieval", status="completed")
    run = Run(
        id=run_id,
        thread_id=uuid4(),
        status="running",
        steps=[step],
        output_message_id=uuid4(),
    )
    msg = Message(id=run.output_message_id, thread_id=run.thread_id, role="assistant", content="Partial")

    def scalar_router(stmt):
        stmt_str = str(stmt)
        if "messages" in stmt_str:
            return msg
        return run

    db = MockDB(scalar_return=scalar_router)
    service = RunService(db)

    # get_run_with_steps
    retrieved_run = await service.get_run_with_steps(run_id, user_id)
    assert retrieved_run.id == run_id
    assert len(retrieved_run.steps) == 1

    # cancel_run
    cancelled_run = await service.cancel_run(run_id, user_id)
    assert cancelled_run.status == "cancelled"
    assert db.committed is True
    assert msg.status == "cancelled"
    assert "[User interrupted this response]" in msg.content


# --- ChatService Extended Methods Tests ---
@pytest.mark.anyio
async def test_chat_service_project_threads_and_messages():
    project_id = uuid4()
    thread_id = uuid4()
    user_id = 1

    thread = Thread(id=thread_id, project_id=project_id, user_id=user_id, title="Chat")
    msg = Message(id=uuid4(), thread_id=thread_id, sequence=1, role="user", content="Hi", metadata_={})

    class ChatMockDB:
        def __init__(self):
            pass

        async def scalar(self, stmt):
            stmt_str = str(stmt)
            if "projects" in stmt_str:
                return project_id
            if "threads" in stmt_str:
                return thread
            return None

        async def scalars(self, stmt):
            stmt_str = str(stmt)
            if "messages" in stmt_str:
                return MockResult(all_vals=[msg])
            return MockResult(all_vals=[thread])

    db = ChatMockDB()
    service = ChatService(db)

    threads = await service.get_project_threads(project_id, user_id)
    assert len(threads) == 1
    assert threads[0].id == thread_id

    messages = await service.get_thread_messages(thread_id, user_id)
    assert len(messages) == 1
    assert messages[0].content == "Hi"


# --- AuthService and AuthLog Tests ---
@pytest.mark.anyio
async def test_auth_service_logs_oauth_and_tokens():
    from app.models.audit_log import AuthLog
    from app.models.auth import OAuthAccount
    from app.core.security import hash_refresh_token
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    user = User(
        id=1,
        email="test@example.com",
        name="Test User",
        is_active=True,
        is_verified=True,
        created_at=now,
        updated_at=now,
    )
    oauth = OAuthAccount(id=1, user_id=1, provider="google", provider_user_id="g123")

    # DB returning user and oauth_account
    class AuthMockDB(MockDB):
        async def execute(self, stmt):
            stmt_str = str(stmt)
            if "oauth_accounts" in stmt_str:
                return MockResult(one=oauth)
            if "users" in stmt_str:
                return MockResult(one=user)
            return MockResult(one=None)

    db = AuthMockDB()
    service = AuthService(db)

    # 1. OAuth login existing user
    from app.core.config import settings
    settings.NEXTAUTH_INTERNAL_SECRET = "testsecret"
    res = await service.oauth_login(
        data=OAuthLoginRequest(
            provider="google",
            provider_user_id="g123",
            email="test@example.com",
            name="Test User",
        ),
        internal_auth="testsecret",
        ip_address="127.0.0.1",
        user_agent="pytest-agent",
    )
    assert res.access_token is not None
    # Check that AuthLog was added
    logs = [obj for obj in db.added if isinstance(obj, AuthLog)]
    assert len(logs) == 1
    assert logs[0].event == "oauth_login"
    assert logs[0].user_id == 1
    assert logs[0].ip_address == "127.0.0.1"
    assert logs[0].user_agent == "pytest-agent"

    # 2. Token refresh logs
    token_str = "refresh-token-xyz"
    stored_rt = RefreshToken(
        id=1,
        user_id=1,
        token_hash=hash_refresh_token(token_str),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )

    class RefreshMockDB(MockDB):
        async def execute(self, stmt):
            stmt_str = str(stmt)
            if "refresh_tokens" in stmt_str:
                return MockResult(one=stored_rt)
            if "users" in stmt_str:
                return MockResult(one=user)
            return MockResult(one=None)

    r_db = RefreshMockDB()
    r_service = AuthService(r_db)
    r_res = await r_service.refresh_access_token(
        refresh_token=token_str,
        ip_address="192.168.1.1",
        user_agent="mobile-app",
    )
    assert r_res.access_token is not None
    r_logs = [obj for obj in r_db.added if isinstance(obj, AuthLog)]
    assert len(r_logs) == 1
    assert r_logs[0].event == "token_refresh"
    assert r_logs[0].ip_address == "192.168.1.1"

    # 3. Logout logs
    l_db = RefreshMockDB()
    l_service = AuthService(l_db)
    await l_service.logout(
        refresh_token=token_str,
        ip_address="192.168.1.1",
        user_agent="mobile-app",
    )
    l_logs = [obj for obj in l_db.added if isinstance(obj, AuthLog)]
    assert len(l_logs) == 1
    assert l_logs[0].event == "logout"
    assert l_logs[0].user_id == 1


# --- APIError and Exception Handlers Tests ---
@pytest.mark.anyio
async def test_exception_handlers_use_apierror_response():
    import json
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from app.core.exceptions import AppException
    from app.core.exception_handlers import (
        app_exception_handler,
        http_exception_handler,
        unhandled_exception_handler,
    )

    req = SimpleNamespace()

    # 1. StarletteHTTPException handler
    http_exc = StarletteHTTPException(status_code=404, detail="Resource not found")
    res = await http_exception_handler(req, http_exc)
    assert res.status_code == 404
    data = json.loads(res.body.decode())
    assert data["success"] is False
    assert data["message"] == "Resource not found"
    assert data["error"]["code"] == "NOT_FOUND"

    # 2. AppException handler
    app_exc = AppException("Custom failed", status_code=400, code="CUSTOM_CODE", details={"hint": "retry"})
    res2 = await app_exception_handler(req, app_exc)
    assert res2.status_code == 400
    data2 = json.loads(res2.body.decode())
    assert data2["success"] is False
    assert data2["message"] == "Custom failed"
    assert data2["error"]["code"] == "CUSTOM_CODE"
    assert data2["error"]["details"] == {"hint": "retry"}

    # 3. Unhandled Exception handler
    res3 = await unhandled_exception_handler(req, Exception("Crash"))
    assert res3.status_code == 500
    data3 = json.loads(res3.body.decode())
    assert data3["success"] is False
    assert data3["message"] == "An unexpected error occurred."
    assert data3["error"]["code"] == "INTERNAL_SERVER_ERROR"

