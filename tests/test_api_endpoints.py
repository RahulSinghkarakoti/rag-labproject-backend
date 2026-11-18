from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.documents import get_document
from app.api.projects import get_project
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project


class FakeResult:
    def __init__(self, one=None, all_vals=None):
        self._one = one
        self._all_vals = all_vals or []

    def scalar_one_or_none(self):
        return self._one

    def scalars(self):
        return self

    def all(self):
        return self._all_vals


class FakeDB:
    def __init__(self, execute_return=None):
        self.execute_return = execute_return

    async def execute(self, stmt):
        if callable(self.execute_return):
            return self.execute_return(stmt)
        return FakeResult(one=self.execute_return)


@pytest.mark.anyio
async def test_get_project_success():
    project_id = uuid4()
    user = SimpleNamespace(id=1)
    fake_project = Project(id=project_id, user_id=1, name="My Project", description="Test")

    db = FakeDB(execute_return=fake_project)
    response = await get_project(project_id=project_id, current_user=user, db=db)

    assert response.success is True
    assert response.data.id == project_id
    assert response.data.name == "My Project"


@pytest.mark.anyio
async def test_get_project_not_found():
    project_id = uuid4()
    user = SimpleNamespace(id=1)

    db = FakeDB(execute_return=None)
    with pytest.raises(HTTPException) as exc_info:
        await get_project(project_id=project_id, current_user=user, db=db)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Project not found"


@pytest.mark.anyio
async def test_get_document_success():
    project_id = uuid4()
    kb_id = uuid4()
    doc_id = uuid4()
    user = SimpleNamespace(id=1)

    fake_kb = KnowledgeBase(id=kb_id, project_id=project_id, user_id=1)
    fake_doc = Document(
        id=doc_id,
        knowledge_base_id=kb_id,
        name="doc.pdf",
        status="completed",
        size_bytes=1024,
    )

    def execute_handler(stmt):
        stmt_str = str(stmt)
        if "knowledge_bases" in stmt_str:
            return FakeResult(one=fake_kb)
        return FakeResult(one=fake_doc)

    db = FakeDB(execute_return=execute_handler)
    response = await get_document(
        project_id=project_id,
        kb_id=kb_id,
        document_id=doc_id,
        current_user=user,
        db=db,
    )

    assert response.success is True
    assert response.data.id == doc_id
    assert response.data.name == "doc.pdf"


@pytest.mark.anyio
async def test_get_document_not_found():
    project_id = uuid4()
    kb_id = uuid4()
    doc_id = uuid4()
    user = SimpleNamespace(id=1)

    fake_kb = KnowledgeBase(id=kb_id, project_id=project_id, user_id=1)

    def execute_handler(stmt):
        stmt_str = str(stmt)
        if "knowledge_bases" in stmt_str:
            return FakeResult(one=fake_kb)
        return FakeResult(one=None)

    db = FakeDB(execute_return=execute_handler)
    with pytest.raises(HTTPException) as exc_info:
        await get_document(
            project_id=project_id,
            kb_id=kb_id,
            document_id=doc_id,
            current_user=user,
            db=db,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found"


def test_v1_mount_routes():
    from fastapi import FastAPI

    from app.api.router import register_routers

    app = FastAPI()
    register_routers(app)
    openapi = app.openapi()
    paths = set(openapi["paths"].keys())

    api_paths = {p for p in paths if p.startswith("/api/") and not p.startswith("/api/v1/")}
    v1_paths = {p for p in paths if p.startswith("/api/v1/")}

    assert len(v1_paths) > 0, "No /api/v1 routes found"
    assert len(api_paths) == 0, f"Found unversioned /api routes that should be disabled: {api_paths}"


def test_message_response_strips_citations_from_metadata():
    from app.schemas.thread import MessageResponse

    msg_id = uuid4()
    thread_id = uuid4()
    raw_data = {
        "id": msg_id,
        "thread_id": thread_id,
        "parent_message_id": None,
        "role": "assistant",
        "content": "Test content",
        "content_json": None,
        "sequence": 1,
        "status": "completed",
        "metadata": {
            "run_id": "dce755c6-70c5-4167-9341-d60cf8d6bede",
            "created_at": "2026-11-18T15:43:01.910849+00:00",
            "citations": [
                {"name": "01_product_overview.md", "index": 1, "mime_type": "text/markdown"},
                {"name": "01_product_overview.md", "index": 2, "mime_type": "text/markdown"},
            ],
            "context_window": {"limit": 8192, "tokens": 622},
        },
    }

    validated = MessageResponse.model_validate(raw_data)
    assert "citations" not in validated.metadata
    assert validated.metadata["run_id"] == "dce755c6-70c5-4167-9341-d60cf8d6bede"
    assert validated.metadata["context_window"] == {"limit": 8192, "tokens": 622}

    # Also test from ORM Message object with metadata_ attribute
    from app.models.message import Message

    orm_msg = Message(
        id=msg_id,
        thread_id=thread_id,
        role="assistant",
        content="ORM content",
        sequence=2,
        status="completed",
        metadata_={
            "run_id": "run-123",
            "citations": [{"name": "test.pdf", "index": 1}],
            "created_at": "2026-11-18T15:43:01Z",
        },
    )
    validated_orm = MessageResponse.model_validate(orm_msg)
    assert "citations" not in validated_orm.metadata
    assert validated_orm.metadata["run_id"] == "run-123"



