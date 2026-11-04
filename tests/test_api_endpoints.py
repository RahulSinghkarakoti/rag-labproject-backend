from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.documents import get_document, get_knowledge_base_documents
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


def test_dual_mount_route_parity():
    from fastapi import FastAPI
    from app.api.router import register_routers

    app = FastAPI()
    register_routers(app)
    openapi = app.openapi()
    paths = set(openapi["paths"].keys())

    api_paths = {p for p in paths if p.startswith("/api/") and not p.startswith("/api/v1/")}
    v1_paths = {p for p in paths if p.startswith("/api/v1/")}

    assert len(api_paths) > 0, "No /api routes found"
    assert len(api_paths) == len(v1_paths), f"Route count mismatch: {len(api_paths)} vs {len(v1_paths)}"

    for p in api_paths:
        v1_equivalent = f"/api/v1{p[4:]}"
        assert v1_equivalent in v1_paths, f"Missing {v1_equivalent} for {p}"

