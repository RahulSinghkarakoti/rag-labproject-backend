from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from app.api.auth import oauth_login
from app.api.knowledge_base import get_knowledge_base_documents, update_knowledge_base
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.run import RetrievalResult
from app.rag.run_manager import RunManager
from app.schemas.auth import OAuthLoginRequest
from app.schemas.knowledge_base import ChunkingConfig, KnowledgeBaseUpdateRequest


def test_chunking_config_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller than chunk_size"):
        ChunkingConfig(chunk_size=200, chunk_overlap=200)


@pytest.mark.anyio
async def test_get_knowledge_base_documents_rejects_unowned_kb():
    project_id = uuid4()
    foreign_kb_id = uuid4()
    user = SimpleNamespace(id=1)

    class FakeResult:
        def __init__(self, value):
            self._value = value

        def scalar_one_or_none(self):
            return self._value

    class FakeDB:
        async def execute(self, stmt):
            return FakeResult(Project(id=project_id, user_id=1, name="Proj"))

        async def scalar(self, stmt):
            # Foreign KB does not belong to (project_id, user_id)
            return None

    with pytest.raises(HTTPException) as exc_info:
        await get_knowledge_base_documents(
            project_id=project_id,
            kb_id=foreign_kb_id,
            current_user=user,
            db=FakeDB(),
        )
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_update_knowledge_base_succeeds_and_returns_doc_count():
    project_id = uuid4()
    kb_id = uuid4()
    user = SimpleNamespace(id=1)
    kb = KnowledgeBase(
        id=kb_id,
        project_id=project_id,
        user_id=1,
        name="Old Name",
        description="Old",
        chunking_config={"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
        embedding_config={"model": "all-MiniLM-L6-v2"},
        indexing_config={},
        metadata_={},
    )

    class FakeResult:
        def __init__(self, one=None, scalar=None):
            self._one = one
            self._scalar = scalar

        def scalar_one_or_none(self):
            return self._one

        def scalar_one(self):
            return self._scalar

    class FakeDB:
        def __init__(self):
            self.calls = 0

        async def execute(self, stmt):
            self.calls += 1
            if self.calls == 1:
                return FakeResult(one=Project(id=project_id, user_id=1, name="Proj"))
            if self.calls == 2:
                return FakeResult(one=kb)
            return FakeResult(scalar=3)

        async def commit(self):
            pass

        async def refresh(self, obj):
            pass

    resp = await update_knowledge_base(
        project_id=project_id,
        kb_id=kb_id,
        payload=KnowledgeBaseUpdateRequest(name="New Name"),
        current_user=user,
        db=FakeDB(),
    )
    assert resp.success is True
    assert resp.data.name == "New Name"
    assert resp.data.document_count == 3


@pytest.mark.anyio
async def test_oauth_login_rejects_bad_internal_secret():
    request = SimpleNamespace(headers={"X-Internal-Auth": "wrong-secret"})
    payload = OAuthLoginRequest(
        provider="google",
        provider_user_id="123",
        email="user@example.com",
        name="User",
    )
    with pytest.raises(HTTPException) as exc_info:
        await oauth_login(request=request, data=payload, db=SimpleNamespace())
    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_record_rerank_batches_retrieval_result_lookup():
    run_id = uuid4()
    chunk_1 = uuid4()
    chunk_2 = uuid4()
    r1 = RetrievalResult(retrieval_run_id=run_id, chunk_id=chunk_1, rank=1, score=0.5, metadata_={})
    r2 = RetrievalResult(retrieval_run_id=run_id, chunk_id=chunk_2, rank=2, score=0.4, metadata_={})

    class FakeScalars:
        def all(self):
            return [r1, r2]

    class FakeDB:
        def __init__(self):
            self.scalars_calls = 0
            self.committed = False

        async def scalars(self, stmt):
            self.scalars_calls += 1
            return FakeScalars()

        async def commit(self):
            self.committed = True

    db = FakeDB()
    mgr = RunManager(db)
    mgr.current_retrieval_run_id = run_id

    docs = [
        SimpleNamespace(id=chunk_2, rerank_score=0.95),
        SimpleNamespace(id=chunk_1, rerank_score=0.60),
    ]
    await mgr.record_rerank(docs)

    assert db.scalars_calls == 1
    assert db.committed is True
    assert r2.rerank_score == 0.95
    assert r2.metadata_["rerank_rank"] == 1
    assert r1.rerank_score == 0.60
    assert r1.metadata_["rerank_rank"] == 2
