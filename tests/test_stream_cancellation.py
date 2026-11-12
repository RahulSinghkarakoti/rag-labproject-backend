import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.core.cancellation import CancellationManager, cancellation_manager
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.run import Run, RunStep
from app.models.user import User
from app.rag.engine import QueryEngine
from app.rag.run_manager import RunManager
from app.services.document_service import DocumentService


class MockDB:
    def __init__(self, execute_return=None):
        self.execute_return = execute_return
        self.added = []
        self.committed = False
        self.flushed = False

    def add(self, item):
        self.added.append(item)

    async def commit(self):
        self.committed = True

    async def flush(self):
        self.flushed = True
        for item in self.added:
            if getattr(item, "id", None) is None:
                item.id = uuid4()

    async def rollback(self):
        pass

    async def refresh(self, item):
        pass

    async def execute(self, stmt):
        if self.execute_return is not None:
            return self.execute_return
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        return mock_result

    async def scalar(self, stmt):
        return None


@pytest.mark.anyio
async def test_cancellation_manager_lifecycle():
    cm = CancellationManager()
    run_id = str(uuid4())

    # Register run
    event = cm.register(run_id)
    assert not event.is_set()
    assert not cm.is_cancelled(run_id)

    # Cancel run with mocked redis
    mock_redis = AsyncMock()
    with patch("app.core.cancellation.get_redis_pool", return_value=mock_redis):
        signalled = await cm.cancel(run_id)
        assert signalled is True

    assert event.is_set()
    assert cm.is_cancelled(run_id)
    mock_redis.publish.assert_awaited_once_with(f"run_cancellation:{run_id}", "cancel")

    # Unregister run
    cm.unregister(run_id)
    assert run_id not in cm._events


@pytest.mark.anyio
async def test_engine_skips_persisting_completed_result_when_cancelled():
    db = MockDB()
    thread_id = uuid4()
    run = Run(thread_id=thread_id, run_type="chat")
    run.id = uuid4()
    run.status = "in_progress"

    message = Message(
        thread_id=thread_id,
        role="human",
        content="What is RAG?",
        sequence=1,
    )
    message.id = uuid4()

    provider = SimpleNamespace(provider_name="test", config=SimpleNamespace(model="test-model"))

    # Register and cancel the run
    cancellation_manager.register(run.id)
    await cancellation_manager.cancel(run.id)

    try:
        await QueryEngine(db)._persist_run_result(
            run=run,
            message=message,
            result={"response": "RAG response"},
            config={"generation": {}},
            provider=provider,
        )

        # Since it was cancelled, output_message_id must NOT be set and status must not become completed
        assert run.output_message_id is None
        assert run.status != "completed"
        # No assistant message added to db
        assistant_msgs = [item for item in db.added if isinstance(item, Message) and item.role == "assistant"]
        assert len(assistant_msgs) == 0
    finally:
        cancellation_manager.unregister(run.id)


@pytest.mark.anyio
async def test_run_manager_cancel_step():
    db = MockDB()
    rm = RunManager(db)

    step = RunStep(
        run_id=uuid4(),
        sequence=1,
        step_type="query_transformation",
        status="in_progress",
    )

    await rm.cancel_step(step, reason="Aborted by user disconnect")

    assert step.status == "cancelled"
    assert step.completed_at is not None
    assert step.error == {"code": "CANCELLED", "message": "Aborted by user disconnect"}


@pytest.mark.anyio
async def test_run_manager_cancel_step_when_task_cancelled():
    db = MockDB()
    rm = RunManager(db)

    step = RunStep(
        run_id=uuid4(),
        sequence=1,
        step_type="reranking",
        status="running",
    )

    async def _run():
        asyncio.current_task().cancel()
        return await rm.cancel_step(step, reason="Request cancelled by user")

    result = await _run()
    assert result.status == "cancelled"
    assert result.completed_at is not None
    assert result.error == {"code": "CANCELLED", "message": "Request cancelled by user"}


@pytest.mark.anyio
async def test_run_service_cancel_marks_running_steps_cancelled():
    from app.services.run_service import RunService

    db = MockDB()
    service = RunService(db)

    run_id = uuid4()
    step1 = RunStep(run_id=run_id, sequence=1, step_type="retrieval", status="completed")
    step2 = RunStep(run_id=run_id, sequence=2, step_type="reranking", status="running")
    mock_run = Run(
        id=run_id,
        thread_id=uuid4(),
        input_message_id=uuid4(),
        status="running",
    )
    mock_run.steps = [step1, step2]

    service.get_run_with_steps = AsyncMock(return_value=mock_run)
    with patch("app.core.cancellation.cancellation_manager.cancel", new_callable=AsyncMock):
        updated = await service.cancel_run(run_id=run_id, user_id=1)

    assert updated.status == "cancelled"
    assert step1.status == "completed"
    assert step2.status == "cancelled"
    assert step2.completed_at is not None
    assert step2.error == {"code": "CANCELLED", "message": "User interrupted this response."}


@pytest.mark.anyio
async def test_serialize_step_with_expired_attributes():
    from sqlalchemy.exc import PendingRollbackError

    from app.rag.pipeline.executor import serialize_step

    class ExpiredStep:
        def __init__(self):
            self.__dict__ = {"id": str(uuid4()), "status": "cancelled", "sequence": 1}

        def __getattr__(self, item):
            raise PendingRollbackError("Expired attribute access", None)

    step = ExpiredStep()
    serialized = serialize_step(step)
    assert serialized["status"] == "cancelled"
    assert "id" in serialized


@pytest.mark.anyio
async def test_document_service_redis_check_before_persist():
    db = MockDB()
    ds = DocumentService(db)

    user = User(id=1, email="test@example.com")
    kb_id = uuid4()
    project_id = uuid4()

    # Mock verify_kb_ownership so it passes
    ds.verify_kb_ownership = AsyncMock(return_value=KnowledgeBase(id=kb_id, project_id=project_id))

    # Test redis_pool is None raises HTTPException 503 before any file write
    with patch("app.services.document_service.get_redis_pool", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            await ds.process_uploads(
                project_id=project_id,
                kb_id=kb_id,
                files=[],
                current_user=user,
            )
        assert exc_info.value.status_code == 503
        assert "Background worker queue is not available" in exc_info.value.detail
