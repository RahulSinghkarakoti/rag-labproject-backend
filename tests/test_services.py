import io
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile

from app.models.message import Message
from app.models.pipeline import PipelineVersion
from app.models.thread import Thread
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService


class MockResult:
    def __init__(self, value=None, values=None):
        self._value = value
        self._values = values or []

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return self._values


class MockDB:
    def __init__(self, scalar_return=None):
        self.scalar_return = scalar_return
        self.added = []
        self.deleted = []
        self.committed = False
        self.flushed = False

    async def scalar(self, stmt):
        if callable(self.scalar_return):
            return self.scalar_return(stmt)
        return self.scalar_return

    async def scalars(self, stmt):
        return MockResult(values=self.scalar_return if isinstance(self.scalar_return, list) else [])

    async def execute(self, stmt):
        return MockResult(value=self.scalar_return)

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


@pytest.mark.anyio
async def test_chat_service_get_thread_not_found():
    db = MockDB(scalar_return=None)
    service = ChatService(db)

    with pytest.raises(HTTPException) as exc_info:
        await service.get_thread(uuid4(), user_id=1)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread not found"


@pytest.mark.anyio
async def test_chat_service_get_pipeline_version_no_pipeline():
    db = MockDB(scalar_return=None)
    service = ChatService(db)
    thread = Thread(id=uuid4(), user_id=1, pipeline_id=None)

    with pytest.raises(HTTPException) as exc_info:
        await service.get_latest_pipeline_version(thread)

    assert exc_info.value.status_code == 400
    assert "no active pipeline" in exc_info.value.detail


@pytest.mark.anyio
async def test_chat_service_sync_pipeline_config_bumps_version():
    pipeline_id = uuid4()
    thread = Thread(
        id=uuid4(),
        user_id=1,
        pipeline_id=pipeline_id,
        metadata_={"temperature": 0.5},
    )
    latest_version = PipelineVersion(
        id=uuid4(),
        version=1,
        config={"temperature": 0.5},
        pipeline_id=pipeline_id,
    )

    db = MockDB(scalar_return=latest_version)
    service = ChatService(db)

    # Change config
    new_config = {"temperature": 0.9}
    await service.sync_thread_pipeline_config(thread, new_config)

    assert len(db.added) == 1
    new_version_obj = db.added[0]
    assert isinstance(new_version_obj, PipelineVersion)
    assert new_version_obj.version == 2
    assert new_version_obj.config == {"temperature": 0.9}
    assert thread.metadata_ == {"temperature": 0.9}


@pytest.mark.anyio
async def test_chat_service_prepare_message_requires_content():
    thread = Thread(id=uuid4(), user_id=1)
    db = MockDB(scalar_return=0)
    service = ChatService(db)

    with pytest.raises(HTTPException) as exc_info:
        await service.prepare_chat_message(thread, content="")

    assert exc_info.value.status_code == 400
    assert "Message content is required" in exc_info.value.detail


@pytest.mark.anyio
async def test_chat_service_edit_message_rejects_assistant_role():
    thread = Thread(id=uuid4(), user_id=1)
    msg_id = uuid4()
    assistant_msg = Message(id=msg_id, thread_id=thread.id, role="assistant", content="AI response")

    db = MockDB(scalar_return=assistant_msg)
    service = ChatService(db)

    with pytest.raises(HTTPException) as exc_info:
        await service.edit_message(thread, msg_id, "New text")

    assert exc_info.value.status_code == 400
    assert "Only user messages can be edited" in exc_info.value.detail


@pytest.mark.anyio
async def test_document_service_calculate_checksum():
    content = b"Hello world testing document checksum"
    file_obj = UploadFile(filename="test.txt", file=io.BytesIO(content))

    checksum = await DocumentService.calculate_checksum(file_obj)
    import hashlib
    expected = hashlib.sha256(content).hexdigest()
    assert checksum == expected


@pytest.mark.anyio
async def test_document_service_verify_kb_ownership_not_found():
    db = MockDB(scalar_return=None)
    service = DocumentService(db)

    with pytest.raises(HTTPException) as exc_info:
        await service.verify_kb_ownership(uuid4(), uuid4(), user_id=1)

    assert exc_info.value.status_code == 404
    assert "Knowledge base or project not found" in exc_info.value.detail
