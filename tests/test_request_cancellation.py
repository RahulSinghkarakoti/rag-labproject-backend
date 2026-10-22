import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.message import Message
from app.models.run import Run
from app.rag.engine import QueryEngine
from app.rag.generation.base import BaseGenerationProvider
from app.schemas.thread import ThreadConfig


class CancellingProvider(BaseGenerationProvider):
    provider_name = "test"

    def __init__(self, config=None):
        super().__init__(config or {})

    def _build_model(self, config):
        return None

    async def ainvoke(self, prompt: str) -> str:
        raise asyncio.CancelledError()

    async def astream(self, prompt: str):
        yield "Partial token 1 "
        yield "Partial token 2 "
        raise asyncio.CancelledError()


@pytest.mark.anyio
async def test_stream_cancellation_persists_status_and_message():
    provider = CancellingProvider()
    thread = MagicMock(id=uuid4(), knowledge_bases=[])
    message = Message(id=uuid4(), thread_id=thread.id, content="What is RAG?", sequence=1)
    run = Run(id=uuid4(), thread_id=thread.id, input_message_id=message.id, status="running")

    db = AsyncMock()
    db.scalars.return_value = MagicMock()
    db.scalars.return_value.all.return_value = []

    engine = QueryEngine(db)
    engine.provider_factory = MagicMock()
    engine.provider_factory.create.return_value = provider

    pipeline_version = MagicMock(config={"generation": {"provider": "groq", "model": "gpt-oss-120b"}})

    tokens = []
    try:
        async for event in engine.run_stream(
            run=run,
            thread=thread,
            message=message,
            pipeline_version=pipeline_version,
        ):
            if event["type"] == "token":
                tokens.append(event["content"])
    except (asyncio.CancelledError, RuntimeError):
        pass

    assert run.status == "cancelled"
    assert run.error["code"] == "CANCELLED"
    assert "User stopped" in run.error["message"]
    assert db.commit.called
