from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.rag.pipeline.query import QueryPipeline, is_greeting_query
from app.rag.generation.base import BaseGenerationProvider
from app.schemas.thread import ThreadConfig


class UnusedProvider(BaseGenerationProvider):
    provider_name = "test"

    def __init__(self, config=None):
        super().__init__(config or {})

    def _build_model(self, config):
        return None

    async def ainvoke(self, prompt: str) -> str:
        raise RuntimeError("Provider should NOT be called for greetings!")

    async def astream(self, prompt: str):
        raise RuntimeError("Provider should NOT be called for greetings!")
        yield ""


def test_is_greeting_query():
    assert is_greeting_query("hi") is True
    assert is_greeting_query("Hello!") is True
    assert is_greeting_query("hey there") is True
    assert is_greeting_query("good morning") is True
    assert is_greeting_query("how are you") is True

    # Non-greetings / actual RAG queries
    assert is_greeting_query("What is the refund policy?") is False
    assert is_greeting_query("Summarize this document") is False
    assert is_greeting_query("How to setup docker environment?") is False


@pytest.mark.anyio
async def test_greeting_bypasses_pipeline_non_streaming():
    provider = UnusedProvider()
    thread = MagicMock(id=uuid4(), knowledge_bases=[])
    message = MagicMock(content="hi", sequence=1)
    db = AsyncMock()
    db.scalars.return_value = MagicMock()
    db.scalars.return_value.all.return_value = []

    pipeline = QueryPipeline(
        db=db,
        config=ThreadConfig(),
        run=MagicMock(),
        thread=thread,
        message=message,
        provider=provider,
    )

    result = await pipeline.execute()

    assert "Hi! How are you?" in result["response"]
    assert result["retrieved_documents"] == []


@pytest.mark.anyio
async def test_greeting_bypasses_pipeline_streaming():
    provider = UnusedProvider()
    thread = MagicMock(id=uuid4(), knowledge_bases=[])
    message = MagicMock(content="hello", sequence=1)
    db = AsyncMock()
    db.scalars.return_value = MagicMock()
    db.scalars.return_value.all.return_value = []

    pipeline = QueryPipeline(
        db=db,
        config=ThreadConfig(),
        run=MagicMock(),
        thread=thread,
        message=message,
        provider=provider,
    )

    events = [event async for event in pipeline.execute_stream()]

    assert len(events) == 2
    assert events[0]["type"] == "token"
    assert "Hi! How are you?" in events[0]["content"]
    assert events[1]["type"] == "complete"
    assert "Hi! How are you?" in events[1]["result"]["response"]
