from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.rag.pipeline.query import QueryPipeline
from app.rag.generation.base import BaseGenerationProvider


class RecordingProvider(BaseGenerationProvider):
    provider_name = "test"

    def __init__(self):
        self.prompts = []

    def _build_model(self, config):
        return None

    async def ainvoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "generated answer"


class StreamingProvider(RecordingProvider):
    async def astream(self, prompt: str):
        self.prompts.append(prompt)
        yield "generated "
        yield "answer"


@pytest.mark.anyio
async def test_query_pipeline_generates_from_queries_documents_and_memory():
    provider = RecordingProvider()
    thread = MagicMock(id=uuid4(), knowledge_bases=[])
    message = MagicMock(content="What is RAG?", sequence=1)
    db = AsyncMock()
    db.scalars.return_value = MagicMock()
    db.scalars.return_value.all.return_value = []
    pipeline = QueryPipeline(
        db=db,
        config={"generation": {"model": "gpt-4o", "temperature": 0.1, "max_tokens": 200}},
        run=MagicMock(),
        thread=thread,
        message=message,
        provider=provider,
    )
    async def memory():
        return [{"role": "human", "content": "We discussed embeddings."}]

    pipeline._load_memory = memory

    result = await pipeline.execute()

    assert result["response"] == "generated answer"
    assert "What is RAG?" in provider.prompts[0]
    assert "We discussed embeddings." in provider.prompts[0]


@pytest.mark.anyio
async def test_query_pipeline_streams_and_assembles_response():
    provider = StreamingProvider()
    db = AsyncMock()
    db.scalars.return_value = MagicMock()
    db.scalars.return_value.all.return_value = []
    pipeline = QueryPipeline(
        db=db,
        config={"generation": {"model": "gpt-4o"}},
        run=MagicMock(),
        thread=MagicMock(id=uuid4(), knowledge_bases=[]),
        message=MagicMock(content="What is RAG?", sequence=1),
        provider=provider,
    )

    events = [event async for event in pipeline.execute_stream()]

    assert [event["content"] for event in events[:-1]] == ["generated ", "answer"]
    assert events[-1]["type"] == "complete"
    assert events[-1]["result"]["response"] == "generated answer"
