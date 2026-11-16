import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message
from app.models.run import Run, RunStep
from app.models.thread import Thread
from app.rag.generation.base import BaseGenerationProvider
from app.rag.memory.summarizer import ThreadSummarizer
from app.rag.pipeline.query import QueryPipeline
from app.rag.query.rewriter import QueryRewriter
from app.schemas.run import ModelCallResponse, RunDetailsResponse
from app.schemas.thread import GenerationConfig, QueryTransformationConfig, ThreadConfig


@pytest.fixture
def mock_db():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_provider():
    provider = AsyncMock(spec=BaseGenerationProvider)
    provider.ainvoke.return_value = "Rewritten search query"
    provider.ainvoke_safe.return_value = "Mocked answer"
    provider.provider_name = "groq"
    provider.config = MagicMock(model="openai/gpt-oss-120b")

    async def mock_stream(*args, **kwargs):
        yield "Streaming "
        yield "RAG response."

    provider.astream = mock_stream
    return provider


@pytest.fixture
def mock_run_manager():
    rm = MagicMock()
    rm.run = Run(id=uuid4())
    rm.start_step = AsyncMock(return_value=RunStep(id=uuid4(), step_type="generation", sequence=1))
    rm.complete_step = AsyncMock(return_value=RunStep(id=uuid4(), step_type="generation", sequence=1))
    rm.fail_step = AsyncMock(return_value=RunStep(id=uuid4(), step_type="generation", sequence=1))
    rm.cancel_step = AsyncMock(return_value=RunStep(id=uuid4(), step_type="generation", sequence=1))
    rm.record_model_call = AsyncMock()
    return rm


@pytest.mark.anyio
async def test_generation_records_model_call_with_tokens_and_latency(
    mock_db, mock_provider, mock_run_manager
):
    config = ThreadConfig(
        generation=GenerationConfig(provider="groq", llm_model="openai/gpt-oss-120b", temperature=0.3, max_tokens=256)
    )
    run = Run(id=uuid4())
    thread = Thread(id=uuid4())
    message = Message(id=uuid4(), content="Explain quantum computing.", sequence=1)

    pipeline = QueryPipeline(
        db=mock_db,
        config=config,
        run=run,
        run_manager=mock_run_manager,
        thread=thread,
        message=message,
        provider=mock_provider,
    )

    with patch("app.rag.pipeline.query.MemoryManager.load", return_value=[]), \
         patch.object(pipeline, "_retrieve", return_value=[]), \
         patch.object(pipeline, "_rerank", return_value=[]):
        events = [event async for event in pipeline.execute_stream()]

    assert any(e["type"] == "complete" for e in events)

    # Verify record_model_call was called for generation
    mock_run_manager.record_model_call.assert_called_once()
    call_kwargs = mock_run_manager.record_model_call.call_args.kwargs

    assert call_kwargs["provider"] == "groq"
    assert call_kwargs["model"] == "openai/gpt-oss-120b"
    assert call_kwargs["status"] == "completed"
    assert call_kwargs["request"]["stage"] == "generation"
    assert "Explain quantum computing." in call_kwargs["request"]["prompt"]
    assert call_kwargs["request"]["temperature"] == 0.3
    assert call_kwargs["request"]["max_tokens"] == 256
    assert call_kwargs["response"]["content"] == "Streaming RAG response."
    assert call_kwargs["input_tokens"] > 0
    assert call_kwargs["output_tokens"] > 0
    assert call_kwargs["total_tokens"] == call_kwargs["input_tokens"] + call_kwargs["output_tokens"]
    assert isinstance(call_kwargs["latency_ms"], int)
    assert call_kwargs["latency_ms"] >= 0


@pytest.mark.anyio
async def test_generation_failure_records_failed_model_call(
    mock_db, mock_provider, mock_run_manager
):
    config = ThreadConfig()
    run = Run(id=uuid4())
    thread = Thread(id=uuid4())
    message = Message(id=uuid4(), content="Will this fail?", sequence=1)

    async def fail_stream(*args, **kwargs):
        yield "Initial chunk"
        raise RuntimeError("Provider API rate limit exceeded")

    mock_provider.astream = fail_stream

    pipeline = QueryPipeline(
        db=mock_db,
        config=config,
        run=run,
        run_manager=mock_run_manager,
        thread=thread,
        message=message,
        provider=mock_provider,
    )

    with patch("app.rag.pipeline.query.MemoryManager.load", return_value=[]), \
         patch.object(pipeline, "_retrieve", return_value=[]), \
         patch.object(pipeline, "_rerank", return_value=[]):
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            _ = [event async for event in pipeline.execute_stream()]

    mock_run_manager.record_model_call.assert_called_once()
    call_kwargs = mock_run_manager.record_model_call.call_args.kwargs
    assert call_kwargs["status"] == "failed"
    assert call_kwargs["request"]["stage"] == "generation"
    assert "Provider API rate limit exceeded" in call_kwargs["error"]["message"]
    assert isinstance(call_kwargs["latency_ms"], int)


@pytest.mark.anyio
async def test_query_transformation_records_model_call(
    mock_provider, mock_run_manager
):
    rewriter = QueryRewriter()
    config = QueryTransformationConfig(mode="rewrite", include_original=False)

    queries = await rewriter.transform(
        query="What is it?",
        provider=mock_provider,
        config=config,
        memory=[{"role": "user", "content": "Tell me about RAG."}],
        run_manager=mock_run_manager,
    )

    assert queries == ["Rewritten search query"]
    mock_run_manager.record_model_call.assert_called_once()
    call_kwargs = mock_run_manager.record_model_call.call_args.kwargs

    assert call_kwargs["provider"] == "groq"
    assert call_kwargs["status"] == "completed"
    assert call_kwargs["request"]["stage"] == "query_transformation"
    assert call_kwargs["request"]["transformer"] == "rewrite"
    assert call_kwargs["response"]["content"] == "Rewritten search query"
    assert call_kwargs["input_tokens"] > 0
    assert call_kwargs["output_tokens"] > 0
    assert isinstance(call_kwargs["latency_ms"], int)


@pytest.mark.anyio
async def test_memory_summarizer_records_model_call(
    mock_provider, mock_run_manager
):
    mock_provider.ainvoke.return_value = "Compacted conversation summary."
    summarizer = ThreadSummarizer(mock_provider)

    turns = [
        {"role": "user", "content": "Hello, I want to learn about FastAPI."},
        {"role": "assistant", "content": "FastAPI is a modern web framework."},
    ]

    summary = await summarizer.summarize_turns(
        existing_summary=None,
        turns=turns,
        run_manager=mock_run_manager,
    )

    assert summary == "Compacted conversation summary."
    mock_run_manager.record_model_call.assert_called_once()
    call_kwargs = mock_run_manager.record_model_call.call_args.kwargs

    assert call_kwargs["provider"] == "groq"
    assert call_kwargs["status"] == "completed"
    assert call_kwargs["request"]["stage"] == "memory_summarizer"
    assert call_kwargs["request"]["turn_count"] == 2
    assert call_kwargs["response"]["content"] == "Compacted conversation summary."
    assert call_kwargs["input_tokens"] > 0
    assert call_kwargs["output_tokens"] > 0
    assert isinstance(call_kwargs["latency_ms"], int)


def test_run_details_response_schema_contains_model_calls():
    run_id = uuid4()
    call_id = uuid4()
    model_call = ModelCallResponse(
        id=call_id,
        run_id=run_id,
        provider="groq",
        model="openai/gpt-oss-120b",
        request={"stage": "generation", "prompt": "test prompt"},
        response={"content": "test response"},
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_ms=120,
        status="completed",
    )

    response = RunDetailsResponse(
        id=run_id,
        thread_id=uuid4(),
        run_type="chat",
        status="completed",
        steps=[],
        model_calls=[model_call],
    )

    data = response.model_dump()
    assert len(data["model_calls"]) == 1
    assert data["model_calls"][0]["id"] == call_id
    assert data["model_calls"][0]["latency_ms"] == 120
    assert data["model_calls"][0]["total_tokens"] == 15
    assert data["model_calls"][0]["request"]["stage"] == "generation"


@pytest.mark.anyio
async def test_pipeline_records_multiple_model_calls_for_transformation_and_generation(
    mock_db, mock_provider, mock_run_manager
):
    config = ThreadConfig(
        query_transformation=QueryTransformationConfig(mode="rewrite"),
        generation=GenerationConfig(provider="groq", llm_model="openai/gpt-oss-120b"),
    )
    run = Run(id=uuid4())
    thread = Thread(id=uuid4())
    message = Message(id=uuid4(), content="How does RAG work?", sequence=1)

    pipeline = QueryPipeline(
        db=mock_db,
        config=config,
        run=run,
        run_manager=mock_run_manager,
        thread=thread,
        message=message,
        provider=mock_provider,
    )

    with patch("app.rag.pipeline.query.MemoryManager.load", return_value=[]), \
         patch.object(pipeline, "_retrieve", return_value=[]), \
         patch.object(pipeline, "_rerank", return_value=[]):
        events = [event async for event in pipeline.execute_stream()]

    assert any(e["type"] == "complete" for e in events)

    # Exactly 2 model calls should be recorded: transformation + generation
    assert mock_run_manager.record_model_call.call_count == 2
    calls = mock_run_manager.record_model_call.call_args_list

    stages = [call.kwargs["request"]["stage"] for call in calls]
    assert "query_transformation" in stages
    assert "generation" in stages
