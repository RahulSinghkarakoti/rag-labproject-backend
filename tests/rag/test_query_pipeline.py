import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.message import Message
from app.models.run import Run
from app.models.thread import Thread
from app.rag.generation.base import BaseGenerationProvider
from app.rag.pipeline.query import QueryPipeline
from app.schemas.thread import ThreadConfig


@pytest.fixture
def mock_db():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_provider():
    provider = AsyncMock(spec=BaseGenerationProvider)
    provider.ainvoke_safe.return_value = "Mocked answer"
    provider.provider_name = "mock"
    provider.config = MagicMock(model="mock-model")
    
    async def mock_stream(*args, **kwargs):
        yield "Mocked "
        yield "answer"
        
    provider.astream = mock_stream
    return provider


@pytest.mark.asyncio
async def test_query_pipeline_execute(mock_db, mock_provider):
    config = ThreadConfig()
    run = Run(id="run-123")
    thread = Thread(id="thread-123")
    message = Message(id="msg-123", content="How does this work?", sequence=1)
    
    pipeline = QueryPipeline(
        db=mock_db,
        config=config,
        run=run,
        thread=thread,
        message=message,
        provider=mock_provider,
    )
    
    with patch("app.rag.pipeline.query.MemoryManager.load", return_value=[]), \
         patch("app.rag.pipeline.query.QueryPipeline._process_query", return_value=["How does this work?"]), \
         patch("app.rag.pipeline.query.QueryPipeline._retrieve", return_value=[]), \
         patch("app.rag.pipeline.query.QueryPipeline._rerank", return_value=[]):
             
        result = await pipeline.execute()
        
        assert result["original_query"] == "How does this work?"
        assert result["answer"] == "Mocked answer"
        assert result["response"] == "Mocked answer"
        mock_provider.ainvoke_safe.assert_called_once()
