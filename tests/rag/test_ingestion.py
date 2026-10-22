import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.rag.pipeline.ingestion import IngestionPipeline


@pytest.mark.anyio
async def test_ingestion_pipeline():
    """
    Smoke-test for IngestionPipeline.process().

    Uses a mock vector store so the test runs without a live DB or embedding
    model — those are exercised by the integration test suite.
    """
    mock_vector_store = AsyncMock()
    mock_vector_store.add_documents = AsyncMock(return_value=None)

    # Provide a real embedder stub (avoids loading the model)
    mock_embedder = MagicMock()
    mock_embedder.embed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    pipeline = IngestionPipeline(
        vector_store=mock_vector_store,
        enbedder=mock_embedder,
    )

    chunks = await pipeline.process(Path("tests/data/sample.txt"))

    assert chunks, "Expected at least one chunk from sample.txt"

    print(f"\nGenerated chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {index + 1} ---")
        print(chunk.page_content)
        print(chunk.metadata)