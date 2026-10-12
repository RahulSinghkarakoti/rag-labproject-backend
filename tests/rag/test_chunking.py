from pathlib import Path

from app.rag.chunking import RecursiveChunker
from app.rag.loaders.factory import LoaderFactory


def test_text_chunking():
    file_path = Path("tests/data/sample.txt")
    # Load
    loader = LoaderFactory()
    documents = loader.load(file_path)
    assert documents
    # Chunk
    chunker = RecursiveChunker(
        chunk_size=100,
        chunk_overlap=20,
    )
    chunks = chunker.split(documents)
    assert chunks
    print(f"\nOriginal documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks):
        print(f"\n{'=' * 60}")
        print(f"CHUNK {index + 1}")
        print(f"{'=' * 60}")
        print("Content:")
        print(chunk.page_content)
        print("\nMetadata:")
        print(chunk.metadata)

import pytest

from langchain_core.documents import Document


def test_empty_documents():
    chunker = RecursiveChunker()
    chunks = chunker.split([])
    assert chunks == []


def test_chunk_metadata():
    documents = [
        Document(
            page_content="A " * 100,
            metadata={
                "source": "test.txt",
            },
        )
    ]
    chunker = RecursiveChunker(
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = chunker.split(documents)
    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["total_chunks"] == len(chunks)
        assert chunk.metadata["source"] == "test.txt"


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveChunker(
            chunk_size=100,
            chunk_overlap=100,
        )