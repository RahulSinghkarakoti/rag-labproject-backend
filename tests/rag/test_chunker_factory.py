from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.chunking import (
    CharacterChunker,
    ChunkerFactory,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
)


def test_factory_creates_recursive_chunker():
    chunker = ChunkerFactory.create("recursive", chunk_size=100, chunk_overlap=10)
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.strategy_name == "recursive"


def test_factory_creates_character_chunker():
    chunker = ChunkerFactory.create("character", chunk_size=15, chunk_overlap=2)
    assert isinstance(chunker, CharacterChunker)
    assert chunker.strategy_name == "character"

    docs = [Document(page_content="First block.\n\nSecond block.")]
    chunks = chunker.split(docs)
    assert len(chunks) == 2
    assert chunks[0].metadata["chunking_strategy"] == "character"


def test_factory_creates_markdown_chunker_for_markdown_file():
    chunker = ChunkerFactory.create(
        "markdown", chunk_size=500, chunk_overlap=50, file_type="markdown"
    )
    assert isinstance(chunker, MarkdownChunker)
    assert chunker.strategy_name == "markdown"

    markdown_text = (
        "# Title Header\n\nIntroductory text here.\n\n"
        "## Subheader Section\n\nSection content details go here."
    )
    docs = [Document(page_content=markdown_text, metadata={"source": "test.md"})]
    chunks = chunker.split(docs)

    assert len(chunks) >= 2
    assert chunks[0].metadata["Header 1"] == "Title Header"
    assert chunks[1].metadata["Header 2"] == "Subheader Section"
    assert chunks[0].metadata["chunking_strategy"] == "markdown"


def test_factory_fallback_markdown_on_pdf():
    chunker = ChunkerFactory.create(
        "markdown", chunk_size=500, chunk_overlap=50, file_type="pdf"
    )
    # Strategy should fall back to recursive for PDF file type
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.strategy_name == "recursive"


def test_markdown_chunker_fallback_on_plain_text_without_headers():
    chunker = MarkdownChunker(chunk_size=100, chunk_overlap=10)
    docs = [Document(page_content="Plain text line without any markdown headers.")]
    chunks = chunker.split(docs)
    assert len(chunks) == 1
    assert "fallback: recursive" in chunks[0].metadata["chunking_strategy"]


def test_factory_creates_semantic_chunker():
    mock_embedder = MagicMock()
    # Return 2D vectors for 3 sentences
    mock_embedder.embed_documents.return_value = [
        [1.0, 0.0],
        [0.9, 0.1],  # Close to first sentence
        [0.0, 1.0],  # Far from second sentence (trigger breakpoint)
    ]

    chunker = ChunkerFactory.create(
        "semantic",
        chunk_size=500,
        chunk_overlap=50,
        file_type="text",
        embedder=mock_embedder,
        breakpoint_threshold_amount=0.3,
    )
    assert isinstance(chunker, SemanticChunker)
    assert chunker.strategy_name == "semantic"

    docs = [
        Document(
            page_content="Apple is a fruit. Apples taste sweet. Python is a programming language."
        )
    ]
    chunks = chunker.split(docs)
    assert len(chunks) == 2
    assert chunks[0].metadata["chunking_strategy"] == "semantic"


def test_factory_fallback_semantic_on_csv():
    chunker = ChunkerFactory.create(
        "semantic", chunk_size=500, chunk_overlap=50, file_type="csv"
    )
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.strategy_name == "recursive"


def test_unknown_strategy_defaults_to_recursive():
    chunker = ChunkerFactory.create("non_existent_strategy")
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.strategy_name == "recursive"
