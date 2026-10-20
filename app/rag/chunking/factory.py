import logging
from typing import Any

from app.rag.chunking.base import BaseChunker
from app.rag.chunking.splitter import (
    CharacterChunker,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
)

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """
    Factory for creating document chunkers based on selected strategy
    and document type compatibility rules.
    """

    STRATEGY_COMPATIBILITY = {
        "recursive": ["pdf", "text", "markdown", "csv"],
        "character": ["pdf", "text", "markdown", "csv"],
        "semantic": ["pdf", "text", "markdown"],
        "markdown": ["markdown", "text"],
    }

    @classmethod
    def create(
        cls,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        file_type: str = "text",
        embedder: Any | None = None,
        **kwargs: Any,
    ) -> BaseChunker:
        normalized_strategy = (strategy or "recursive").lower().strip()
        normalized_file_type = (file_type or "text").lower().strip()

        # Document type compatibility validation and fallback logic
        if normalized_strategy == "markdown" and normalized_file_type not in ["markdown", "text"]:
            logger.warning(
                f"Strategy 'markdown' is not optimal for file type '{file_type}'. "
                f"Falling back to 'recursive' strategy."
            )
            normalized_strategy = "recursive"

        elif normalized_strategy == "semantic" and normalized_file_type == "csv":
            logger.warning(
                f"Strategy 'semantic' is not optimal for tabular file type 'csv'. "
                f"Falling back to 'recursive' strategy."
            )
            normalized_strategy = "recursive"

        if normalized_strategy == "character":
            return CharacterChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=kwargs.get("separator", "\n\n"),
            )

        elif normalized_strategy == "markdown":
            return MarkdownChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                headers_to_split_on=kwargs.get("headers_to_split_on"),
            )

        elif normalized_strategy == "semantic":
            return SemanticChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedder=embedder,
                breakpoint_threshold_amount=kwargs.get("breakpoint_threshold_amount", 0.4),
            )

        else:
            if normalized_strategy not in cls.STRATEGY_COMPATIBILITY:
                logger.warning(
                    f"Unknown chunking strategy '{strategy}'. Defaulting to 'recursive'."
                )
            return RecursiveChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
