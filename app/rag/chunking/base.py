from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseChunker(ABC):
    """
    Base interface for document chunkers.
    """

    strategy_name: str = "base"

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks.
        """
        raise NotImplementedError

    def _add_chunk_metadata(self, chunks: list[Document]) -> None:
        """
        Add chunk index, total chunks, and chunking strategy to metadata.
        """
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["total_chunks"] = total_chunks
            if "chunking_strategy" not in chunk.metadata:
                chunk.metadata["chunking_strategy"] = self.strategy_name