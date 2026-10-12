from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseChunker(ABC):
    """
    Base interface for document chunkers.
    """

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks.
        """
        raise NotImplementedError