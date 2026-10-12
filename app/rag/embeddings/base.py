from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Base interface for embedding providers.
    """

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a query.
        """
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """
        raise NotImplementedError