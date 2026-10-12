from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):

    @abstractmethod
    async def add_documents(
        self,
        documents: list[Any],
        embeddings: list[list[float]],
        *,
        metadata: list[dict] | None = None,
    ) -> None:
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[Any]:
        pass

    @abstractmethod
    async def delete_document(
        self,
        document_id: str,
    ) -> None:
        pass