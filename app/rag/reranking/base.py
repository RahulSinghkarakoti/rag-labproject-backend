from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Interface for query-document rerankers."""

    @abstractmethod
    def rerank(self, queries: list[str], documents: list[Any]) -> list[Any]:
        raise NotImplementedError
