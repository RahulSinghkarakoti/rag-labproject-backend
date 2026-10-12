from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseLoader(ABC):
    """
    Base interface for all document loaders.
    """

    @abstractmethod
    def load(self, file_path: str | Path) -> list[Document]:
        """
        Load a file and return LangChain Documents.
        """
        raise NotImplementedError