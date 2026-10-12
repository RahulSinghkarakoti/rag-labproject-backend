from pathlib import Path

from langchain_core.documents import Document

from app.rag.loaders.detector import FileTypeDetector
from app.rag.loaders.registry import LoaderRegistry


class LoaderFactory:
    """
    Automatically detects a file type and creates
    the appropriate loader.
    """

    def __init__(
        self,
        registry: LoaderRegistry | None = None,
        detector: FileTypeDetector | None = None,
    ):
        self.registry = registry or LoaderRegistry()
        self.detector = detector or FileTypeDetector()

    def get_loader(
        self,
        file_path: str | Path,
    ):
        file_type = self.detector.detect(file_path)

        return self.registry.get(file_type)

    def load(
        self,
        file_path: str | Path,
    ) -> list[Document]:
        loader = self.get_loader(file_path)

        return loader.load(file_path)