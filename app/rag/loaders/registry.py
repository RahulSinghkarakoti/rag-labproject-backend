from app.rag.loaders.base import BaseLoader
from app.rag.loaders.csv import CSVFileLoader
from app.rag.loaders.pdf import PDFLoader
from app.rag.loaders.text import TextFileLoader


class LoaderRegistry:
    """
    Registry of available document loaders.
    """

    def __init__(self):
        self._loaders: dict[str, type[BaseLoader]] = {
            "pdf": PDFLoader,
            "text": TextFileLoader,
            "csv": CSVFileLoader,
        }

    def register(
        self,
        name: str,
        loader_class: type[BaseLoader],
    ) -> None:
        self._loaders[name] = loader_class

    def get(self, name: str) -> BaseLoader:
        loader_class = self._loaders.get(name)

        if not loader_class:
            raise ValueError(
                f"No loader registered for type: {name}"
            )

        return loader_class()