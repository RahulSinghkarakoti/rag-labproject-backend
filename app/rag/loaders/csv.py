from pathlib import Path

from langchain_community.document_loaders import CSVLoader as LangChainCSVLoader
from langchain_core.documents import Document

from app.rag.loaders.base import BaseLoader


class CSVFileLoader(BaseLoader):
    """
    Loader for CSV files.
    """

    def load(self, file_path: str | Path) -> list[Document]:
        loader = LangChainCSVLoader(
            file_path=str(file_path),
        )

        return loader.load()