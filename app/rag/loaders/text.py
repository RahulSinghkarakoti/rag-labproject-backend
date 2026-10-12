from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.rag.loaders.base import BaseLoader


class TextFileLoader(BaseLoader):
    """
    Loader for plain text files.
    """

    def load(self, file_path: str | Path) -> list[Document]:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        return loader.load()