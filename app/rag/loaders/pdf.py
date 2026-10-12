from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.rag.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    """

    def load(self, file_path: str | Path) -> list[Document]:
        loader = PyPDFLoader(str(file_path))

        documents = loader.load()

        return documents