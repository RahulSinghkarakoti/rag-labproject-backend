from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.chunking.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Chunk documents using LangChain's
    RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(
        self,
        documents: list[Document],
    ) -> list[Document]:
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        self._add_chunk_metadata(chunks)

        return chunks

    @staticmethod
    def _add_chunk_metadata(
        chunks: list[Document],
    ) -> None:
        total_chunks = len(chunks)

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["total_chunks"] = total_chunks