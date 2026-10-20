import logging
import re
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.rag.chunking.base import BaseChunker

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Chunk documents using LangChain's RecursiveCharacterTextSplitter.
    """

    strategy_name = "recursive"

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
            raise ValueError("chunk_overlap must be smaller than chunk_size")

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

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)
        self._add_chunk_metadata(chunks)
        return chunks


class CharacterChunker(BaseChunker):
    """
    Chunk documents using LangChain's CharacterTextSplitter with a fixed separator.
    """

    strategy_name = "character"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separator: str = "\n\n",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)
        self._add_chunk_metadata(chunks)
        return chunks


class MarkdownChunker(BaseChunker):
    """
    Chunk Markdown documents by headers (#, ##, ###), then sub-chunk large sections.
    Falls back to RecursiveChunker if no Markdown header structure is present.
    """

    strategy_name = "markdown"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        headers = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        self.recursive_sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        all_chunks: list[Document] = []

        for doc in documents:
            if not re.search(r"(?m)^#{1,3}\s+", doc.page_content):
                logger.warning(
                    "No Markdown headers found in document content. Falling back to RecursiveCharacterTextSplitter."
                )
                fallback_chunker = RecursiveChunker(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                fallback_chunks = fallback_chunker.split([doc])
                for chunk in fallback_chunks:
                    chunk.metadata["chunking_strategy"] = "markdown (fallback: recursive)"
                all_chunks.extend(fallback_chunks)
                continue

            header_splits = self.markdown_splitter.split_text(doc.page_content)
            for split_doc in header_splits:
                split_metadata = {**doc.metadata, **split_doc.metadata}
                if len(split_doc.page_content) > self.chunk_size:
                    sub_docs = self.recursive_sub_splitter.create_documents(
                        texts=[split_doc.page_content],
                        metadatas=[split_metadata],
                    )
                    all_chunks.extend(sub_docs)
                else:
                    split_doc.metadata = split_metadata
                    all_chunks.append(split_doc)

        self._add_chunk_metadata(all_chunks)
        return all_chunks


class SemanticChunker(BaseChunker):
    """
    Chunk documents semantically based on sentence embeddings and distance thresholds.
    Falls back to RecursiveChunker if sentence tokenization or embedder is unavailable.
    """

    strategy_name = "semantic"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        embedder: Any | None = None,
        breakpoint_threshold_amount: float = 0.4,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        if self.embedder is None:
            logger.warning(
                "Embedder is not configured for SemanticChunker. Falling back to RecursiveChunker."
            )
            fallback_chunker = RecursiveChunker(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunks = fallback_chunker.split(documents)
            for chunk in chunks:
                chunk.metadata["chunking_strategy"] = "semantic (fallback: recursive)"
            return chunks

        all_chunks: list[Document] = []
        for doc in documents:
            sentences = self._split_into_sentences(doc.page_content)
            if not sentences:
                continue

            try:
                doc_chunks = self._chunk_sentences_by_embedding(sentences, doc.metadata)
                all_chunks.extend(doc_chunks)
            except Exception as exc:
                logger.warning(
                    f"Semantic chunking failed with error: {exc}. Falling back to RecursiveChunker."
                )
                fallback_chunker = RecursiveChunker(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                fallback_chunks = fallback_chunker.split([doc])
                for chunk in fallback_chunks:
                    chunk.metadata["chunking_strategy"] = "semantic (fallback: recursive)"
                all_chunks.extend(fallback_chunks)

        self._add_chunk_metadata(all_chunks)
        return all_chunks

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def _chunk_sentences_by_embedding(
        self, sentences: list[str], metadata: dict
    ) -> list[Document]:
        if len(sentences) == 1:
            return [Document(page_content=sentences[0], metadata=dict(metadata))]

        embeddings = self.embedder.embed_documents(sentences)

        import numpy as np

        chunks: list[Document] = []
        current_sentences: list[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(len(sentences) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                distance = 1.0 - cosine_sim
            else:
                distance = 1.0

            next_sentence = sentences[i + 1]
            if distance > self.breakpoint_threshold_amount or (
                current_len + len(next_sentence) > self.chunk_size
            ):
                chunks.append(
                    Document(
                        page_content=" ".join(current_sentences),
                        metadata=dict(metadata),
                    )
                )
                current_sentences = [next_sentence]
                current_len = len(next_sentence)
            else:
                current_sentences.append(next_sentence)
                current_len += len(next_sentence) + 1

        if current_sentences:
            chunks.append(
                Document(
                    page_content=" ".join(current_sentences),
                    metadata=dict(metadata),
                )
            )

        return chunks