import asyncio
from pathlib import Path

from langchain_core.documents import Document

from app.rag.chunking import RecursiveChunker
from app.rag.loaders import LoaderFactory
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.vectorstore import PgVectorStore
from app.core.sse import publisher

class IngestionPipeline:
    """
    Orchestrates the document ingestion workflow. 
        File → Load → Chunk → Embed → Store
    """

    def __init__(
        self,
        loader_factory: LoaderFactory | None = None,
        chunker: RecursiveChunker | None = None,
        enbedder: SentenceTransformerEmbedder | None = None,
        vector_store: PgVectorStore | None = None, 
        document_id: str | None = None,
        metadata: dict | None = None,
    ):
        self.loader_factory = loader_factory or LoaderFactory()
        self.chunker = chunker or RecursiveChunker()
        self.embedder = enbedder 
        self.vector_store = vector_store

        self.document_id = document_id
        self.metadata = metadata or {} 

    
    async def process(
        self,
        file_path: str | Path,
    ) -> list[Document]:

        # 1. Load
        await sendSSE(data={
                    "document_id": self.document_id,
                    "message": "Document loading",
                    "status": "loading", 
                    "file_path": str(file_path)
                    }, event="ingestion_status")
        documents = self.loader_factory.load(file_path)
        for document in documents:
            # PostgreSQL text columns reject NUL bytes sometimes emitted by PDF extraction.
            document.page_content = document.page_content.replace("\x00", "")
        await sendSSE(data={
                    "document_id": self.document_id,
                    "message": "Document loaded",
                    "status": "loaded",
                    "file_path": str(file_path),
                    "document_count": len(documents)
                }, event="ingestion_status")
        # 2. Chunk
        await sendSSE(data={
                    "document_id": self.document_id,
                    "message": "Document chunking",
                    "status": "chunking",
                    "file_path": str(file_path)
                }, event="ingestion_status")
        chunks = self.chunker.split(documents)
        await sendSSE(data={
                    "document_id": self.document_id,
                    "message": "Document chunked",
                    "status": "chunked",
                    "file_path": str(file_path),
                    "chunk_count": len(chunks)
                }, event="ingestion_status")

        print(f"[IngestionPipeline] Chunked {len(documents)} document(s) into {len(chunks)} chunk(s)")
        # 3. Embed
        await sendSSE(data={
                    "document_id": self.document_id,
                    "message": "Document embedding",
                    "status": "embedding",
                    "file_path": str(file_path)
                }, event="ingestion_status")
        embeddings = []
        batch_size = 16
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings.extend(
                self.embedder.embed_documents([chunk.page_content for chunk in batch])
            )
            embedded_count = min(start + batch_size, len(chunks))
            print(
                f"[IngestionPipeline] Embedded {embedded_count}/{len(chunks)} chunks",
                flush=True,
            )
            await sendSSE(data={
                "document_id": self.document_id,
                "message": "Document embedding progress",
                "status": "embedding",
                "file_path": str(file_path),
                "embedded_count": embedded_count,
                "total_chunks": len(chunks),
            }, event="ingestion_status")
        await sendSSE(data={
                    "document_id": self.document_id,
                    "message": "Document embedded",
                    "status": "embedded",
                    "file_path": str(file_path),
                    "embedding_count": len(embeddings)
                }, event="ingestion_status")
        # 4. Store
        await sendSSE(data={
                    "document_id": self.document_id,
                    "message": "Document storing",
                    "status": "storing",
                    "file_path": str(file_path)
                }, event="ingestion_status")
        await self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            document_id=self.document_id,
            metadata=[
                {
                    **chunk.metadata,
                    **self.metadata,
                }
                for chunk in chunks
            ],
        )
        await sendSSE(data={
            "document_id": self.document_id,
            "message": "Document stored",
            "status": "stored",
            "file_path": str(file_path),
            "stored_count": len(chunks)
        }, event="ingestion_status")
        return chunks


async def sendSSE(data: dict | None = None, event: str = "notification") -> None:
    """Helper function to send SSE notifications."""
    print(f"[sendSSE] Sending SSE event: {event} with data: {data}", flush=True)
    try:
        await asyncio.wait_for(
            publisher.publish(event=event, data=data),
            timeout=2,
        )
    except Exception as exc:
        print(f"[sendSSE] Publish skipped: {exc}", flush=True)
