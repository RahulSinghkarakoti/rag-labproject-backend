from pathlib import Path

from sqlalchemy import select

from app.core.sse import publisher
from app.database.database import AsyncSessionLocal
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.core.model_config import validate_embedding_model
from app.rag.pipeline.ingestion import IngestionPipeline
from app.rag.chunking import ChunkerFactory
from app.rag.loaders import LoaderFactory
from app.rag.loaders.detector import FileTypeDetector
from app.rag.vectorstore import PgVectorStore
from app.rag.embeddings import SentenceTransformerEmbedder

async def ingest_document(ctx: dict, document_id: str):
    async with AsyncSessionLocal() as db:

        # 1. Fetch document with row lock
        result = await db.execute(
            select(Document).where(Document.id == document_id).with_for_update()
        )

        document = result.scalar_one_or_none()
        if document is None:
            raise ValueError(
                f"Document {document_id} not found"
            )

        user_channel = "ingestion_channel"
        try:
            kb = await db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == document.knowledge_base_id))
            if kb is None:
                raise ValueError(f"Knowledge base for document {document_id} not found")
            if kb.user_id is not None:
                user_channel = f"ingestion_channel:{kb.user_id}"
            # 2. Mark processing
            document.status = "processing"
            await db.commit()
            await publisher.publish(
                data={
                    "document_id": document_id,
                    "status": "processing",
                    "filename": document.name,
                    "message": "Document ingestion started",
                },
                event="ingestion_status",
                channel=user_channel,
            )
            # 3. Build RAG components
            loader = LoaderFactory()
            detector = FileTypeDetector()
            file_type = detector.detect(document.storage_path)

            embedding_config = kb.embedding_config or {}
            embedding_model = embedding_config.get("model", "all-MiniLM-L6-v2")
            validate_embedding_model(embedding_model)
            embedder = SentenceTransformerEmbedder(model_name=embedding_model)

            chunk_config = kb.chunking_config or {}
            chunker = ChunkerFactory.create(
                strategy=chunk_config.get("strategy", "recursive"),
                chunk_size=chunk_config.get("chunk_size", 1000),
                chunk_overlap=chunk_config.get("chunk_overlap", 200),
                file_type=file_type,
                embedder=embedder,
            )

            vector_store = PgVectorStore(db)
            pipeline = IngestionPipeline(
                loader_factory=loader,
                chunker=chunker,
                enbedder=embedder,
                vector_store=vector_store,
                document_id=document_id,
                sse_channel=user_channel,
                metadata={
                    "filename": document.name,
                    "type": document.mime_type,
                    "file_type": file_type,
                    "size_bytes": document.size_bytes,
                    "storage_path": document.storage_path,
                    "knowledge_base_id": str(document.knowledge_base_id),
                    "embedding_model": embedder.model_name,
                    "embedding_dimension": embedder.dimension(),
                    "chunking_strategy": chunker.strategy_name,
                    "chunking_config": chunk_config,
                },
            )
            # 4. Run ingestion
            await pipeline.process(
                file_path=Path(document.storage_path)
            )

            # 5. Mark completed
            document.status = "completed"
            await db.commit()
            await publisher.publish(
                data={
                    "document_id": document_id,
                    "status": "completed",
                    "filename": document.name,
                    "message": "Document ingestion completed",
                },
                event="ingestion_status",
                channel=user_channel,
            )

            return {
                "document_id": document_id,
                "status": "completed",
            }

        except Exception as exc:

            await db.rollback()

            # Re-fetch/update if necessary
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )

            document = result.scalar_one_or_none()

            if document:
                document.status = "failed"
                await db.commit()
                await publisher.publish(
                    data={
                        "document_id": document_id,
                        "status": "failed",
                        "filename": document.name,
                        "error": str(exc),
                        "message": "Document ingestion failed",
                    },
                    event="ingestion_status",
                    channel=user_channel,
                )

            raise
