import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.document import DocumentChunk
from app.rag.vectorstore.base import BaseVectorStore

from app.models.document import Document, DocumentChunk
from app.models.knowledge_base import KnowledgeBase

class PgVectorStore(BaseVectorStore):

    def __init__(self, db: AsyncSession):
        self.db = db

    async def add_documents(
        self,
        documents: list[Any],
        embeddings: list[list[float]],
        *,
        document_id: str,
        metadata: list[dict] | None = None,
    ) -> None:

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents and embeddings must match."
            )

        if metadata is not None and len(metadata) != len(documents):
            raise ValueError(
                "Number of metadata entries must match documents."
            )

        document_uuid = uuid.UUID(str(document_id))

        chunks = []

        for index, (document, embedding) in enumerate(
            zip(documents, embeddings)
        ):
            doc_metadata = metadata[index] if metadata else {}

            # Merge metadata coming from the LangChain document
            langchain_metadata = getattr(
                document,
                "metadata",
                {},
            ) or {}

            combined_metadata = {
                **langchain_metadata,
                **doc_metadata,
            }

            chunk = DocumentChunk(
                document_id=document_uuid,
                chunk_index=index,
                content=document.page_content,
                embedding=embedding,
                page_number=combined_metadata.get("page"),
                section=combined_metadata.get("section"),
                embedding_model=combined_metadata.get(
                    "embedding_model"
                ),
                metadata_=combined_metadata,
            )

            chunks.append(chunk)

        self.db.add_all(chunks)
        print(f"[PgVectorStore] Added {len(chunks)} chunks for document {document_id}")
        await self.db.flush()

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[DocumentChunk]:

        filters = filters or {}

        result = await self.similarity_search_with_scores(
            query_embedding,
            top_k=top_k,
            filters=filters,
        )
        return [document for document, _ in result]

    async def similarity_search_with_scores(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        stmt = (
            select(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
        )

        document_id = filters.get("document_id")

        if document_id:
            stmt = stmt.where(
                DocumentChunk.document_id
                == uuid.UUID(str(document_id))
            )

        knowledge_base_id = filters.get("knowledge_base_id")

        if knowledge_base_id:
            if isinstance(knowledge_base_id, (list, tuple, set)):
                stmt = stmt.where(
                    Document.knowledge_base_id.in_(
                        [uuid.UUID(str(value)) for value in knowledge_base_id]
                    )
                )
            else:
                stmt = stmt.where(
                    Document.knowledge_base_id
                    == uuid.UUID(str(knowledge_base_id))
                )

        project_id = filters.get("project_id")

        if project_id:
            stmt = stmt.join(
                KnowledgeBase,
                KnowledgeBase.id == Document.knowledge_base_id,
            )

            stmt = stmt.where(
                KnowledgeBase.project_id
                == uuid.UUID(str(project_id))
            )

        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        stmt = stmt.add_columns(distance.label("distance")).order_by(distance).limit(top_k)

        result = await self.db.execute(stmt)
        return [(chunk, 1.0 - float(distance)) for chunk, distance in result.all()]

    async def keyword_search(
        self,
        query: str,
        *,
        filters: dict | None = None,
    ) -> list[DocumentChunk]:
        filters = filters or {}
        terms = [term for term in query.split() if term]
        stmt = select(DocumentChunk).options(joinedload(DocumentChunk.document)).join(Document, Document.id == DocumentChunk.document_id)
        if terms:
            # shortcut: SQL LIKE narrows candidates, acceptable for current corpora; move BM25 to database FTS at scale.
            from sqlalchemy import or_
            stmt = stmt.where(or_(*(DocumentChunk.content.ilike(f"%{term}%") for term in terms)))
        knowledge_base_id = filters.get("knowledge_base_id")
        if knowledge_base_id:
            ids = knowledge_base_id if isinstance(knowledge_base_id, (list, tuple, set)) else [knowledge_base_id]
            stmt = stmt.where(Document.knowledge_base_id.in_([uuid.UUID(str(value)) for value in ids]))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def delete_document(
        self,
        document_id: str,
    ) -> None:

        document_uuid = uuid.UUID(str(document_id))

        stmt = delete(DocumentChunk).where(
            DocumentChunk.document_id == document_uuid
        )

        await self.db.execute(stmt)

        await self.db.flush()
