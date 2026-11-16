#!/usr/bin/env python3
"""
Ingestion script for RAG Evaluation Benchmark.
Ingests 21 documents from /home/rahulkarakoti/Downloads/archive/documents.csv
into PostgreSQL/PgVector under a dedicated KnowledgeBase: "Eval - Enter The Gungeon".
"""

import asyncio
import csv
import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Increase CSV max field size for large text bodies
csv.field_size_limit(sys.maxsize)

from sqlalchemy import delete, select
from langchain_core.documents import Document as LCDocument

from app.database.database import AsyncSessionLocal
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentChunk
from app.models.user import User
from app.rag.chunking import RecursiveChunker
from app.rag.embeddings import SentenceTransformerEmbedder

DEFAULT_DATASET_DIR = EVAL_DIR / "dataset"
KB_NAME = "Eval - Enter The Gungeon"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


async def get_or_create_eval_user(session) -> int:
    result = await session.execute(select(User).order_by(User.id.asc()))
    user = result.scalars().first()
    if not user:
        raise RuntimeError("No user found in the database. Please ensure at least one user exists.")
    return user.id


async def ingest_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR, force_reindex: bool = False):
    docs_csv = dataset_dir / "documents.csv"
    if not docs_csv.exists():
        raise FileNotFoundError(f"documents.csv not found at {docs_csv}")

    print(f"🚀 Initializing Eval Ingestion from: {docs_csv}")
    print(f"   Model: {EMBEDDING_MODEL} | Chunk Size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

    chunker = RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedder = SentenceTransformerEmbedder(model_name=EMBEDDING_MODEL)

    async with AsyncSessionLocal() as session:
        user_id = await get_or_create_eval_user(session)

        # Check for existing eval knowledge base
        kb_query = await session.execute(
            select(KnowledgeBase).where(KnowledgeBase.name == KB_NAME)
        )
        kb = kb_query.scalars().first()

        if kb:
            if not force_reindex:
                # Check document count
                doc_count_query = await session.execute(
                    select(Document).where(Document.knowledge_base_id == kb.id)
                )
                existing_docs = doc_count_query.scalars().all()
                if len(existing_docs) >= 21:
                    print(f"✅ Knowledge Base '{KB_NAME}' already contains {len(existing_docs)} documents.")
                    print(f"   KB ID: {kb.id}")
                    print("   (Pass --reindex to wipe and re-embed)")
                    return kb.id

            print(f"⚠️ Reindexing: Removing previous evaluation documents in KB {kb.id}...")
            await session.execute(delete(Document).where(Document.knowledge_base_id == kb.id))
            await session.commit()
        else:
            print(f"📦 Creating new Knowledge Base: '{KB_NAME}' for user {user_id}")
            kb = KnowledgeBase(
                id=uuid.uuid4(),
                user_id=user_id,
                name=KB_NAME,
                description="Evaluation benchmark corpus for Enter The Gungeon dataset",
                embedding_config={"model": EMBEDDING_MODEL, "dimension": 384},
                chunking_config={"strategy": "recursive", "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
                indexing_config={"strategy": "hnsw"},
                metadata_={"dataset_dir": str(dataset_dir), "source": "rag_eval"},
            )
            session.add(kb)
            await session.commit()
            await session.refresh(kb)

        kb_id = kb.id
        print(f"📂 Reading {docs_csv}...")
        raw_rows = []
        with open(docs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(row)

        print(f"📄 Found {len(raw_rows)} documents to process.")

        total_chunks = 0
        for i, row in enumerate(raw_rows):
            doc_idx = int(row["index"])
            source_url = row["source_url"]
            text = row["text"]
            title = source_url.split("/")[-1].replace("_", " ")

            doc = Document(
                id=uuid.uuid4(),
                knowledge_base_id=kb_id,
                name=f"[{doc_idx}] {title}",
                mime_type="text/plain",
                size_bytes=len(text.encode("utf-8")),
                status="completed",
                metadata_={
                    "document_index": doc_idx,
                    "source_url": source_url,
                    "title": title,
                },
            )
            session.add(doc)
            await session.flush()

            # Split document into chunks
            lc_docs = [LCDocument(page_content=text, metadata={"source": source_url, "document_index": doc_idx})]
            chunks = chunker.split(lc_docs)
            chunk_texts = [c.page_content for c in chunks]

            if chunk_texts:
                embeddings = embedder.embed_documents(chunk_texts)
                db_chunks = []
                for c_idx, (c, emb) in enumerate(zip(chunks, embeddings)):
                    chunk_obj = DocumentChunk(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        chunk_index=c_idx,
                        content=c.page_content,
                        embedding=emb,
                        embedding_model=EMBEDDING_MODEL,
                        embedding_dimension=384,
                        metadata_={
                            "document_index": doc_idx,
                            "source_url": source_url,
                            "title": title,
                            "chunk_index": c_idx,
                        },
                    )
                    db_chunks.append(chunk_obj)

                session.add_all(db_chunks)
                await session.flush()
                total_chunks += len(db_chunks)

            print(f"   [{i+1}/{len(raw_rows)}] Indexed Doc #{doc_idx}: '{title}' -> {len(chunks)} chunks")

        await session.commit()
        print(f"🎉 Ingestion Complete!")
        print(f"   Total Documents: {len(raw_rows)}")
        print(f"   Total Chunks Created: {total_chunks}")
        print(f"   Knowledge Base ID: {kb_id}")
        return kb_id


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv
    asyncio.run(ingest_dataset(force_reindex=force_reindex))
