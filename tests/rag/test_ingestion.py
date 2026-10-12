from pathlib import Path

from app.rag.pipeline.ingestion import IngestionPipeline
from app.database.database import get_db
from app.rag.vectorstore import PgVectorStore


def test_ingestion_pipeline():

    vector_store = PgVectorStore(db=get_db())
    pipeline = IngestionPipeline()


    chunks = pipeline.process(
        Path("tests/data/sample.txt")
    )

    assert chunks

    print(f"\nGenerated chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {index + 1} ---")
        print(chunk.page_content)
        print(chunk.metadata)