from pathlib import Path
import pytest
from app.rag.loaders.factory import LoaderFactory


def test_pdf_loader():
    file_path = Path("tests/data/Module_01_SacredTexts.pdf")
    if not file_path.exists():
        pytest.skip(f"Test fixture missing: {file_path}")

    factory = LoaderFactory()

    documents = factory.load(file_path)

    assert len(documents) > 0
    assert documents[0].page_content

    print(f"\nLoaded PDF documents: {len(documents)}") 


def test_text_loader():
    file_path = Path("tests/data/RAG Lab — Project Requirements & Technical Specification.md")
    if not file_path.exists():
        pytest.skip(f"Test fixture missing: {file_path}")
    factory = LoaderFactory()

    documents = factory.load(file_path)

    assert len(documents) > 0
    assert documents[0].page_content

    print(f"\nLoaded Text documents: {len(documents)}") 