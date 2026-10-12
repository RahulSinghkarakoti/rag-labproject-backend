from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.embeddings import sentence_transformer as embedding_module


def test_embedder_reuses_loaded_model(monkeypatch):
    calls = []

    class FakeModel:
        pass

    def load_model(model_name):
        calls.append(model_name)
        return FakeModel()

    embedding_module._load_model.cache_clear()
    monkeypatch.setattr(embedding_module, "SentenceTransformer", load_model)

    first = SentenceTransformerEmbedder("test-model")
    second = SentenceTransformerEmbedder("test-model")

    assert first.model is second.model
    assert calls == ["test-model"]
    embedding_module._load_model.cache_clear()


def test_sentence_transformer_embedding():

    embedder = SentenceTransformerEmbedder()

    texts = [
        "RAG LAB is a retrieval augmented generation system.",
        "Documents are split into chunks.",
        "Embeddings represent text as vectors.",
    ]

    embeddings = embedder.embed_documents(texts)

    assert len(embeddings) == len(texts)

    dimension = embedder.dimension()

    assert dimension > 0

    for embedding in embeddings:
        assert len(embedding) == dimension

    print(f"\nModel: {embedder.model_name}")
    print(f"Dimension: {dimension}")
    print(f"Documents: {len(embeddings)}")
    print(f"First vector length: {len(embeddings[0])}")
    print(f"First vector: {embeddings[0][:10]}")


def test_query_embedding():

    embedder = SentenceTransformerEmbedder()

    query = "How does RAG work?"

    embedding = embedder.embed_query(query)

    assert len(embedding) == embedder.dimension()

    print(f"\nQuery dimension: {len(embedding)}")
    print(f"Query vector: {embedding[:10]}")
