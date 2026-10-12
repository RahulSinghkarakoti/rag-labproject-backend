from app.rag.model_registry import ModelRegistry


def test_registry_loads_each_model_once():
    ModelRegistry.clear()
    calls = []

    def load(name):
        calls.append(name)
        return object()

    first_embedding = ModelRegistry.get_embedding_model("embed", loader=load)
    second_embedding = ModelRegistry.get_embedding_model("embed", loader=load)
    first_reranker = ModelRegistry.get_reranking_model("rerank", loader=load)
    second_reranker = ModelRegistry.get_reranking_model("rerank", loader=load)

    assert first_embedding is second_embedding
    assert first_reranker is second_reranker
    assert calls == ["embed", "rerank"]
    ModelRegistry.clear()
