from types import SimpleNamespace

from app.rag.reranking import rerank_documents
from app.schemas.thread import RerankingConfig


def test_reranker_ranks_candidates_and_applies_threshold(monkeypatch):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return [0.1, 0.9, 0.8, 0.2]

    monkeypatch.setattr("app.rag.reranking.reranker._load_model", lambda _: FakeModel())
    documents = [SimpleNamespace(content="first"), SimpleNamespace(content="second")]
    config = RerankingConfig(top_k=2, score_threshold=0.5)

    result = rerank_documents(["q1", "q2"], documents, config)

    assert result == [documents[1], documents[0]]


def test_reranker_can_return_scores(monkeypatch):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return [0.4]

    monkeypatch.setattr("app.rag.reranking.reranker._load_model", lambda _: FakeModel())
    config = RerankingConfig(return_documents=False)

    assert rerank_documents(["q"], [SimpleNamespace(content="doc")], config) == [0.4]
