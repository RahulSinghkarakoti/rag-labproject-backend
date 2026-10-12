from typing import Any

from app.rag.model_registry import ModelRegistry
from app.rag.reranking.base import BaseReranker


def _load_model(model_name: str):
    return ModelRegistry.get_reranking_model(model_name)


_load_model.cache_clear = ModelRegistry.clear_reranking_models


class CrossEncoderReranker(BaseReranker):
    def __init__(self, *, model: str, top_k: int, batch_size: int, score_threshold: float | None,
                 return_documents: bool):
        self.model = _load_model(model)
        self.top_k = top_k
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.return_documents = return_documents

    def rerank(self, queries: list[str], documents: list[Any]):
        pairs = [(query, document.content) for query in queries for document in documents]
        if not pairs:
            return []

        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        best_scores = [max(scores[index::len(queries)]) for index in range(len(documents))]
        ranked = sorted(zip(documents, best_scores), key=lambda item: item[1], reverse=True)
        if self.score_threshold is not None:
            ranked = [item for item in ranked if item[1] >= self.score_threshold]
        ranked = ranked[:self.top_k]
        return [document for document, _ in ranked] if self.return_documents else [score for _, score in ranked]


def rerank_documents(queries: list[str], documents: list[Any], config) -> list[Any]:
    return CrossEncoderReranker(
        model=config.model,
        top_k=config.top_k,
        batch_size=config.batch_size,
        score_threshold=config.score_threshold,
        return_documents=config.return_documents,
    ).rerank(queries, documents)
