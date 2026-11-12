from typing import Any

from app.core.model_config import SUPPORTED_RERANKING_MODELS
from app.rag.model_registry import ModelRegistry
from app.rag.reranking.base import BaseReranker


def _load_model(model_name: str):
    return ModelRegistry.get_reranking_model(model_name)


_load_model.cache_clear = ModelRegistry.clear_reranking_models


def preload_reranking_models(model_names: list[str] | None = None) -> None:
    for model_name in SUPPORTED_RERANKING_MODELS if model_names is None else model_names:
        print(f"[ModelRegistry] Loading reranking model: {model_name}", flush=True)
        _load_model(model_name)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, *, model: str, top_k: int, batch_size: int, score_threshold: float | None,
                 return_documents: bool):
        self.model = _load_model(model)
        self.top_k = top_k
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.return_documents = return_documents

    def rerank(self, queries: list[str], documents: list[Any]):
        if not queries or not documents:
            return []

        # Clamp document text length to 1500 chars (~384 tokens) to prevent quadratic attention blowup
        pairs = [
            (query, (getattr(document, "content", "") or str(document))[:1500])
            for query in queries
            for document in documents
        ]
        if not pairs:
            return []

        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        num_docs = len(documents)
        # Pairs are organized as query0 (doc0..docN-1), query1 (doc0..docN-1)...
        # For document index d_idx, its score with query q_idx is at q_idx * num_docs + d_idx
        best_scores = [
            max(scores[q_idx * num_docs + d_idx] for q_idx in range(len(queries)))
            for d_idx in range(num_docs)
        ]
        ranked = sorted(zip(documents, best_scores, strict=False), key=lambda item: item[1], reverse=True)
        if self.score_threshold is not None:
            ranked = [item for item in ranked if item[1] >= self.score_threshold]
        ranked = ranked[:self.top_k]
        if self.return_documents:
            for document, score in ranked:
                document.rerank_score = float(score)
            return [document for document, _ in ranked]
        else:
            return [score for _, score in ranked]


def rerank_documents(queries: list[str], documents: list[Any], config) -> list[Any]:
    return CrossEncoderReranker(
        model=config.model,
        top_k=config.top_k,
        batch_size=config.batch_size,
        score_threshold=config.score_threshold,
        return_documents=config.return_documents,
    ).rerank(queries, documents)
