"""Process-local registry for heavyweight embedding and reranking models."""

from threading import RLock
from typing import Any, Callable


class ModelRegistry:
    embedding_models: dict[str, Any] = {}
    reranking_models: dict[str, Any] = {}
    _lock = RLock()

    @classmethod
    def get_embedding_model(cls, model_name: str, loader: Callable[[str], Any] | None = None) -> Any:
        with cls._lock:
            if model_name not in cls.embedding_models:
                if loader is None:
                    from sentence_transformers import SentenceTransformer

                    loader = SentenceTransformer
                cls.embedding_models[model_name] = loader(model_name)
            return cls.embedding_models[model_name]

    @classmethod
    def get_reranking_model(cls, model_name: str, loader: Callable[[str], Any] | None = None) -> Any:
        with cls._lock:
            if model_name not in cls.reranking_models:
                if loader is None:
                    from sentence_transformers import CrossEncoder

                    loader = CrossEncoder
                cls.reranking_models[model_name] = loader(model_name)
            return cls.reranking_models[model_name]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls.embedding_models.clear()
            cls.reranking_models.clear()

    @classmethod
    def clear_embedding_models(cls) -> None:
        with cls._lock:
            cls.embedding_models.clear()

    @classmethod
    def clear_reranking_models(cls) -> None:
        with cls._lock:
            cls.reranking_models.clear()


model_registry = ModelRegistry

__all__ = ["ModelRegistry", "model_registry"]
