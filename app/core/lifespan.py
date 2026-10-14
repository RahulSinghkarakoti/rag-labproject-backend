from contextlib import asynccontextmanager
import os
import sys

from fastapi import FastAPI

from app.core.redis import close_redis_pool, init_redis_pool
from app.rag.embeddings import preload_embedding_models
from app.rag.model_registry import ModelRegistry
from app.rag.reranking import preload_reranking_models
from app.core.model_config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_RERANKING_MODELS


def _models_to_preload(models: frozenset[str], label: str) -> list[str]:
    available = sorted(models)
    print(f"\n{label} models: {', '.join(available)}")
    while True:
        choice = input("Preload all, none, or enter comma-separated model names [all]: ").strip().lower()
        if not choice or choice == "all":
            return available
        if choice == "none":
            return []
        selected = [model.strip() for model in choice.split(",")]
        invalid = sorted(set(selected) - models)
        if not invalid:
            return selected
        print(f"Unknown {label.lower()} model(s): {', '.join(invalid)}")


def _configured_models(models: frozenset[str], environment_variable: str) -> list[str] | None:
    configured = os.getenv(environment_variable)
    if configured is None:
        return None
    if configured == "none":
        return []
    selected = sorted(models) if configured == "all" else configured.split(",")
    invalid = sorted(set(selected) - models)
    if invalid:
        raise ValueError(f"Unknown model(s) in {environment_variable}: {', '.join(invalid)}")
    return selected


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_registry = ModelRegistry
    embedding_models = _configured_models(SUPPORTED_EMBEDDING_MODELS, "PRELOAD_EMBEDDING_MODELS")
    reranking_models = _configured_models(SUPPORTED_RERANKING_MODELS, "PRELOAD_RERANKING_MODELS")
    if embedding_models is None and reranking_models is None and sys.stdin.isatty():
        embedding_models = _models_to_preload(SUPPORTED_EMBEDDING_MODELS, "Embedding")
        reranking_models = _models_to_preload(SUPPORTED_RERANKING_MODELS, "Reranking")
    if embedding_models is None:
        embedding_models = list(SUPPORTED_EMBEDDING_MODELS)
    if reranking_models is None:
        reranking_models = list(SUPPORTED_RERANKING_MODELS)
    preload_embedding_models(embedding_models)
    preload_reranking_models(reranking_models)
    await init_redis_pool()
    yield
    await close_redis_pool()
