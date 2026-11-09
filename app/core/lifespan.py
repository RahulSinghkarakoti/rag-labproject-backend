from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.core.config import settings
from app.core.model_config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_RERANKING_MODELS
from app.core.redis import close_redis_pool, init_redis_pool
from app.rag.embeddings import preload_embedding_models
from app.rag.model_registry import ModelRegistry
from app.rag.reranking import preload_reranking_models

logger = logging.getLogger(__name__)


def resolve_preload_models(
    supported_models: frozenset[str],
    configured_value: str | None,
    preload_all: bool = False,
) -> list[str]:
    """Resolve which models to preload based on environment settings.
    
    - If preload_all is True: returns all supported models.
    - If configured_value is "all": returns all supported models.
    - If configured_value is "none", empty, or None: returns empty list (deferred to lazy loading).
    - Otherwise parses comma-separated model names and validates against supported_models.
    """
    if preload_all:
        return sorted(supported_models)

    if not configured_value:
        return []

    val = configured_value.strip()
    if val.lower() == "none" or not val:
        return []

    if val.lower() == "all":
        return sorted(supported_models)

    selected = [m.strip() for m in val.split(",") if m.strip()]
    invalid = sorted(set(selected) - supported_models)
    if invalid:
        raise ValueError(
            f"Unknown model(s) configured for preload: {', '.join(invalid)}. "
            f"Supported models: {sorted(supported_models)}"
        )
    return selected


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_registry = ModelRegistry

    embedding_models = resolve_preload_models(
        SUPPORTED_EMBEDDING_MODELS,
        settings.PRELOAD_EMBEDDING_MODELS,
        preload_all=settings.PRELOAD_ALL_MODELS,
    )
    reranking_models = resolve_preload_models(
        SUPPORTED_RERANKING_MODELS,
        settings.PRELOAD_RERANKING_MODELS,
        preload_all=settings.PRELOAD_ALL_MODELS,
    )

    if embedding_models:
        logger.info("[Lifespan] Preloading embedding models: %s", embedding_models)
        preload_embedding_models(embedding_models)
    else:
        logger.info("[Lifespan] Skipping embedding model preloading (deferred to lazy loading)")

    if reranking_models:
        logger.info("[Lifespan] Preloading reranking models: %s", reranking_models)
        preload_reranking_models(reranking_models)
    else:
        logger.info("[Lifespan] Skipping reranking model preloading (deferred to lazy loading)")

    await init_redis_pool()
    yield
    await close_redis_pool()
