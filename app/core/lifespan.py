from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.redis import close_redis_pool, init_redis_pool
from app.rag.embeddings import preload_embedding_models
from app.rag.model_registry import ModelRegistry
from app.rag.reranking import preload_reranking_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_registry = ModelRegistry
    preload_embedding_models()
    preload_reranking_models()
    await init_redis_pool()
    yield
    await close_redis_pool()
