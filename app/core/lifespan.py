from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.redis import close_redis_pool, init_redis_pool
from app.rag.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_registry = ModelRegistry
    await init_redis_pool()
    yield
    await close_redis_pool()
