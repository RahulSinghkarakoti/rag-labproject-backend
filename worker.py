import asyncio
from arq import create_pool
from arq.connections import RedisSettings
from app.workers import ingest_document
from app.rag.embeddings import preload_embedding_models
from app.rag.model_registry import ModelRegistry
from app.rag.reranking import preload_reranking_models
from app.core.config import settings
from app.core.lifespan import _configured_models
from app.core.model_config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_RERANKING_MODELS


async def startup(ctx):
    ctx["model_registry"] = ModelRegistry
    preload_embedding_models(_configured_models(SUPPORTED_EMBEDDING_MODELS, "PRELOAD_EMBEDDING_MODELS"))
    preload_reranking_models(_configured_models(SUPPORTED_RERANKING_MODELS, "PRELOAD_RERANKING_MODELS"))
    print("[Worker] Ready; embedding and reranking models preloaded.", flush=True)


async def shutdown(ctx):
    print("[Worker] Shutting down...")



# ARQ Worker Settings
class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
    job_timeout = 1800

    on_startup = startup
    on_shutdown = shutdown
