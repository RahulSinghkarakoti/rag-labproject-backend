import asyncio
from arq import create_pool
from arq.connections import RedisSettings
from app.workers import ingest_document
from app.rag.embeddings import preload_embedding_models
from app.rag.model_registry import ModelRegistry
from app.rag.reranking import preload_reranking_models


async def startup(ctx):
    ctx["model_registry"] = ModelRegistry
    preload_embedding_models()
    preload_reranking_models()
    print("[Worker] Ready; embedding and reranking models preloaded.", flush=True)


async def shutdown(ctx):
    print("[Worker] Shutting down...")



# ARQ Worker Settings
class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings(host="localhost", port=6379)
    job_timeout = 1800

    on_startup = startup
    on_shutdown = shutdown
