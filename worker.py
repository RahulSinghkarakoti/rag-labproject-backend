import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from arq import create_pool
from arq.connections import RedisSettings

from app.core.config import settings

if settings.HF_TOKEN and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN
from app.core.lifespan import resolve_preload_models
from app.core.model_config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_RERANKING_MODELS
from app.rag.embeddings import preload_embedding_models
from app.rag.model_registry import ModelRegistry
from app.rag.reranking import preload_reranking_models
from app.workers import ingest_document

logger = logging.getLogger(__name__)


async def startup(ctx):
    ctx["model_registry"] = ModelRegistry

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
        print(f"[Worker] Preloading embedding models: {embedding_models}", flush=True)
        preload_embedding_models(embedding_models)
    else:
        print("[Worker] Skipping embedding model preload (deferred to lazy loading).", flush=True)

    if reranking_models:
        print(f"[Worker] Preloading reranking models: {reranking_models}", flush=True)
        preload_reranking_models(reranking_models)
    else:
        print("[Worker] Skipping reranking model preload (deferred to lazy loading).", flush=True)

    print("[Worker] Ready.", flush=True)


async def shutdown(ctx):
    print("[Worker] Shutting down...", flush=True)


# ARQ Worker Settings
class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
    job_timeout = 1800

    on_startup = startup
    on_shutdown = shutdown
