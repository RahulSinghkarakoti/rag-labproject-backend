import asyncio
from arq import create_pool
from arq.connections import RedisSettings
from app.workers import ingest_document
from app.rag.model_registry import ModelRegistry


async def startup(ctx):
    ctx["model_registry"] = ModelRegistry
    print("[Worker] Ready; embedding models are selected per knowledge base.")


async def shutdown(ctx):
    print("[Worker] Shutting down...")



# ARQ Worker Settings
class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings(host="localhost", port=6379)

    on_startup = startup
    on_shutdown = shutdown
