from fastapi import APIRouter, FastAPI

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.knowledge_base import router as kb_router
from app.api.pipelines import router as pipeline_router
from app.api.projects import router as project_router
from app.api.run import router as runs_router
from app.api.sse import router as sse_router
from app.api.threads import router as threads_router

# Unified API router aggregating all domain sub-routers
api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(project_router)
api_router.include_router(kb_router)
api_router.include_router(documents_router)
api_router.include_router(threads_router)
api_router.include_router(pipeline_router)
api_router.include_router(runs_router)
api_router.include_router(sse_router)


def register_routers(app: FastAPI) -> None:
    """Register all application routes with FastAPI app instance.

    Mounts:
      - Health checks at root (/, /health, /db-check)
      - Standard versioned API at /api/v1
    """
    app.include_router(health_router)
    app.include_router(api_router, prefix="/api/v1")
    # Backward-compatible unversioned API (commented out as frontend now uses /api/v1):
    # app.include_router(api_router, prefix="/api")
