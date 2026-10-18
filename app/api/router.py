from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.knowledge_base import router as kb_router
from app.api.projects import router as project_router
from app.api.upload import router as upload_router
from app.api.sse import router as sse_router
from app.api.pipelines import router as pipeline_router
from app.api.threads import router as threads_router
from app.api.run import router as runs_router

def register_routers(app: FastAPI) -> None:
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(project_router)
    app.include_router(kb_router)
    app.include_router(upload_router)
    app.include_router(threads_router, tags=["Threads"])
    app.include_router(pipeline_router, tags=["Pipelines"])       
    app.include_router(runs_router, prefix="/api/v1", tags=["Runs"])
    app.include_router(sse_router, prefix="/api/v1/sse", tags=["SSE Stream"])

