from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.database import get_db

router = APIRouter(tags=["Health"])


@router.get("/")
def root():
    return {"message": "RAG API is running"}


@router.get("/health")
def health():
    return {
        "success": True,
        "message": "API is healthy.",
        "data": {
            "status": "ok",
        },
    }


@router.get("/db-check")
async def test_db_connection(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT current_database();"))
    database = result.scalar()
    return {
        "status": "connected",
        "database": database,
        "app_name": settings.APP_NAME,
    }
