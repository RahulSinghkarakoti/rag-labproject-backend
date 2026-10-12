from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
Base = declarative_base()

from app.core.config import settings


DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    # echo=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)



async def get_db():
    async with AsyncSessionLocal() as db:
        yield db