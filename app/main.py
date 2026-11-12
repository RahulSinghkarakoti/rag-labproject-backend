import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.core.config import settings

if getattr(settings, "HF_TOKEN", None) and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

from app.api.router import register_routers
from app.core.exception_handlers import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.middleware import setup_cors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

setup_cors(app)
register_exception_handlers(app)
register_routers(app)
