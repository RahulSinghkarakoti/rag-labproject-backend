from fastapi import FastAPI

from app.api.router import register_routers
from app.core.exception_handlers import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.middleware import setup_cors

from dotenv import load_dotenv 
load_dotenv()

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

setup_cors(app)
register_exception_handlers(app)
register_routers(app)
