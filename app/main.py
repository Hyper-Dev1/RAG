import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.database import init_db
from app.api.router import api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    init_db()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="RAG Question Generator API",
    description="API for generating questions based on book paragraphs using RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok"}
