"""
FastAPI application entry point.
Configured following MDUPythonTeam/fastapi_skeleton modular layout.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.database import init_db
from src.langchain.indexing import index_company_policy
from src.rag.api import router as chat_router

# Setup root logger
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("src.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handling database initialization and vector indexing.
    Gracefully handles environments where PostgreSQL is not running locally.
    """
    logger.info("Initializing WorkPilot AI Assistant backend...")

    # 1. Database schema initialization
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.warning("Database connection failed (%s). App will continue using embedded storage.", e)

    # 2. Document vector indexing (PostgreSQL pgvector or embedded local fallback)
    try:
        indexed_count = await index_company_policy()
        logger.info("Company policy knowledge base ready (%d chunks indexed).", indexed_count)
    except Exception as e:
        logger.error("Error during initial vector indexing: %s", e, exc_info=True)

    yield

    logger.info("WorkPilot AI Assistant backend shutting down.")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Enterprise HR Policy RAG Assistant powered by LangChain and FastAPI",
    lifespan=lifespan,
)

# CORS Middleware
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
@app.get("/api/health", tags=["system"], include_in_schema=False)
@app.get("/api/v1/health", tags=["system"], include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify backend service availability."""
    return {"status": "ok", "app": settings.app_name}


# Include RAG router at /api/v1 (standard REST prefix), /api, and root for full frontend compatibility
app.include_router(chat_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api", include_in_schema=False)
app.include_router(chat_router, include_in_schema=False)
