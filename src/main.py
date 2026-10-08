"""
FastAPI application entry point.
Configured following MDUPythonTeam/fastapi_skeleton modular layout.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from src.agents.api import router as agents_router
from src.assistant.api import router as assistant_router
from src.core.config import settings
from src.database.connection import engine, init_db
from src.database.qdrant import qdrant_service
from src.rag.api import documents_router, rag_router, router as chat_router
from src.auth.api import router as auth_router
from src.rag.service import index_company_policy


# Setup root logger
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("src.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handling database initialization, Qdrant collection setup, and vector indexing.
    Gracefully handles environments where PostgreSQL is not running locally.
    """
    logger.info("Initializing AI Assistant backend...")

    # 1. Database schema initialization
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.warning("Database connection failed (%s). App will continue using embedded storage.", e)

    # 2. Qdrant Cloud collection initialization
    try:
        if qdrant_service.is_configured():
            qdrant_service.connect()
            qdrant_service.create_collection(dimension=settings.embedding_dimension)
            logger.info("Qdrant collection '%s' verified.", qdrant_service.collection_name)
    except Exception as e:
        logger.warning("Qdrant collection setup warning: %s", e)

    # 3. Document vector indexing (Qdrant Cloud + PostgreSQL pgvector fallback)
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
@app.get("/api/v1/health", tags=["system"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint to verify backend service availability, PostgreSQL, and Qdrant."""
    pg_status = "healthy"
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        pg_status = f"unhealthy ({exc})"

    qdrant_status = qdrant_service.health_check()
    embedding_status = "available" if settings.embedding_model else "unavailable"

    overall_status = (
        "healthy"
        if (pg_status == "healthy" and qdrant_status.get("status") in ("healthy", "not_configured"))
        else "degraded"
    )

    return {
        "status": overall_status,
        "app": settings.app_name,
        "postgresql": pg_status,
        "qdrant": qdrant_status,
        "embedding": embedding_status,
    }


# Include RAG router at /api/v1 (standard REST prefix), /api, and root for full frontend compatibility
app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.include_router(rag_router)
# Separate agentic workflow; the existing chat/RAG router remains unchanged.
app.include_router(agents_router, prefix="/api/v1")
app.include_router(assistant_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
