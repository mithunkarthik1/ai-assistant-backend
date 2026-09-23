"""
Database configuration, session management, and application settings module.
Combines environment configuration (Pydantic Settings) and async SQLAlchemy infrastructure.
Supports resilient fallback if PostgreSQL is not running locally.
"""
import logging
from collections.abc import AsyncIterator

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ============================================================
# 1. APPLICATION & DATABASE CONFIGURATION
# ============================================================

class Settings(BaseSettings):
    """
    Centralized configuration for the RAG Chatbot application.
    Loaded automatically from .env with strongly typed defaults.
    """
    app_name: str = "RAG Chatbot API"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_db"
    policy_file_path: str = "data/company_policy.txt"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 6
    log_level: str = "INFO"
    cors_origins: str = "*"

    # LLM Settings (Groq, OpenAI, xAI Grok, Ollama, OpenRouter)
    llm_api_key: str | None = None
    llm_model: str = "openai/gpt-oss-20b"
    llm_base_url: str | None = None

    # Embeddings & Vector Settings
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    min_similarity: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
logger = logging.getLogger("src.database")


# ============================================================
# 2. SQLALCHEMY ENGINE & SESSION SETUP
# ============================================================

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """
    Initializes database schema, creates the vector extension if supported,
    and sets up langchain_pg_embedding tables.
    """
    from src.rag.model import ChatMessage  # noqa: F401

    # 1. Attempt to enable vector extension if available
    try:
        async with engine.connect() as conn:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.commit()
                logger.info("PostgreSQL pgvector extension verified/enabled.")
            except Exception as e:
                await conn.rollback()
                logger.debug("vector extension not enabled or not needed: %s", e)
    except Exception as e:
        logger.warning("Database connection unavailable during extension setup: %s", e)

    # 2. Create application tables and embedding tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY,
                name VARCHAR,
                cmetadata JSONB
            )
            """))
            await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id VARCHAR PRIMARY KEY,
                collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
                embedding float8[],
                document VARCHAR,
                cmetadata JSONB
            )
            """))
        logger.info("Application and embedding tables initialized successfully.")
    except Exception as e:
        logger.warning("Could not initialize PostgreSQL tables: %s (Running in local mode)", e)


async def get_db() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency yielding an async database session per request.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error("Database session error; rolled back: %s", e)
            raise
        finally:
            await session.close()
