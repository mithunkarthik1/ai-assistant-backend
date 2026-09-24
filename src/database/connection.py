"""Database engine, session management, and schema initialization."""

import logging
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings


logger = logging.getLogger("src.database")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""


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
    """Initialize application and pgvector-compatible tables."""
    from src.rag.model import ChatMessage  # noqa: F401

    try:
        async with engine.connect() as conn:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.commit()
                logger.info("PostgreSQL pgvector extension verified/enabled.")
            except Exception as exc:
                await conn.rollback()
                logger.debug("vector extension not enabled or not needed: %s", exc)
    except Exception as exc:
        logger.warning("Database connection unavailable during extension setup: %s", exc)

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
            try:
                await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    id VARCHAR PRIMARY KEY,
                    collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
                    embedding vector(384),
                    document VARCHAR,
                    cmetadata JSONB
                )
                """))
                await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_hnsw
                ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops)
                """))
            except Exception as exc:
                logger.debug("pgvector native type fallback: %s", exc)
                await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    id VARCHAR PRIMARY KEY,
                    collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
                    embedding float8[],
                    document VARCHAR,
                    cmetadata JSONB
                )
                """))
        logger.info("Application, pgvector tables, and HNSW indexes initialized successfully.")
    except Exception as exc:
        logger.warning("Could not initialize PostgreSQL tables: %s (Running in local mode)", exc)


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield one async database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as exc:
            await session.rollback()
            logger.error("Database session error; rolled back: %s", exc)
            raise
        finally:
            await session.close()
