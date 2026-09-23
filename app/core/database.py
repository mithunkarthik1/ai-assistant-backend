"""
Database connection and session management module.
Initializes the async SQLAlchemy engine, session maker, and database tables.
"""
import logging
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger("app.core.database")


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
    from app.chat_bot.model import ChatMessage  # noqa: F401

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
        logger.warning("Could not connect to database for extension setup: %s", e)

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
        logger.error("Failed to initialize database tables: %s", e, exc_info=True)
        raise


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
