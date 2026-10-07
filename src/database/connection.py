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
    """Initialize relational application tables in the database."""
    from src.rag.model import ChatMessage, ChatSession, Document, DocumentChunk  # noqa: F401

    try:
        async with engine.begin() as conn:
            # 1. Ensure all base application tables exist first
            await conn.run_sync(Base.metadata.create_all)

            # 2. Migrate legacy column names on documents table if present
            await conn.execute(text("""
            DO $migrate$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'documents' AND column_name = 'id'
                ) THEN
                    ALTER TABLE documents RENAME COLUMN id TO document_id;
                END IF;
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'documents' AND column_name = 'filename'
                ) THEN
                    ALTER TABLE documents RENAME COLUMN filename TO file_name;
                END IF;
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'documents' AND column_name = 'file_size'
                ) THEN
                    ALTER TABLE documents ALTER COLUMN file_size DROP NOT NULL;
                END IF;
            END $migrate$;
            """))
            # Ensure required columns exist on documents
            await conn.execute(text("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64) DEFAULT '';
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0;
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
            """))
            await conn.execute(text("""
            ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS session_id VARCHAR(255);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);
            """))
        logger.info("Application relational tables initialized successfully.")
    except Exception as exc:
        logger.warning("Database initialization notice: %s", exc)


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
