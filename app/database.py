from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.database_url, echo=False, future=True, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create PostgreSQL extensions and initialize tables."""
    from app.chat_bot.model import ChatMessage  # noqa: F401

    # 1. Attempt to enable vector extension if available on PostgreSQL server
    async with engine.connect() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
        except Exception:
            await conn.rollback()

    # 2. Create application tables and fallback chunk tables
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


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
