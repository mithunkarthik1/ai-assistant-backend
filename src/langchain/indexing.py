"""
Document indexing module.
Loads company policy text, splits into structured chunks, and indexes embeddings into
PostgreSQL pgvector (if available) AND the local embedded vector store (service-free).
"""
import logging
import uuid
from pathlib import Path

from langchain_core.documents import Document as LangChainDoc
from sqlalchemy import text

from src.config import settings
from src.database import engine
from src.langchain.splitter import split_documents
from src.langchain.vectorstore import get_local_vector_store, store_documents

logger = logging.getLogger("src.langchain.indexing")

POLICY_DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
POLICY_FILENAME = "company_policy.txt"


def get_policy_file_path() -> Path:
    """
    Resolves the company policy text file path dynamically,
    supporting local development and Docker container execution.
    """
    raw_path = Path(settings.policy_file_path)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    if raw_path.exists():
        return raw_path

    backend_root = Path(__file__).resolve().parent.parent.parent
    candidate = backend_root / raw_path
    if candidate.exists():
        return candidate

    return raw_path


def load_policy_content() -> str:
    """
    Reads the company policy handbook from the configured text file.
    """
    path = get_policy_file_path()
    if not path.exists():
        logger.error("Company policy text file not found at: %s", path)
        raise FileNotFoundError(
            f"Company policy text file not found at: {path} (configured: {settings.policy_file_path})"
        )
    return path.read_text(encoding="utf-8")


async def index_company_policy(force_reindex: bool = False) -> int:
    """
    Indexes the company policy text file.
    Persists to both PostgreSQL (if available) and local embedded store.
    """
    try:
        policy_text = load_policy_content()
    except Exception as e:
        logger.error("Failed to load policy content for indexing: %s", e)
        return 0

    # 1. Check if already indexed in local embedded store
    local_chunks = get_local_vector_store()
    if not force_reindex and len(local_chunks) >= 30:
        logger.info("Company policy already indexed in local embedded vector store (%d chunks).", len(local_chunks))
        # Also ensure PostgreSQL has it if connected
        try:
            async with engine.connect() as conn:
                res = await conn.execute(
                    text("SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id"),
                    {"doc_id": str(POLICY_DOC_ID)},
                )
                row = res.fetchone()
                db_count = row[0] if row else 0
                if db_count < len(local_chunks):
                    logger.info("Syncing %d local chunks to PostgreSQL...", len(local_chunks))
                    raw_policy = LangChainDoc(
                        page_content=policy_text,
                        metadata={
                            "document_id": str(POLICY_DOC_ID),
                            "filename": POLICY_FILENAME,
                            "page": 1,
                        },
                    )
                    policy_chunks = split_documents([raw_policy])
                    await store_documents(policy_chunks)
        except Exception:
            pass # DB service may not be running locally; that's perfectly fine
        return len(local_chunks)

    # 2. Check if already indexed in PostgreSQL
    existing_count = 0
    vector_dim = 0
    try:
        async with engine.connect() as conn:
            res = await conn.execute(
                text("""
                SELECT count(*),
                       COALESCE(MAX(CASE WHEN embedding IS NOT NULL THEN array_length(embedding, 1) ELSE 0 END), 0)
                FROM langchain_pg_embedding
                WHERE cmetadata->>'document_id' = :doc_id
                """),
                {"doc_id": str(POLICY_DOC_ID)},
            )
            row = res.fetchone()
            existing_count = row[0] if row else 0
            vector_dim = row[1] if row else 0
    except Exception as e:
        logger.debug("Database check skipped (DB service offline or tables not ready): %s", e)

    if not force_reindex and existing_count >= 30 and vector_dim >= 384:
        logger.info("Company policy already indexed in PostgreSQL (%d chunks).", existing_count)
        return existing_count

    # 3. Perform fresh indexing
    try:
        logger.info("Generating fresh embeddings for Company Policy...")
        raw_policy = LangChainDoc(
            page_content=policy_text,
            metadata={
                "document_id": str(POLICY_DOC_ID),
                "filename": POLICY_FILENAME,
                "page": 1,
            },
        )
        policy_chunks = split_documents([raw_policy])
        if policy_chunks:
            await store_documents(policy_chunks)
            logger.info("Successfully indexed %d policy chunks into vector store.", len(policy_chunks))
            return len(policy_chunks)
    except Exception as e:
        logger.error("Error during policy document indexing: %s", e, exc_info=True)

    return 0
