import logging
import uuid
from pathlib import Path
from langchain_core.documents import Document as LangChainDoc
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.rag.splitter import split_documents
from app.rag.vectorstore import store_documents

logger = logging.getLogger("rag.indexing")

POLICY_DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
POLICY_FILENAME = "company_policy.txt"


def get_policy_file_path() -> Path:
    """
    Resolves the company policy text file path dynamically,
    supporting both local development and Docker container execution.
    """
    raw_path = Path(settings.policy_file_path)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    # Try relative to current working directory (e.g. /app or backend/)
    if raw_path.exists():
        return raw_path

    # Try relative to the backend project root (3 levels up from indexing.py)
    backend_root = Path(__file__).resolve().parent.parent.parent
    candidate = backend_root / raw_path
    if candidate.exists():
        return candidate

    return raw_path


def load_policy_content() -> str:
    """
    Reads the company policy handbook from the dedicated .txt file.
    """
    path = get_policy_file_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Company policy text file not found at: {path} (configured: {settings.policy_file_path})"
        )
    return path.read_text(encoding="utf-8")


async def index_company_policy(force_reindex: bool = False) -> int:
    """
    Indexes the company policy text file into PostgreSQL with structure-aware granular dense semantic vectors.
    Automatically detects if existing embeddings need migration or re-indexing.
    Returns the number of indexed chunks.
    """
    policy_text = load_policy_content()
    policy_bytes = policy_text.encode("utf-8")
    content_size = len(policy_bytes)

    # Check existing chunks and embedding dimensionality in langchain_pg_embedding
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

    if not force_reindex and existing_count >= 30 and vector_dim >= 384:
        logger.info("Company policy already indexed with %d-d semantic vectors (%d chunks).", vector_dim, existing_count)
        return existing_count

    if existing_count > 0:
        logger.info("Re-indexing %d existing chunks with granular semantic vector embeddings...", existing_count)
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id"),
                {"doc_id": str(POLICY_DOC_ID)},
            )

    logger.info("Indexing Company Policy (%d bytes) from %s with dense semantic vectors...", content_size, settings.policy_file_path)

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
        logger.info("Successfully indexed %d policy chunks into vectorstore.", len(policy_chunks))
        return len(policy_chunks)

    return 0
