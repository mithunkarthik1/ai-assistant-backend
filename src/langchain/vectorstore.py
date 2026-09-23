"""
Vector storage module supporting both PostgreSQL pgvector and local embedded file storage.

How we use vector search without a running database service:
- When PostgreSQL + pgvector is running, embeddings and chunks are stored in `langchain_pg_embedding`.
- When NO database service is running locally, embeddings and chunks are stored in an embedded
  local JSON store (`data/local_vector_store.json`) and searched via CPU NumPy cosine similarity.
  This allows 100% of RAG features to work locally without Docker or database services.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from src.config import settings
from src.database import engine
from src.langchain.embeddings import get_embedding_model

logger = logging.getLogger("src.langchain.vectorstore")

LOCAL_VECTOR_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "local_vector_store.json"


def get_local_vector_store() -> list[dict[str, Any]]:
    """
    Reads the local file-based vector store without requiring any external database service.
    """
    if not LOCAL_VECTOR_STORE_PATH.exists():
        return []
    try:
        content = LOCAL_VECTOR_STORE_PATH.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as e:
        logger.error("Failed to read local vector store file: %s", e)
        return []


def save_local_vector_store(entries: list[dict[str, Any]]) -> None:
    """
    Saves chunks and precomputed embeddings to local disk.
    Requires no running database service or network connections.
    """
    try:
        LOCAL_VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_VECTOR_STORE_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d vector records to local embedded store: %s", len(entries), LOCAL_VECTOR_STORE_PATH)
    except Exception as e:
        logger.error("Failed to write local vector store file: %s", e)


async def store_documents(
    documents: Sequence[Document],
    embeddings: Embeddings | None = None,
    collection_name: str = "rag_documents",
) -> list[str]:
    """
    Stores document chunks with dense vector embeddings.
    Tries PostgreSQL first; if unavailable, stores in local embedded vector store.
    """
    if not documents:
        return []

    try:
        embed_model = embeddings or get_embedding_model()
        texts = [d.page_content for d in documents]
        logger.info("Computing dense embeddings for %d chunks using local FastEmbed...", len(texts))
        vectors = embed_model.embed_documents(texts)
    except Exception as e:
        logger.error("Failed to generate document embeddings: %s", e, exc_info=True)
        raise

    doc_ids: list[str] = []
    local_entries: list[dict[str, Any]] = []

    for doc, vec in zip(documents, vectors):
        chunk_id = str(uuid.uuid4())
        doc_ids.append(chunk_id)
        local_entries.append({
            "id": chunk_id,
            "document": doc.page_content,
            "cmetadata": doc.metadata,
            "embedding": vec,
        })

    # Always persist locally as embedded fallback (zero service required)
    save_local_vector_store(local_entries)

    # Try storing to PostgreSQL if database service is accessible
    collection_id = uuid.UUID("3a896d38-6cd0-4856-834b-12e07cff388e")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("""
                INSERT INTO langchain_pg_collection (uuid, name) VALUES (:uuid, :name)
                ON CONFLICT (uuid) DO NOTHING
                """),
                {"uuid": collection_id, "name": collection_name},
            )

            for entry in local_entries:
                meta_json = json.dumps(entry["cmetadata"])
                await conn.execute(
                    text("""
                    INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                    VALUES (:id, :col_id, CAST(:vec AS float8[]), :doc, CAST(:meta AS jsonb))
                    ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding, document = excluded.document, cmetadata = excluded.cmetadata
                    """),
                    {
                        "id": entry["id"],
                        "col_id": collection_id,
                        "vec": entry["embedding"],
                        "doc": entry["document"],
                        "meta": meta_json,
                    },
                )
        logger.info("Synchronized %d chunks to PostgreSQL pgvector.", len(doc_ids))
    except Exception as e:
        logger.warning(
            "PostgreSQL vector service is not reachable (%s). Falling back to local embedded vector store.",
            e,
        )

    return doc_ids
