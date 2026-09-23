"""
PostgreSQL pgvector storage module.
Stores LangChain Document chunks with dense semantic embeddings in langchain_pg_embedding.
"""
import json
import logging
import uuid
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.rag.embeddings import get_embedding_model

logger = logging.getLogger("rag.vectorstore")


def get_vectorstore(
    embeddings: Embeddings | None = None,
    collection_name: str = "rag_documents",
) -> PGVector:
    """
    Returns an async PGVector vectorstore instance connected to PostgreSQL.
    """
    embed_model = embeddings or get_embedding_model()
    return PGVector(
        embeddings=embed_model,
        collection_name=collection_name,
        connection=settings.database_url,
        use_jsonb=True,
        async_mode=True,
        create_extension=False,
    )


async def store_documents(
    documents: Sequence[Document],
    embeddings: Embeddings | None = None,
    collection_name: str = "rag_documents",
) -> list[str]:
    """
    Stores a batch of LangChain Document chunks into PostgreSQL with dense semantic vectors.
    """
    if not documents:
        return []

    try:
        embed_model = embeddings or get_embedding_model()
        texts = [d.page_content for d in documents]
        logger.info("Computing dense semantic embeddings for %d chunks...", len(texts))
        vectors = embed_model.embed_documents(texts)
    except Exception as e:
        logger.error("Failed to generate document embeddings during storage: %s", e, exc_info=True)
        raise

    collection_id = uuid.UUID("3a896d38-6cd0-4856-834b-12e07cff388e")
    doc_ids = []

    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("""
                INSERT INTO langchain_pg_collection (uuid, name) VALUES (:uuid, :name)
                ON CONFLICT (uuid) DO NOTHING
                """),
                {"uuid": collection_id, "name": collection_name},
            )

            for doc, vec in zip(documents, vectors):
                chunk_id = str(uuid.uuid4())
                doc_ids.append(chunk_id)
                meta_json = json.dumps(doc.metadata)
                await conn.execute(
                    text("""
                    INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                    VALUES (:id, :col_id, CAST(:vec AS float8[]), :doc, CAST(:meta AS jsonb))
                    ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding, document = excluded.document, cmetadata = excluded.cmetadata
                    """),
                    {
                        "id": chunk_id,
                        "col_id": collection_id,
                        "vec": vec,
                        "doc": doc.page_content,
                        "meta": meta_json,
                    },
                )

        logger.info(
            "Stored %d chunks with %d-dimensional dense vectors in PostgreSQL.",
            len(doc_ids),
            len(vectors[0]) if vectors else 0,
        )
        return doc_ids
    except Exception as e:
        logger.error("Database error while storing document chunks: %s", e, exc_info=True)
        raise
