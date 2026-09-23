"""
Vector similarity retriever module for company policy documents.
Performs dense vector semantic search against PostgreSQL pgvector or local embedded file store.

How vector search works without a database service:
- Query is converted into a 384-dimensional dense vector using local FastEmbed (CPU ONNX runtime).
- If PostgreSQL pgvector service is offline, candidate vectors are loaded from `data/local_vector_store.json`.
- Dot product and cosine similarity are computed in memory using NumPy (`np.dot(q, c) / (||q|| * ||c||)`).
- Chunks above the similarity threshold are ranked and returned to the LLM.
"""
import logging
import uuid
from typing import Any, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from src.config import settings
from src.database import engine
from src.langchain.embeddings import get_embedding_model
from src.langchain.vectorstore import get_local_vector_store

logger = logging.getLogger("src.langchain.retriever")


def contextualize_query(query: str, chat_history: Sequence[Any] | None = None) -> str:
    """
    Contextualizes brief follow-up queries with recent conversation context.
    If the question is brief (e.g. 'what about my spouse?', 'how much?'),
    prepends the previous user turn to maintain semantic continuity.
    """
    if not chat_history:
        return query

    cleaned = query.strip()
    words = cleaned.split()

    is_short = len(words) <= 5
    starts_with_connector = cleaned.lower().startswith(
        ("and ", "also ", "what about", "how about", "what if", "can i also", "does it", "is it")
    )

    if not (is_short or starts_with_connector):
        return query

    last_user_query = None
    for msg in reversed(chat_history):
        role = (
            getattr(msg, "role", None)
            or getattr(msg, "type", None)
            or (msg.get("role") if isinstance(msg, dict) else "")
            or "user"
        )
        content = (
            getattr(msg, "content", "")
            if not isinstance(msg, dict)
            else msg.get("content", "")
        )
        if role in ("user", "human") and content.strip():
            last_user_query = content.strip()
            break

    if last_user_query and last_user_query.lower() != cleaned.lower():
        combined = f"{last_user_query} - {cleaned}"
        logger.debug("Contextualized follow-up query: '%s' -> '%s'", cleaned, combined)
        return combined

    return query


async def retrieve_semantic_chunks(
    document_id: str | uuid.UUID,
    query: str,
    top_k: int = 6,
    embeddings: Embeddings | None = None,
    min_similarity: float | None = None,
    chat_history: Sequence[Any] | None = None,
) -> list[Document]:
    """
    Performs dense vector semantic search across chunk embeddings.
    Queries PostgreSQL pgvector when available; otherwise queries the local embedded store.
    """
    threshold = min_similarity if min_similarity is not None else settings.min_similarity
    search_query = contextualize_query(query, chat_history)

    try:
        embed_model = embeddings or get_embedding_model()
        q_vec = embed_model.embed_query(search_query)
        q_arr = np.array(q_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm == 0:
            q_norm = 1e-9
    except Exception as e:
        logger.error("Failed to generate query embedding: %s", e)
        return []

    # 1. Attempt retrieval from PostgreSQL pgvector
    rows = []
    try:
        async with engine.connect() as conn:
            res = await conn.execute(
                text(
                    "SELECT id, cmetadata, document, embedding "
                    "FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'document_id' = :doc_id"
                ),
                {"doc_id": str(document_id)},
            )
            rows = res.fetchall()
    except Exception as e:
        logger.debug("PostgreSQL query skipped (offline or not reachable): %s", e)

    scored_chunks: list[tuple[float, Document]] = []

    # 2. If PostgreSQL returned rows, score them
    if rows:
        for row in rows:
            try:
                cmetadata = row[1] or {}
                content = row[2] or ""
                emb = row[3]
                if emb and len(emb) == len(q_arr):
                    chunk_arr = np.array(emb, dtype=np.float32)
                    c_norm = float(np.linalg.norm(chunk_arr))
                    sim = float(np.dot(q_arr, chunk_arr) / (q_norm * c_norm)) if c_norm > 0 else 0.0
                else:
                    sim = 0.0
                meta = {**cmetadata, "score": round(sim, 4)}
                scored_chunks.append((sim, Document(page_content=content, metadata=meta)))
            except Exception as e:
                logger.warning("Error computing similarity for PostgreSQL chunk: %s", e)
                continue

    # 3. Fallback: If no DB rows were found, use local embedded vector store (no DB service needed!)
    if not scored_chunks:
        local_entries = get_local_vector_store()
        if local_entries:
            logger.info("Using local embedded vector store (%d entries, zero DB service required).", len(local_entries))
            for item in local_entries:
                try:
                    cmetadata = item.get("cmetadata", {})
                    content = item.get("document", "")
                    emb = item.get("embedding", [])
                    if emb and len(emb) == len(q_arr):
                        chunk_arr = np.array(emb, dtype=np.float32)
                        c_norm = float(np.linalg.norm(chunk_arr))
                        sim = float(np.dot(q_arr, chunk_arr) / (q_norm * c_norm)) if c_norm > 0 else 0.0
                    else:
                        sim = 0.0
                    meta = {**cmetadata, "score": round(sim, 4)}
                    scored_chunks.append((sim, Document(page_content=content, metadata=meta)))
                except Exception as e:
                    logger.warning("Error computing similarity for local chunk: %s", e)
                    continue

    if not scored_chunks:
        logger.warning("No embeddings available in either PostgreSQL or local embedded store.")
        return []

    # Sort descending by semantic similarity
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    if not scored_chunks or scored_chunks[0][0] < threshold:
        logger.info(
            "Top score (%.4f) below minimum similarity threshold (%.2f) for query: '%s'",
            scored_chunks[0][0] if scored_chunks else 0.0,
            threshold,
            query,
        )
        return []

    top_score = scored_chunks[0][0]
    effective_threshold = max(threshold, top_score - 0.12)
    relevant = [doc for score, doc in scored_chunks[:top_k] if score >= effective_threshold]

    logger.info(
        "Semantic Search for '%s': retrieved %d chunks (top_score=%.4f, threshold=%.4f)",
        search_query,
        len(relevant),
        top_score,
        effective_threshold,
    )
    return relevant


async def retrieve_relevant_chunks(
    document_id: str | uuid.UUID,
    query: str,
    top_k: int | None = None,
    embeddings: Embeddings | None = None,
    min_similarity: float | None = None,
    chat_history: Sequence[Any] | None = None,
) -> list[Document]:
    """
    Convenience wrapper to retrieve the top semantically relevant policy chunks.
    """
    k = top_k if top_k is not None else settings.top_k
    threshold = min_similarity if min_similarity is not None else settings.min_similarity

    return await retrieve_semantic_chunks(
        document_id=document_id,
        query=query,
        top_k=k,
        embeddings=embeddings,
        min_similarity=threshold,
        chat_history=chat_history,
    )
