"""
Qdrant Cloud vector database client and service.
Provides reusable connection pooling, idempotent collection management,
point upserts with stable deterministic IDs, payload filtering, and health monitoring.
"""

import logging
import uuid
from typing import Any, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from src.core.config import settings

logger = logging.getLogger("src.database.qdrant")

NAMESPACE_CHUNK_ID = uuid.UUID("a74f4b46-7788-4e89-b88a-2c6396f8a845")


def chunk_id_to_qdrant_id(chunk_id: str) -> str:
    """
    Deterministically transforms a stable string chunk_id into a valid RFC 4122 UUID
    compatible with Qdrant point IDs. Guarantees 100% idempotency across re-indexes.
    """
    try:
        # If it's already a valid UUID string, use it directly
        val = uuid.UUID(chunk_id)
        return str(val)
    except (ValueError, AttributeError):
        return str(uuid.uuid5(NAMESPACE_CHUNK_ID, str(chunk_id)))


class QdrantService:
    """
    Reusable, production-grade Qdrant client wrapper.
    Manages client connection, collection lifecycles, vector upserting,
    similarity search, point deletion, and cluster health checks.
    """

    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._collection_name: str = settings.qdrant_collection or "document_chunks"

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def is_configured(self) -> bool:
        """Returns True if QDRANT_URL is provided in configuration."""
        return bool(settings.qdrant_url and settings.qdrant_url.strip())

    def connect(self) -> QdrantClient | None:
        """
        Lazily creates or returns the singleton QdrantClient connection.
        Returns None if Qdrant is not configured or unreachable.
        """
        if not self.is_configured():
            return None

        if self._client is not None:
            return self._client

        try:
            url = settings.qdrant_url.strip()
            api_key = settings.qdrant_api_key.strip() if settings.qdrant_api_key else None
            # Mask host in logs for security
            masked_host = url.split("@")[-1] if "@" in url else url
            logger.info("Initializing Qdrant client for host: %s", masked_host)

            self._client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=20.0,
                check_compatibility=False,
            )
            return self._client
        except Exception as e:
            logger.error("Failed to initialize QdrantClient: %s", e)
            self._client = None
            return None

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Checks whether the collection exists in Qdrant."""
        client = self.connect()
        if client is None:
            return False

        col_name = collection_name or self._collection_name
        try:
            return bool(client.collection_exists(collection_name=col_name))
        except Exception as e:
            logger.warning("Error checking collection '%s' existence: %s", col_name, e)
            return False

    def create_collection(
        self,
        dimension: int = 384,
        distance: Distance = Distance.COSINE,
        collection_name: str | None = None,
    ) -> bool:
        """
        Idempotently creates a Qdrant collection with the given vector dimensions and distance metric.
        If the collection already exists, does nothing and returns True.
        """
        client = self.connect()
        if client is None:
            logger.warning("Qdrant not connected; skipping collection creation.")
            return False

        col_name = collection_name or self._collection_name
        try:
            if self.collection_exists(col_name):
                logger.info("Qdrant collection '%s' already exists.", col_name)
                return True

            logger.info(
                "Creating Qdrant collection '%s' (dimension=%d, distance=%s)...",
                col_name,
                dimension,
                distance.name,
            )
            client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=dimension, distance=distance),
            )
            try:
                from qdrant_client.models import PayloadSchemaType
                client.create_payload_index(
                    collection_name=col_name,
                    field_name="document_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.warning("Payload index creation on document_id skipped: %s", e)
            logger.info("Qdrant collection '%s' created successfully.", col_name)
            return True
        except UnexpectedResponse as e:
            if "already exists" in str(e).lower():
                try:
                    from qdrant_client.models import PayloadSchemaType
                    client.create_payload_index(
                        collection_name=col_name,
                        field_name="document_id",
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
                logger.info("Qdrant collection '%s' already exists (concurrency safe).", col_name)
                return True
            logger.error("Unexpected Qdrant response while creating collection: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to create Qdrant collection '%s': %s", col_name, e)
            return False

    def upsert_points(
        self,
        points: Sequence[PointStruct],
        collection_name: str | None = None,
    ) -> bool:
        """
        Upserts vector points with payloads into Qdrant.
        """
        if not points:
            return True

        client = self.connect()
        if client is None:
            logger.warning("Qdrant client not available; cannot upsert points.")
            return False

        col_name = collection_name or self._collection_name
        try:
            client.upsert(
                collection_name=col_name,
                points=list(points),
                wait=True,
            )
            logger.info("Successfully upserted %d points to Qdrant collection '%s'.", len(points), col_name)
            return True
        except Exception as e:
            logger.error("Failed to upsert points into Qdrant collection '%s': %s", col_name, e)
            raise

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        collection_name: str | None = None,
    ) -> list[ScoredPoint]:
        """
        Executes vector similarity search on Qdrant with optional payload filters.
        """
        client = self.connect()
        if client is None:
            logger.warning("Qdrant client not connected for search.")
            return []

        col_name = collection_name or self._collection_name

        query_filter: Filter | None = None
        if filters:
            conditions = []
            for k, v in filters.items():
                if v is not None:
                    conditions.append(
                        FieldCondition(
                            key=k,
                            match=MatchValue(value=str(v)),
                        )
                    )
            if conditions:
                query_filter = Filter(must=conditions)

        try:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=col_name,
                    query=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                return response.points
            elif hasattr(client, "search"):
                results = client.search(
                    collection_name=col_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                return results
            return []
        except Exception as e:
            logger.error("Qdrant vector search failed in collection '%s': %s", col_name, e)
            return []

    def delete_points(
        self,
        chunk_ids: Sequence[str],
        collection_name: str | None = None,
    ) -> bool:
        """
        Deletes vector points from Qdrant given a sequence of chunk_ids or point UUIDs.
        """
        if not chunk_ids:
            return True

        client = self.connect()
        if client is None:
            return False

        col_name = collection_name or self._collection_name
        qdrant_ids = [chunk_id_to_qdrant_id(cid) for cid in chunk_ids]

        try:
            client.delete(
                collection_name=col_name,
                points_selector=PointIdsList(points=qdrant_ids),
                wait=True,
            )
            logger.info("Deleted %d points from Qdrant collection '%s'.", len(qdrant_ids), col_name)
            return True
        except Exception as e:
            logger.error("Failed to delete points from Qdrant: %s", e)
            return False

    def delete_by_document(
        self,
        document_id: str | uuid.UUID,
        collection_name: str | None = None,
    ) -> bool:
        """
        Deletes all vector points associated with a specific document_id from Qdrant.
        """
        client = self.connect()
        if client is None:
            return False

        col_name = collection_name or self._collection_name
        doc_str = str(document_id)

        try:
            client.delete(
                collection_name=col_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=doc_str),
                        )
                    ]
                ),
                wait=True,
            )
            logger.info("Deleted all points for document '%s' from Qdrant collection '%s'.", doc_str, col_name)
            return True
        except Exception as e:
            logger.error("Failed to delete document '%s' points from Qdrant: %s", doc_str, e)
            return False

    def delete_by_filename(
        self,
        file_name: str,
        collection_name: str | None = None,
    ) -> bool:
        """
        Deletes all vector points associated with a specific file_name from Qdrant.
        """
        client = self.connect()
        if client is None:
            return False

        col_name = collection_name or self._collection_name
        try:
            client.delete(
                collection_name=col_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_name",
                            match=MatchValue(value=file_name),
                        )
                    ]
                ),
                wait=True,
            )
            logger.info("Deleted all points for file '%s' from Qdrant collection '%s'.", file_name, col_name)
            return True
        except Exception as e:
            logger.error("Failed to delete file '%s' points from Qdrant: %s", file_name, e)
            return False

    def health_check(self) -> dict[str, Any]:
        """
        Performs a ping and collection verification to verify Qdrant cluster connectivity.
        """
        if not self.is_configured():
            return {
                "status": "not_configured",
                "message": "Qdrant URL not configured.",
            }

        client = self.connect()
        if client is None:
            return {
                "status": "unhealthy",
                "message": "Failed to connect to Qdrant cluster.",
            }

        try:
            exists = self.collection_exists(self._collection_name)
            points_count = 0
            if exists:
                col_info = client.get_collection(self._collection_name)
                points_count = col_info.points_count or 0

            return {
                "status": "healthy",
                "collection": self._collection_name,
                "collection_exists": exists,
                "points_count": points_count,
            }
        except Exception as e:
            logger.warning("Qdrant health check warning: %s", e)
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Singleton instance for the application
qdrant_service = QdrantService()
