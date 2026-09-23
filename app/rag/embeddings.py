"""
Embeddings factory module for dense semantic vector representations.
Supports local CPU-optimized FastEmbed (BAAI/bge-small-en-v1.5) and OpenAIEmbeddings.
"""
import logging
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger("rag.embeddings")

_cached_fastembed = None


class FastEmbedEmbeddings(Embeddings):
    """
    Local dense semantic vector embeddings powered by FastEmbed (ONNX Runtime).
    Generates 384-dimensional dense semantic embeddings locally on CPU with zero external API keys.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        global _cached_fastembed
        if _cached_fastembed is None:
            try:
                from fastembed import TextEmbedding
                logger.info("Initializing FastEmbed semantic model: %s", model_name)
                _cached_fastembed = TextEmbedding(model_name=model_name)
            except ImportError as e:
                logger.error("FastEmbed library not available: %s", e)
                raise
        self.model = _cached_fastembed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = list(self.model.embed(texts))
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error("Error generating document embeddings: %s", e)
            raise

    def embed_query(self, text: str) -> List[float]:
        try:
            embeddings = list(self.model.embed([text]))
            return embeddings[0].tolist()
        except Exception as e:
            logger.error("Error generating query embedding for text: '%s': %s", text, e)
            raise


def get_embedding_model(
    api_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Embeddings:
    """
    Returns the configured semantic embedding model.
    Defaults to local FastEmbed (BAAI/bge-small-en-v1.5) for true dense vector semantic search,
    or OpenAIEmbeddings if provider is set to 'openai' with a valid key.
    """
    prov = (provider or settings.embedding_provider).lower()
    key = api_key or settings.llm_api_key

    if prov == "openai" and key:
        logger.info("Using OpenAIEmbeddings (model=%s)", model or settings.embedding_model or "text-embedding-3-small")
        return OpenAIEmbeddings(
            api_key=key,
            model=model or settings.embedding_model or "text-embedding-3-small",
        )

    # Local FastEmbed for 100% free dense semantic search
    model_name = model or settings.embedding_model or "BAAI/bge-small-en-v1.5"
    if "text-embedding" in model_name:
        model_name = "BAAI/bge-small-en-v1.5"

    return FastEmbedEmbeddings(model_name=model_name)
