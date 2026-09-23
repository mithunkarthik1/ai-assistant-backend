import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("rag.llm")


def get_llm(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    allow_fake: bool = False,
    fake_response: str | None = None,
) -> BaseChatModel | None:
    """
    Returns an OpenAI-compatible Chat model instance (Groq, OpenAI, xAI Grok, Ollama, OpenRouter).
    Configured with zero-temperature for strict anti-hallucination factual grounding.
    """
    key = api_key or settings.llm_api_key or ""
    b_url = base_url or settings.llm_base_url
    model_name = model or settings.llm_model

    # 1. Groq (Fast, free-tier available)
    if key.startswith("gsk_") or (b_url and "groq.com" in b_url):
        b_url = b_url or "https://api.groq.com/openai/v1"
        if not model_name or "llama-3.3-70b-versatile" in model_name:
            model_name = "openai/gpt-oss-120b"
        logger.info("Initializing Groq LLM (model=%s, base_url=%s)", model_name, b_url)

    # 2. xAI (Grok)
    elif key.startswith("xai-") or (b_url and "x.ai" in b_url):
        b_url = b_url or "https://api.x.ai/v1"
        model_name = model_name if (model_name and "grok" in model_name.lower()) else "grok-2-latest"
        logger.info("Initializing xAI Grok LLM (model=%s, base_url=%s)", model_name, b_url)

    # 3. OpenRouter
    elif key.startswith("sk-or-") or (b_url and "openrouter.ai" in b_url):
        b_url = b_url or "https://openrouter.ai/api/v1"
        model_name = model_name or "meta-llama/llama-3.3-70b-instruct"
        logger.info("Initializing OpenRouter LLM (model=%s, base_url=%s)", model_name, b_url)

    # 4. Local Ollama (no key required)
    elif (b_url and "11434" in b_url) or (not key and b_url):
        b_url = b_url or "http://localhost:11434/v1"
        key = key or "ollama"
        model_name = model_name or "llama3"
        logger.info("Initializing Ollama local LLM (model=%s, base_url=%s)", model_name, b_url)

    # 5. Standard OpenAI
    elif key.startswith("sk-"):
        b_url = b_url or "https://api.openai.com/v1"
        model_name = model_name or "gpt-4o-mini"
        logger.info("Initializing OpenAI LLM (model=%s, base_url=%s)", model_name, b_url)

    # 6. Generic custom base_url
    elif b_url and key:
        model_name = model_name or "gpt-4o-mini"
        logger.info("Initializing Custom LLM (model=%s, base_url=%s)", model_name, b_url)

    elif not key:
        logger.warning("No LLM API key configured in settings (LLM_API_KEY).")
        return None

    return ChatOpenAI(
        api_key=key,
        base_url=b_url,
        model=model_name,
        temperature=temperature,
        timeout=30.0,
    )
