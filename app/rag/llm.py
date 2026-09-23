import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("rag.llm")

# Dedicated xAI (Grok) Configuration
XAI_DEFAULT_API_KEY = ""
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_DEFAULT_MODEL = "grok-2-latest"


def get_llm(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    allow_fake: bool = False,
    fake_response: str | None = None,
) -> BaseChatModel | None:
    """
    Returns an xAI (Grok) Chat model instance configured with zero-temperature
    for strict anti-hallucination factual grounding.
    """
    key = api_key or settings.llm_api_key or XAI_DEFAULT_API_KEY
    if not key:
        return None

    b_url = base_url or settings.llm_base_url or XAI_BASE_URL
    model_name = model
    if not model_name:
        if settings.llm_model and "grok" in settings.llm_model.lower():
            model_name = settings.llm_model
        else:
            model_name = XAI_DEFAULT_MODEL

    # Ensure xAI base url when an xAI key is provided
    if key.startswith("xai-") and not base_url and not settings.llm_base_url:
        b_url = XAI_BASE_URL

    logger.info("Initializing xAI Grok LLM (model=%s, base_url=%s, temperature=%.1f)", model_name, b_url, temperature)
    return ChatOpenAI(
        api_key=key,
        base_url=b_url,
        model=model_name,
        temperature=temperature,
        timeout=15.0,
    )
