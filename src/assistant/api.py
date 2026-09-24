"""FastAPI endpoint for the unified assistant workflow."""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.errors import AgentError
from src.assistant.schema import AssistantRequest, AssistantResponse
from src.assistant.service import UnifiedAssistantService
from src.core.config import settings as agent_settings
from src.database.connection import get_db

logger = logging.getLogger("src.assistant.api")
router = APIRouter(prefix="/assistant", tags=["assistant"])


@lru_cache(maxsize=1)
def get_assistant_service() -> UnifiedAssistantService:
    """Create the unified service without constructing an LLM until needed."""
    return UnifiedAssistantService(agent_settings)


@router.post(
    "/chat",
    response_model=AssistantResponse,
    status_code=status.HTTP_200_OK,
    summary="Route a request to policy RAG, the project agent, or direct LLM",
)
async def assistant_chat(
    request: AssistantRequest,
    db: AsyncSession = Depends(get_db),
    service: UnifiedAssistantService = Depends(get_assistant_service),
) -> AssistantResponse:
    try:
        return await service.chat(request, db)
    except AgentError as exc:
        logger.error("Unified assistant agent error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unified assistant request failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the assistant request.",
        ) from exc
