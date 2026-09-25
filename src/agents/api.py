"""FastAPI routes for the isolated agentic assistant."""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from src.agents.errors import AgentConfigurationError
from src.agents.schema import AgentRequest, AgentResponse
from src.agents.service import AgentService, create_agent_service
from src.core.config import settings

logger = logging.getLogger("src.agents.api")
router = APIRouter(prefix="/agents", tags=["agents"])


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    """Create one process-local agent service and conversation store."""
    return create_agent_service(settings)


@router.post(
    "/chat",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the agent to answer directly or use the Project API",
)
async def agent_chat(
    request: AgentRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    try:
        return await service.chat(request)
    except AgentConfigurationError as exc:
        logger.error("Agent configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Agent request failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the agent request.",
        ) from exc
