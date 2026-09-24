"""Unified orchestration between the existing RAG service and agent service."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.agents.schema import AgentRequest
from src.agents.service import AgentService, ConversationStore, create_agent_service
from src.assistant.router import classify_request
from src.assistant.schema import AssistantRequest, AssistantResponse
from src.core.config import AgentSettings
from src.rag.schema import ChatRequest

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("src.assistant.service")


class UnifiedAssistantService:
    """Routes a single conversation to policy RAG, the project agent, or direct LLM."""

    def __init__(
        self,
        agent_settings: AgentSettings,
        store: ConversationStore | None = None,
        agent_service: AgentService | None = None,
        rag_service_factory: Callable[[AsyncSession], Any] | None = None,
    ) -> None:
        self.agent_settings = agent_settings
        self.store = store or ConversationStore(max_messages=agent_settings.memory_max_messages)
        self._agent_service = agent_service
        self.rag_service_factory = rag_service_factory

    def _get_agent_service(self) -> AgentService:
        if self._agent_service is None:
            self._agent_service = create_agent_service(self.agent_settings, store=self.store)
        return self._agent_service

    async def chat(
        self,
        request: AssistantRequest,
        db: AsyncSession,
    ) -> AssistantResponse:
        session_id = request.session_id or str(uuid.uuid4())
        session = self.store.get(session_id)
        decision = classify_request(
            request.message,
            current_project_id=session.current_project_id,
            explicit_project_id=request.project_id,
        )
        logger.info(
            "Unified assistant request session_id=%s route=%s message=%s",
            session_id,
            decision.route,
            request.message,
        )

        if decision.route == "clarification":
            answer = "Which project ID should I use for that request?"
            self.store.append(session_id, request.message, answer, session.current_project_id)
            return AssistantResponse(
                answer=answer,
                route="clarification",
                session_id=session_id,
            )

        if decision.route == "policy":
            if self.rag_service_factory is None:
                # Lazy import keeps agent-only tests independent of the existing
                # RAG provider imports while preserving the existing service.
                from src.rag.service import ChatService

                self.rag_service_factory = ChatService
            response = await self.rag_service_factory(db).chat(
                ChatRequest(
                    message=request.message,
                    session_id=session_id,
                    history=session.messages,
                )
            )
            self.store.append(session_id, request.message, response.answer, session.current_project_id)
            return AssistantResponse(
                answer=response.answer,
                route="policy",
                session_id=session_id,
                sources=response.sources,
                show_pdf=response.show_pdf,
            )

        agent_response = await self._get_agent_service().chat(
            AgentRequest(
                message=request.message,
                session_id=session_id,
                project_id=request.project_id or decision.project_id,
                history=session.messages,
                preferred_route="project_api" if decision.route == "project" else "direct",
            )
        )
        return AssistantResponse(
            answer=agent_response.answer,
            route="project" if agent_response.route == "project_api" else "direct",
            session_id=agent_response.session_id,
            selected_tool=agent_response.selected_tool,
            operation=agent_response.operation,
            project_id=agent_response.project_id,
            tool_error=agent_response.tool_error,
        )
