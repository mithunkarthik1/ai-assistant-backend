"""Tests for centralized routing between policy RAG and the project agent."""

from types import SimpleNamespace

import pytest

from src.agents.schema import AgentRequest, AgentResponse
from src.agents.service import ConversationStore
from src.assistant.router import classify_request
from src.assistant.schema import AssistantRequest
from src.assistant.service import UnifiedAssistantService
from src.core.config import AgentSettings
from src.rag.schema import ChatResponse


class FakeRagService:
    def __init__(self) -> None:
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        return ChatResponse(
            answer="Policy answer",
            sources=[],
            session_id=request.session_id,
            show_pdf=False,
        )


class FakeAgentService:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def chat(self, request):
        self.requests.append(request)
        is_project = request.preferred_route == "project_api"
        return AgentResponse(
            answer="Project answer" if is_project else "Direct answer",
            session_id=request.session_id or "generated",
            route="project_api" if is_project else "direct",
            selected_tool="project_api" if is_project else None,
            operation="status" if is_project else None,
            project_id=request.project_id or ("PROJ-123" if is_project else None),
        )


def test_router_classifies_policy_project_and_direct_requests():
    assert classify_request("How many PTO days do I get?").route == "policy"
    assert classify_request("What is the status of PROJ-123?").route == "project"
    assert classify_request("Explain REST APIs.").route == "direct"


def test_router_requests_project_id_when_project_context_is_missing():
    decision = classify_request("What is the current project status?")

    assert decision.route == "clarification"


@pytest.mark.asyncio
async def test_unified_service_delegates_policy_requests_to_existing_rag():
    rag = FakeRagService()
    agent = FakeAgentService()
    service = UnifiedAssistantService(
        AgentSettings(),
        store=ConversationStore(),
        agent_service=agent,
        rag_service_factory=lambda db: rag,
    )

    response = await service.chat(
        AssistantRequest(message="How many PTO days do I get?", session_id="policy-session"),
        SimpleNamespace(),
    )

    assert response.route == "policy"
    assert response.answer == "Policy answer"
    assert len(rag.requests) == 1
    assert agent.requests == []


@pytest.mark.asyncio
async def test_unified_service_delegates_project_requests_to_agent():
    rag = FakeRagService()
    agent = FakeAgentService()
    service = UnifiedAssistantService(
        AgentSettings(),
        store=ConversationStore(),
        agent_service=agent,
        rag_service_factory=lambda db: rag,
    )

    response = await service.chat(
        AssistantRequest(
            message="What is the status of PROJ-123?",
            session_id="project-session",
            project_id="PROJ-123",
        ),
        SimpleNamespace(),
    )

    assert response.route == "project"
    assert response.selected_tool == "project_api"
    assert agent.requests[0].preferred_route == "project_api"
    assert agent.requests[0].project_id == "PROJ-123"
    assert rag.requests == []
