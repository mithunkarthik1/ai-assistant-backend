"""Focused tests for the isolated agentic assistant POC."""

import httpx
import pytest

from src.agents.errors import InvalidProjectIdError, ProjectApiError, ProjectApiTimeoutError
from src.agents.schema import AgentDecision, AgentRequest
from src.agents.service import (
    AgentService,
    ConversationStore,
    ProjectAgentGraph,
    ProjectApiClient,
)


class FakeLLM:
    def __init__(self, decisions: list[AgentDecision], answer: str = "LLM answer") -> None:
        self.decisions = list(decisions)
        self.answer_text = answer
        self.decide_calls: list[dict] = []
        self.answer_calls: list[dict] = []

    async def decide(self, request, conversation, current_project_id=None):
        self.decide_calls.append({
            "request": request,
            "conversation": conversation,
            "current_project_id": current_project_id,
        })
        return self.decisions.pop(0)

    async def answer(self, request, conversation, decision, tool_result=None, tool_error=None):
        self.answer_calls.append({
            "request": request,
            "conversation": conversation,
            "decision": decision,
            "tool_result": tool_result,
            "tool_error": tool_error,
        })
        return self.answer_text


class FakeProjectApi:
    def __init__(self, result=None, error=None) -> None:
        self.result = result or {"id": "PROJ-123", "status": "active"}
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def execute(self, operation, project_id):
        self.calls.append((operation, project_id))
        if self.error:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_direct_llm_response_does_not_call_project_api():
    llm = FakeLLM([AgentDecision(route="direct")], answer="A direct answer.")
    project_api = FakeProjectApi()
    service = AgentService(ProjectAgentGraph(llm, project_api), ConversationStore())

    response = await service.chat(AgentRequest(message="What is an API?", session_id="direct"))

    assert response.answer == "A direct answer."
    assert response.route == "direct"
    assert response.selected_tool is None
    assert project_api.calls == []


@pytest.mark.asyncio
async def test_project_api_tool_selection():
    llm = FakeLLM([
        AgentDecision(route="project_api", operation="status", project_id="PROJ-123")
    ])
    project_api = FakeProjectApi()
    service = AgentService(ProjectAgentGraph(llm, project_api), ConversationStore())

    response = await service.chat(AgentRequest(message="What is project PROJ-123 status?"))

    assert response.selected_tool == "project_api"
    assert response.operation == "status"
    assert project_api.calls == [("status", "PROJ-123")]


@pytest.mark.asyncio
async def test_successful_project_api_call_is_passed_to_final_response():
    llm = FakeLLM([
        AgentDecision(route="project_api", operation="owner", project_id="PROJ-123")
    ], answer="The owner is Alex.")
    project_api = FakeProjectApi(result={"id": "PROJ-123", "owner": "Alex"})
    service = AgentService(ProjectAgentGraph(llm, project_api), ConversationStore())

    response = await service.chat(AgentRequest(message="Who owns PROJ-123?"))

    assert response.answer == "The owner is Alex."
    assert llm.answer_calls[0]["tool_result"] == {"id": "PROJ-123", "owner": "Alex"}


@pytest.mark.asyncio
async def test_project_api_failure_is_reported_to_final_response():
    llm = FakeLLM([
        AgentDecision(route="project_api", operation="milestones", project_id="PROJ-123")
    ], answer="I could not retrieve the milestones.")
    project_api = FakeProjectApi(error=ProjectApiError("Project API unavailable."))
    service = AgentService(ProjectAgentGraph(llm, project_api), ConversationStore())

    response = await service.chat(AgentRequest(message="Show PROJ-123 milestones."))

    assert response.tool_error == "Project API unavailable."
    assert llm.answer_calls[0]["tool_error"] == "Project API unavailable."


@pytest.mark.asyncio
async def test_follow_up_retains_current_project_context():
    llm = FakeLLM([
        AgentDecision(route="project_api", operation="details", project_id="PROJ-123"),
        AgentDecision(route="project_api", operation="status"),
    ])
    project_api = FakeProjectApi()
    service = AgentService(ProjectAgentGraph(llm, project_api), ConversationStore())

    await service.chat(AgentRequest(message="Tell me about PROJ-123.", session_id="follow-up"))
    response = await service.chat(
        AgentRequest(message="What is its current status?", session_id="follow-up")
    )

    assert llm.decide_calls[1]["current_project_id"] == "PROJ-123"
    assert project_api.calls[-1] == ("status", "PROJ-123")
    assert response.project_id == "PROJ-123"


def test_project_api_rejects_invalid_project_ids():
    client = ProjectApiClient(base_url="http://project-api.test")

    with pytest.raises(InvalidProjectIdError):
        client.build_url("status", "bad/project")


@pytest.mark.asyncio
async def test_project_api_successful_http_call():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "PROJ-123", "status": "active"})

    client = ProjectApiClient(
        base_url="http://project-api.test",
        transport=httpx.MockTransport(handler),
    )

    result = await client.execute("status", "PROJ-123")

    assert result["status"] == "active"


@pytest.mark.asyncio
async def test_project_api_timeout_is_normalized():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = ProjectApiClient(
        base_url="http://project-api.test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProjectApiTimeoutError):
        await client.execute("details", "PROJ-123")
