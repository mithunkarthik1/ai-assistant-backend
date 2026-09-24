"""Central agent service for planning, tools, workflow, and conversation state."""

import json
import logging
import re
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict
from urllib.parse import quote

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agents.errors import (
    AgentConfigurationError,
    AgentResponseError,
    InvalidProjectIdError,
    MissingProjectIdError,
    ProjectApiError,
    ProjectApiTimeoutError,
    ProjectDataMissingError,
    ProjectNotFoundError,
)
from src.agents.schema import AgentDecision, AgentRequest, AgentResponse, ProjectOperation
from src.core.config import AgentSettings, settings


logger = logging.getLogger("src.agents.service")
_PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class AgentLLM:
    """Configurable OpenAI-compatible LLM adapter for planning and final answers."""

    def __init__(self, agent_settings: AgentSettings) -> None:
        if not agent_settings.llm_api_key and not agent_settings.llm_base_url:
            raise AgentConfigurationError(
                "Configure AGENT_LLM_API_KEY or AGENT_LLM_BASE_URL before using the agent."
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise AgentConfigurationError(
                "Install langchain-openai to use the configured OpenAI-compatible LLM."
            ) from exc

        self.model = ChatOpenAI(
            api_key=agent_settings.llm_api_key or "local",
            base_url=agent_settings.llm_base_url,
            model=agent_settings.llm_model,
            temperature=0.0,
            timeout=30.0,
        )

    async def decide(
        self,
        request: str,
        conversation: Sequence[dict[str, str]],
        current_project_id: str | None = None,
    ) -> AgentDecision:
        """Ask the LLM to choose direct answering or a Project API operation."""
        prompt = (
            "Return JSON only with this shape: "
            '{"route":"direct|project_api","operation":"details|status|owner|milestones|null",'
            '"project_id":"string|null","rationale":"string"}.\n'
            "Use route=project_api when the user asks for project-specific facts. "
            "Use the conversation and current project context to resolve follow-ups such as "
            '"What is its current status?" Use route=direct for general questions.\n\n'
            f"Current project context: {current_project_id or 'none'}\n"
            f"Conversation: {json.dumps(list(conversation), ensure_ascii=False)}\n"
            f"User request: {request}"
        )

        raw = await self._invoke(
            "You are the planning step of a small project assistant. Do not answer the user; classify the request.",
            prompt,
        )
        try:
            decision = AgentDecision.model_validate_json(self._extract_json(raw))
        except Exception:
            logger.warning("LLM planning response was not valid JSON; using deterministic fallback.")
            decision = self._fallback_decision(request, current_project_id)

        if decision.route == "project_api" and not decision.project_id:
            decision = decision.model_copy(update={"project_id": current_project_id})
        return decision

    async def answer(
        self,
        request: str,
        conversation: Sequence[dict[str, str]],
        decision: AgentDecision,
        tool_result: dict[str, Any] | None = None,
        tool_error: str | None = None,
    ) -> str:
        """Generate the final response after direct reasoning or tool execution."""
        context = {
            "conversation": list(conversation),
            "decision": decision.model_dump(),
            "project_api_result": tool_result,
            "project_api_error": tool_error,
        }
        prompt = (
            "Answer the user's request clearly and concisely. Use only the Project API result for "
            "project-specific facts. If the Project API failed or data is missing, explain that "
            "without inventing project details. If this is a general request, answer directly.\n\n"
            f"Agent context: {json.dumps(context, ensure_ascii=False, default=str)}\n"
            f"User request: {request}"
        )
        answer = await self._invoke(
            "You are WorkPilot's general-purpose project assistant.",
            prompt,
        )
        if not answer.strip():
            raise AgentResponseError("The LLM returned an empty response.")
        return answer.strip()

    async def _invoke(self, system_prompt: str, human_prompt: str) -> str:
        try:
            response = await self.model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ])
        except Exception as exc:
            logger.error("Agent LLM request failed: %s", exc, exc_info=True)
            raise AgentResponseError("The configured LLM could not process the request.") from exc
        return self._content(response)

    @staticmethod
    def _content(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _extract_json(raw: str) -> str:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in planning response.")
        return cleaned[start : end + 1]

    @staticmethod
    def _fallback_decision(request: str, current_project_id: str | None) -> AgentDecision:
        lower = request.lower()
        operation: ProjectOperation | None = None
        if any(word in lower for word in ("status", "state", "progress")):
            operation = "status"
        elif "owner" in lower or "responsible" in lower:
            operation = "owner"
        elif any(word in lower for word in ("milestone", "milestones", "timeline")):
            operation = "milestones"
        elif "project" in lower and any(word in lower for word in ("detail", "information", "about")):
            operation = "details"

        if operation:
            return AgentDecision(
                route="project_api",
                operation=operation,
                project_id=current_project_id,
                rationale="Deterministic fallback detected a project lookup intent.",
            )
        return AgentDecision(
            route="direct",
            rationale="Deterministic fallback selected a direct answer.",
        )


def validate_project_id(project_id: str | None) -> str:
    """Validate an identifier before placing it into an API path."""
    if not project_id:
        raise MissingProjectIdError("A project ID is required for this operation.")
    if not _PROJECT_ID_PATTERN.fullmatch(project_id):
        raise InvalidProjectIdError(f"Invalid project ID: {project_id!r}.")
    return project_id


class ProjectApiClient:
    """HTTP client for the Project API tool."""

    def __init__(
        self,
        base_url: str | None = None,
        path_prefix: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.project_api_base_url).rstrip("/")
        self.path_prefix = (path_prefix or settings.project_api_path_prefix).strip("/")
        self.timeout_seconds = timeout_seconds or settings.project_api_timeout_seconds
        self.transport = transport

    def build_url(self, operation: ProjectOperation, project_id: str) -> str:
        """Build the endpoint URL for a validated project operation."""
        safe_project_id = quote(validate_project_id(project_id), safe="-_.~")
        path = f"/{self.path_prefix}/{safe_project_id}"
        if operation != "details":
            path += f"/{operation}"
        return f"{self.base_url}{path}"

    async def execute(self, operation: ProjectOperation, project_id: str) -> dict[str, Any]:
        """Execute a Project API operation and normalize common failures."""
        url = self.build_url(operation, project_id)
        logger.info(
            "Project API request operation=%s project_id=%s url=%s",
            operation,
            project_id,
            url,
        )

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.error("Project API timeout operation=%s project_id=%s", operation, project_id)
            raise ProjectApiTimeoutError(
                "The Project API did not respond before the timeout."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("Project API project not found operation=%s project_id=%s", operation, project_id)
                raise ProjectNotFoundError(f"Project {project_id!r} was not found.") from exc
            logger.error(
                "Project API HTTP failure status=%s operation=%s project_id=%s",
                exc.response.status_code,
                operation,
                project_id,
            )
            raise ProjectApiError(
                f"Project API returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "Project API request failure operation=%s project_id=%s error=%s",
                operation,
                project_id,
                exc,
            )
            raise ProjectApiError("The Project API could not be reached.") from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("Project API returned invalid JSON operation=%s project_id=%s", operation, project_id)
            raise ProjectDataMissingError("The Project API returned invalid data.") from exc

        if not isinstance(data, dict) or not data:
            logger.error(
                "Project API returned missing data operation=%s project_id=%s result=%s",
                operation,
                project_id,
                data,
            )
            raise ProjectDataMissingError("The Project API returned no project data.")

        logger.info(
            "Project API result operation=%s project_id=%s result_keys=%s",
            operation,
            project_id,
            sorted(data.keys()),
        )
        return data


class AgentGraphState(TypedDict, total=False):
    request: str
    conversation: list[dict[str, str]]
    current_project_id: str | None
    project_id: str | None
    preferred_route: str | None
    decision: AgentDecision
    tool_result: dict[str, Any] | None
    tool_error: str | None
    final_response: str


class ProjectAgentGraph:
    """LangGraph workflow for decide -> tool -> final response."""

    def __init__(self, llm: AgentLLM, project_api: ProjectApiClient) -> None:
        self.llm = llm
        self.project_api = project_api
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentGraphState)
        workflow.add_node("decide", self._decide)
        workflow.add_node("project_api", self._execute_project_api)
        workflow.add_node("finalize", self._finalize)
        workflow.add_edge(START, "decide")
        workflow.add_conditional_edges(
            "decide",
            self._select_next_step,
            {"project_api": "project_api", "finalize": "finalize"},
        )
        workflow.add_edge("project_api", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    async def _decide(self, state: AgentGraphState) -> dict[str, Any]:
        decision = await self.llm.decide(
            request=state["request"],
            conversation=state.get("conversation", []),
            current_project_id=state.get("current_project_id"),
        )
        project_id = decision.project_id or state.get("project_id") or state.get("current_project_id")
        if project_id and decision.project_id != project_id:
            decision = decision.model_copy(update={"project_id": project_id})

        preferred_route = state.get("preferred_route")
        if preferred_route == "direct":
            decision = decision.model_copy(update={"route": "direct", "operation": None})
        elif preferred_route == "project_api":
            decision = decision.model_copy(update={"route": "project_api"})
        logger.info(
            "Agent decision route=%s operation=%s project_id=%s rationale=%s",
            decision.route,
            decision.operation,
            decision.project_id,
            decision.rationale,
        )
        return {"decision": decision, "project_id": project_id}

    @staticmethod
    def _select_next_step(state: AgentGraphState) -> str:
        return "project_api" if state["decision"].route == "project_api" else "finalize"

    async def _execute_project_api(self, state: AgentGraphState) -> dict[str, Any]:
        decision = state["decision"]
        if not decision.operation:
            error = "The agent selected the Project API but did not select an operation."
            logger.error(error)
            return {"tool_error": error}
        if not decision.project_id:
            error = "A project ID is required for this Project API request."
            logger.error(error)
            return {"tool_error": error}

        try:
            result = await self.project_api.execute(decision.operation, decision.project_id)
            return {"tool_result": result, "current_project_id": decision.project_id}
        except ProjectApiError as exc:
            logger.error(
                "Project API tool failed operation=%s project_id=%s error=%s",
                decision.operation,
                decision.project_id,
                exc,
            )
            return {
                "tool_error": str(exc),
                "current_project_id": decision.project_id,
            }

    async def _finalize(self, state: AgentGraphState) -> dict[str, Any]:
        decision = state["decision"]
        try:
            answer = await self.llm.answer(
                request=state["request"],
                conversation=state.get("conversation", []),
                decision=decision,
                tool_result=state.get("tool_result"),
                tool_error=state.get("tool_error"),
            )
        except Exception as exc:
            logger.error("Agent final response failed: %s", exc, exc_info=True)
            answer = "I couldn't complete that request because the configured AI service failed."
        logger.info("Agent final response generated route=%s", decision.route)
        return {"final_response": answer}

    async def ainvoke(
        self,
        request: str,
        conversation: Sequence[dict[str, str]],
        current_project_id: str | None = None,
        project_id: str | None = None,
        preferred_route: str | None = None,
    ) -> AgentGraphState:
        """Invoke the compiled graph with a clean request state."""
        state: AgentGraphState = {
            "request": request,
            "conversation": list(conversation),
            "current_project_id": current_project_id,
            "project_id": project_id,
            "preferred_route": preferred_route,
        }
        return await self.graph.ainvoke(state)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    current_project_id: str | None = None


class ConversationStore:
    """Process-local conversation state for the first POC."""

    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id, SessionState())
            return SessionState(
                messages=list(state.messages),
                current_project_id=state.current_project_id,
            )

    def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        current_project_id: str | None,
    ) -> None:
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.messages.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ])
            state.messages = state.messages[-self.max_messages :]
            if current_project_id:
                state.current_project_id = current_project_id


class AgentService:
    """Central agent application service and conversation coordinator."""

    def __init__(
        self,
        graph: ProjectAgentGraph,
        store: ConversationStore | None = None,
    ) -> None:
        self.graph = graph
        self.store = store or ConversationStore()

    async def chat(self, request: AgentRequest) -> AgentResponse:
        session_id = request.session_id or str(uuid.uuid4())
        session = self.store.get(session_id)
        conversation = request.history or session.messages
        conversation = conversation + [{"role": "user", "content": request.message}]
        logger.info(
            "Agent user request session_id=%s message=%s",
            session_id,
            request.message,
        )

        result = await self.graph.ainvoke(
            request=request.message,
            conversation=conversation,
            current_project_id=session.current_project_id,
            project_id=request.project_id,
            preferred_route=request.preferred_route,
        )
        decision = result["decision"]
        answer = result.get("final_response", "I couldn't generate a response.")
        current_project_id = result.get("current_project_id") or decision.project_id
        self.store.append(
            session_id=session_id,
            user_message=request.message,
            assistant_message=answer,
            current_project_id=current_project_id,
        )
        return AgentResponse(
            answer=answer,
            session_id=session_id,
            route=decision.route,
            selected_tool="project_api" if decision.route == "project_api" else None,
            operation=decision.operation,
            project_id=current_project_id,
            tool_error=result.get("tool_error"),
        )


def create_agent_service(
    agent_settings: AgentSettings,
    store: ConversationStore | None = None,
) -> AgentService:
    """Create the configured agent service for FastAPI dependency injection."""
    return AgentService(
        graph=ProjectAgentGraph(
            llm=AgentLLM(agent_settings),
            project_api=ProjectApiClient(
                base_url=agent_settings.project_api_base_url,
                path_prefix=agent_settings.project_api_path_prefix,
                timeout_seconds=agent_settings.project_api_timeout_seconds,
            ),
        ),
        store=store or ConversationStore(max_messages=agent_settings.memory_max_messages),
    )
