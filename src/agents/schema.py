"""Pydantic schemas and shared types for the agentic assistant."""

from typing import Any, Literal

from pydantic import BaseModel, Field


ProjectOperation = Literal["details", "status", "owner", "milestones"]
AgentRoute = Literal["direct", "project_api"]


class AgentDecision(BaseModel):
    """The planner's decision about how a request should be answered."""

    route: AgentRoute
    operation: ProjectOperation | None = None
    project_id: str | None = None
    rationale: str = ""


class AgentRequest(BaseModel):
    """Request body for the separate agent endpoint."""

    message: str = Field(..., min_length=1, description="User request for the project assistant")
    session_id: str | None = Field(default=None, description="In-memory conversation identifier")
    project_id: str | None = Field(
        default=None,
        description="Optional explicit project identifier for the current request",
    )
    history: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional shared conversation history supplied by an orchestrator",
    )
    preferred_route: AgentRoute | None = Field(
        default=None,
        description="Optional route selected by an upstream orchestrator",
    )


class AgentResponse(BaseModel):
    """Response returned by the agent endpoint."""

    answer: str
    session_id: str
    route: AgentRoute
    selected_tool: str | None = None
    operation: ProjectOperation | None = None
    project_id: str | None = None
    tool_error: str | None = None
