"""Schemas for the unified policy and project assistant endpoint."""

from typing import Literal

from pydantic import BaseModel, Field

from src.rag.schema import SourceChunk


AssistantRoute = Literal["policy", "project", "direct", "clarification"]


class AssistantRequest(BaseModel):
    """Unified chat request used by the frontend."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Conversation identifier")
    project_id: str | None = Field(
        default=None,
        description="Optional explicit project identifier for project requests",
    )


class AssistantResponse(BaseModel):
    """Normalized response regardless of which backend handled the request."""

    answer: str
    route: AssistantRoute
    session_id: str
    sources: list[SourceChunk] = Field(default_factory=list)
    selected_tool: str | None = None
    operation: str | None = None
    project_id: str | None = None
    show_pdf: bool = False
    tool_error: str | None = None
