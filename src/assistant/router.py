"""Lightweight deterministic router for the unified assistant POC."""

import re
from typing import Literal

from pydantic import BaseModel


AssistantRouteName = Literal["policy", "project", "direct", "clarification"]

_PROJECT_ID_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*[-_][A-Za-z0-9_-]+\b")
_PROJECT_LOOKUP_TERMS = (
    "project",
    "milestone",
    "milestones",
    "project status",
    "project owner",
    "project details",
)
_PROJECT_ACTION_TERMS = ("status", "owner", "milestone", "milestones", "details", "progress")
_POLICY_TERMS = (
    "policy",
    "handbook",
    "pto",
    "leave",
    "sick",
    "maternity",
    "paternity",
    "bereavement",
    "remote work",
    "hybrid",
    "working hours",
    "expense",
    "reimbursement",
    "insurance",
    "health benefit",
    "dental",
    "vision",
    "gym",
    "wellness",
    "password",
    "mfa",
    "vpn",
    "harassment",
    "posh",
    "notice period",
    "resignation",
    "promotion",
    "learning budget",
    "study leave",
    "laptop",
    "hardware",
)


class RouteDecision(BaseModel):
    route: AssistantRouteName
    reason: str
    project_id: str | None = None


def classify_request(
    message: str,
    current_project_id: str | None = None,
    explicit_project_id: str | None = None,
) -> RouteDecision:
    """Choose a backend without adding another LLM call to this first POC."""
    lower = message.lower().strip()
    project_match = _PROJECT_ID_PATTERN.search(message)
    detected_project_id = explicit_project_id or (project_match.group(0) if project_match else None)
    has_project_id = bool(detected_project_id)
    has_project_action = any(term in lower for term in _PROJECT_ACTION_TERMS)
    mentions_project = any(term in lower for term in _PROJECT_LOOKUP_TERMS)
    refers_to_current_project = bool(
        current_project_id
        and any(phrase in lower for phrase in ("its ", "their ", "this project", "that project"))
        and has_project_action
    )

    if has_project_id or refers_to_current_project:
        return RouteDecision(
            route="project",
            reason="Project identifier or project follow-up detected.",
            project_id=detected_project_id or current_project_id,
        )

    if mentions_project and has_project_action:
        return RouteDecision(
            route="clarification",
            reason="A project lookup was detected but no project ID is available.",
        )

    if has_project_action and not any(term in lower for term in _POLICY_TERMS):
        return RouteDecision(
            route="clarification",
            reason="The request appears to ask for project-specific data but has no project context.",
        )

    if any(term in lower for term in _POLICY_TERMS):
        return RouteDecision(route="policy", reason="Company policy terminology detected.")

    return RouteDecision(route="direct", reason="No policy or project-specific intent detected.")
