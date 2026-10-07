"""
Central assistant router.
Supports both deterministic rule-based classification and model-based (LLM) classification
to intelligently route inquiries between Policy/Document RAG, Project Management Agent, and Direct LLM.
"""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger("src.assistant.router")

AssistantRouteName = Literal["policy", "project", "direct", "clarification"]

# Legitimate project IDs follow formats like PROJ-123, PRJ-456, or PROJECT-123
_EXPLICIT_PROJECT_KEY_PATTERN = re.compile(
    r"\b(?:PROJ|PRJ|PROJECT)[-_][A-Za-z0-9_-]+\b|\b[A-Z]{2,6}-\d+\b",
    re.IGNORECASE,
)
_PROJECT_PREFIX_PATTERN = re.compile(
    r"\bproject\s+([A-Za-z0-9_-]+)\b",
    re.IGNORECASE,
)

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
    "policies",
    "handbook",
    "document",
    "documents",
    "uploaded",
    "upload",
    "pdf",
    "file",
    "files",
    "knowledge base",
    "kb",
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
    "guideline",
    "guidelines",
    "rule",
    "rules",
    "procedure",
    "procedures",
    "contract",
    "agreement",
    "clause",
    "summary",
    "summarize",
)

_EXCLUDED_FROM_PROJECT_ID = {
    "status", "owner", "milestone", "milestones", "details", "progress",
    "info", "update", "updates", "workpilot", "company", "policy", "handbook",
    "leave", "document", "documents", "file", "files", "pdf", "txt", "overview",
}


def extract_project_id(message: str) -> str | None:
    """Extracts a valid project ID from the message, strictly ignoring filenames and document terms."""
    # Check standard project keys (e.g. PROJ-123, PRJ-101)
    m = _EXPLICIT_PROJECT_KEY_PATTERN.search(message)
    if m:
        val = m.group(0)
        # Exclude if it looks like a filename (e.g. .pdf, .txt)
        if not any(val.lower().endswith(ext) for ext in (".pdf", ".txt", ".docx", ".doc", ".md")):
            return val

    # Check 'project <id>' phrase (e.g. 'project alpha', 'project 123')
    m2 = _PROJECT_PREFIX_PATTERN.search(message)
    if m2:
        val = m2.group(1).strip()
        clean = val.lower().rstrip(".,:;!?")
        if (
            clean not in _EXCLUDED_FROM_PROJECT_ID
            and not any(clean.endswith(ext) for ext in (".pdf", ".txt", ".docx", ".doc", ".md"))
            and not any(term in clean for term in ("policy", "leave", "handbook", "company"))
        ):
            return val

    return None


class RouteDecision(BaseModel):
    route: AssistantRouteName
    reason: str
    project_id: str | None = None


def classify_request(
    message: str,
    current_project_id: str | None = None,
    explicit_project_id: str | None = None,
    known_documents: Sequence[str] | None = None,
) -> RouteDecision:
    """
    Deterministic rule-based intent router.
    Categorizes the request into 'policy' (RAG), 'project' (Project API Agent),
    'clarification', or 'direct' (General LLM).
    """
    lower = message.lower().strip()
    detected_project_id = explicit_project_id or extract_project_id(message)
    has_project_id = bool(detected_project_id)
    has_project_action = any(term in lower for term in _PROJECT_ACTION_TERMS)
    mentions_project = any(term in lower for term in _PROJECT_LOOKUP_TERMS)
    refers_to_current_project = bool(
        current_project_id
        and any(phrase in lower for phrase in ("its ", "their ", "this project", "that project"))
        and has_project_action
    )

    # 1. Document / Knowledge Base checks
    has_policy_terms = any(term in lower for term in _POLICY_TERMS)
    has_known_doc = bool(known_documents and any(doc.lower() in lower for doc in known_documents))
    has_file_ref = any(ext in lower for ext in (".pdf", ".txt", ".docx", ".doc"))

    # If it asks about a document, policy, or file, prioritize RAG (policy)
    if (has_known_doc or has_file_ref) and not (has_project_id and mentions_project):
        return RouteDecision(route="policy", reason="Document or file reference detected.")

    if has_policy_terms and not has_project_id and not mentions_project:
        return RouteDecision(route="policy", reason="Company policy terminology detected.")

    # 2. Project Management checks
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

    if has_project_action and not has_policy_terms:
        return RouteDecision(
            route="clarification",
            reason="The request appears to ask for project-specific data but has no project context.",
        )

    if has_policy_terms:
        return RouteDecision(route="policy", reason="Company policy terminology detected.")

    return RouteDecision(route="direct", reason="No policy or project-specific intent detected.")


async def classify_request_with_model(
    message: str,
    current_project_id: str | None = None,
    explicit_project_id: str | None = None,
    known_documents: Sequence[str] | None = None,
    llm: Any = None,
) -> RouteDecision:
    """
    Model-based semantic router that uses an LLM to accurately classify user inquiries.
    Falls back to deterministic rules if the LLM is unconfigured, times out, or errors.
    """
    deterministic = classify_request(
        message=message,
        current_project_id=current_project_id,
        explicit_project_id=explicit_project_id,
        known_documents=known_documents,
    )

    if llm is None:
        return deterministic

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        system_msg = SystemMessage(
            content=(
                "You are an intent classification router for an enterprise AI assistant.\n"
                "Classify the user inquiry into one of these 4 routes:\n"
                "- 'policy': Questions about company policies, employee handbooks, HR rules, benefits, working hours, leave, or contents of uploaded documents/files.\n"
                "- 'project': Questions querying structured project management facts (status, milestones, owner, progress) for a specific software project (e.g., PROJ-123).\n"
                "- 'clarification': Inquiries asking for project details/status/milestones but missing a project ID.\n"
                "- 'direct': General knowledge questions, programming help, math, or conversational chit-chat.\n\n"
                "Output strictly a JSON object with keys:\n"
                '{"route": "policy|project|clarification|direct", "project_id": "string or null", "reason": "brief explanation"}'
            )
        )
        user_prompt = (
            f"User message: {message}\n"
            f"Current project context: {current_project_id or 'none'}\n"
            f"Known documents: {', '.join(known_documents) if known_documents else 'none'}"
        )
        resp = await llm.ainvoke([system_msg, HumanMessage(content=user_prompt)])
        content = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            route = data.get("route")
            if route in ("policy", "project", "clarification", "direct"):
                return RouteDecision(
                    route=route,
                    reason=data.get("reason", "Model-classified intent"),
                    project_id=data.get("project_id") or deterministic.project_id,
                )
    except Exception as e:
        logger.warning("Model-based router invocation failed (%s). Using deterministic decision.", e)

    return deterministic
