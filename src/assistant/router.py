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
    "guide",
    "manual",
    "documentation",
    "document",
    "documents",
    "uploaded",
    "upload",
    "pdf",
    "file",
    "files",
    "spec",
    "report",
    "knowledge base",
    "kb",
    "pto",
    "leave",
    "sick",
    "casual leave",
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
    "diagram",
    "workflow",
    "er diagram",
    "chart",
    "ocr",
    "table",
)

_GENERAL_KNOWLEDGE_TERMS = (
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "golang",
    "rust",
    "php",
    "html",
    "css",
    "sql",
    "nosql",
    "mongodb",
    "react",
    "vue",
    "angular",
    "node",
    "nodejs",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "algorithm",
    "algorithms",
    "data structure",
    "binary search",
    "recursion",
    "machine learning",
    "deep learning",
    "neural network",
    "artificial intelligence",
    "nlp",
    "llm",
    "rest",
    "rest api",
    "rest apis",
    "write code",
    "how to code",
    "programming",
    "write a script",
    "debug this code",
    "write a function",
    "what is a class",
    "what is an object",
    "who is",
    "who was",
    "tell me a joke",
    "write a poem",
)

_ANAPHORIC_FOLLOWUP_TERMS = (
    "explain this",
    "explain this image",
    "explain the image",
    "explain this diagram",
    "what does this mean",
    "what does this show",
    "what does it show",
    "what the work flow shows",
    "what the workflow shows",
    "show me a image",
    "show me an image",
    "give that image",
    "give image",
    "image of",
    "heading image",
    "sub heading image",
    "give the content of this",
    "tell me more about this",
    "details of this",
    "what is this",
    "explain that",
    "explain it",
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
    history: Sequence[Any] | None = None,
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

    # 2. General Knowledge / Conceptual definition checks (e.g. "what is endpoint", "what is python")
    is_general_definition = bool(
        re.match(r"^what\s+(?:is|are|does\s+\w+\s+mean)\s+(?:an?\s+)?([a-zA-Z0-9_\s]+?)\??$", lower)
    )
    has_general_coding = any(
        re.search(r"\b" + re.escape(t) + r"\b", lower) for t in _GENERAL_KNOWLEDGE_TERMS
    )
    if (has_general_coding or is_general_definition) and not has_known_doc and not has_file_ref and not has_project_id and not mentions_project:
        specific_policy_match = any(
            p in lower for p in ("pto", "leave", "handbook", "policy", "sick leave", "notice period", "resignation", "loss of pay", "wfh")
        )
        if not specific_policy_match:
            return RouteDecision(route="direct", reason="General conceptual definition or programming knowledge query.")

    # 3. Follow-up / Anaphoric resolution using history
    is_anaphoric = any(phrase in lower for phrase in _ANAPHORIC_FOLLOWUP_TERMS) or (
        any(w in lower.split() for w in ("this", "that", "it", "its", "image", "diagram", "table", "workflow", "endpoints", "endpoint"))
        and len(lower.split()) <= 7
    )
    if is_anaphoric and history and not has_project_id:
        # Check what prior assistant or user messages were discussing
        last_asst_text = ""
        for msg in reversed(history):
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if role == "assistant" and content:
                last_asst_text = content.lower()
                break

        # If previous assistant answer contained document references, images, tables, or policies
        if any(
            marker in last_asst_text
            for marker in (".docx", ".pdf", "policy", "document", "retrieved", "table", "/media/", "diagram", "workflow", "endpoints", "endpoint")
        ):
            return RouteDecision(
                route="policy",
                reason="Follow-up question directly referring to recently retrieved document content.",
            )

    # 4. Technical specification / Document element explanation (e.g. "explain the api endpoints", "define the endpoints")
    is_explain_or_define = bool(
        re.match(r"^(?:explain|define|describe|show|display|give)\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)", lower)
    )
    has_spec_terms = any(term in lower for term in ("endpoint", "endpoints", "diagram", "workflow", "table", "schema", "architecture", "er diagram"))
    if is_explain_or_define and has_spec_terms and not has_general_coding and not has_project_id:
        return RouteDecision(
            route="policy",
            reason="Technical specification or document element explanation request.",
        )

    # 5. Standard Document & Policy Routing
    if (has_known_doc or has_file_ref) and not (has_project_id and mentions_project):
        return RouteDecision(route="policy", reason="Document or file reference detected.")

    if has_policy_terms and not has_project_id and not mentions_project:
        return RouteDecision(route="policy", reason="Company policy terminology detected.")

    # 6. Project Management checks
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
    history: Sequence[Any] | None = None,
) -> RouteDecision:
    """
    Model-based semantic router that uses an LLM to accurately classify user inquiries.
    Domain-agnostic: relies on dynamic knowledge base document titles and vector similarity.
    Falls back to deterministic rules if the LLM is unconfigured, times out, or errors.
    """
    deterministic = classify_request(
        message=message,
        current_project_id=current_project_id,
        explicit_project_id=explicit_project_id,
        known_documents=known_documents,
        history=history,
    )

    if llm is None:
        return deterministic

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        recent_history_text = ""
        if history:
            recent_msgs = []
            for h in history[-4:]:
                r = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else "")
                c = getattr(h, "content", "") if not isinstance(h, dict) else h.get("content", "")
                if c:
                    recent_msgs.append(f"{r}: {c[:120]}")
            recent_history_text = "\n".join(recent_msgs)

        system_msg = SystemMessage(
            content=(
                "You are an intelligent intent classification router for an enterprise AI assistant with access to:\n"
                "1. 'policy': Questions about company documents, policies, employee handbooks, technical specs, internal guides, uploaded files, or any topic covered by indexed documents in the knowledge base. Also inquiries starting with 'explain ...', 'define ...', 'describe ...', or 'show ...' referring to architecture, API endpoints, diagrams, workflows, or prior document content (e.g., 'explain the api endpoints', 'explain this').\n"
                "2. 'project': Questions querying structured project management data (status, milestones, owner, progress, tasks) for a specific software project (e.g., PROJ-123).\n"
                "3. 'clarification': Inquiries asking for project details/status/milestones but missing a project ID.\n"
                "4. 'direct': General conceptual definition questions ('what is <term>' or 'what does <term> mean', such as 'what is endpoint', 'what is python', 'what is rest api') without referencing a specific document, or general programming/math chit-chat.\n\n"
                "Output strictly a JSON object with keys:\n"
                '{"route": "policy|project|clarification|direct", "project_id": "string or null", "reason": "brief explanation"}'
            )
        )
        user_prompt = (
            f"User message: {message}\n"
            f"Recent conversation history:\n{recent_history_text or 'none'}\n"
            f"Current project context: {current_project_id or 'none'}\n"
            f"Available documents in knowledge base: {', '.join(known_documents) if known_documents else 'none'}"
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
