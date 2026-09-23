"""
Stage 1: Context Analyzer
Responsible for understanding conversation history, classifying topic transitions,
resolving elliptical follow-ups, and tracking multi-turn state.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class TopicTransition(str, Enum):
    NEW_TOPIC = "NEW_TOPIC"
    SAME_TOPIC = "SAME_TOPIC"
    RELATED_TOPIC = "RELATED_TOPIC"


@dataclass
class ConversationContext:
    transition: TopicTransition = TopicTransition.NEW_TOPIC
    active_topic: str | None = None
    active_subtopic: str | None = None
    prior_leave_balance: int | None = None
    prior_leave_type: str | None = None  # "pto" or "sick"
    prior_days_requested: int | None = None
    is_elliptical: bool = False
    resolved_query: str = ""
    entities: list[str] = field(default_factory=list)


class ContextAnalyzer:
    """Analyzes user queries in the context of recent chat history."""

    EXPLICIT_TOPIC_KEYWORDS = {
        "insurance": ["insurance", "hospital", "hospitalization", "inpatient", "mediclaim", "dental", "vision", "health insurance", "group health"],
        "travel": ["travel", "flight", "flights", "hotel", "hotels", "meal", "meals", "per diem", "reimbursement", "finance portal", "ticket", "tickets"],
        "leave": ["leave", "leaves", "pto", "vacation", "sick leave", "casual leave", "maternity", "paternity", "bereavement"],
        "notice": ["notice period", "resignation", "resign", "buyout", "f&f", "final settlement", "probation"],
        "remote": ["remote", "remotely", "hybrid", "wfh", "work from home", "home office", "stipend", "furniture", "mbps"],
        "hardware": ["laptop", "macbook", "dell", "monitor", "keyboard", "mouse", "headset", "hardware"],
        "security": ["password", "passwords", "mfa", "2fa", "wireguard", "vpn", "auto-lock", "screen lock"],
        "appraisal": ["appraisal", "rating", "increment", "promotion", "5-point", "review cycle", "competency"],
        "conduct": ["conduct", "harassment", "posh", "discrimination", "retaliation", "icc", "whistleblower"],
    }

    FOLLOW_UP_TRIGGERS = (
        "what about", "how about", "and what", "and for", "can i also", "can i ", "could i ", "do i ",
        "is that", "does it", "is it", "what if", "can it", "how much is that", "how much ",
        "what is the limit for that", "any allowance for that", "and ", "also ",
        "for whom", "for who", "who is", "who are", "who can", "who does", "whom", "who",
        "for my ", "for our ", "for ", "its is ", "is it ", "is that ",
        "per year", "per annum", "per day", "per time",
    )

    @classmethod
    def analyze(cls, question: str, chat_history: Sequence[Any] | None = None) -> ConversationContext:
        q_lower = question.strip().lower()
        context = ConversationContext(resolved_query=question)

        if not chat_history:
            context.transition = TopicTransition.NEW_TOPIC
            context.active_topic = cls._detect_explicit_topic(q_lower)
            return context

        # 1. Extract active topic and state chronologically from history (latest first)
        history_topic = None
        history_travel_subtopic = None
        prior_balance = None
        prior_leave_type = None
        prior_days = None

        for msg in reversed(chat_history):
            role = getattr(msg, "role", None) or getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "") or ""
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            c_lower = content.lower()

            # Travel subtopics
            if history_travel_subtopic is None:
                if any(w in c_lower for w in ["flight", "flights", "airline", "premium economy", "economy class"]):
                    history_travel_subtopic = "flight"
                elif any(w in c_lower for w in ["hotel", "hotels", "lodging", "night"]):
                    history_travel_subtopic = "hotel"
                elif any(w in c_lower for w in ["meal", "meals", "food", "lunch", "dinner", "per diem", "alcohol"]):
                    history_travel_subtopic = "meal"
                elif any(w in c_lower for w in ["finance portal", "portal", "30 days", "receipt"]):
                    history_travel_subtopic = "submission"

            # Primary policy topic
            if history_topic is None:
                for topic, kws in cls.EXPLICIT_TOPIC_KEYWORDS.items():
                    if any(kw in c_lower for kw in kws):
                        history_topic = topic
                        break

            # Prior running leave balance (e.g. "**13 PTO days left**", "**14 days left**")
            if role in ("assistant", "ai") and prior_balance is None:
                balance_match = re.search(r"\*\*(\d+)\s*(?:pto\s+)?days?\s+left\*\*", c_lower)
                if balance_match:
                    try:
                        prior_balance = int(balance_match.group(1))
                        if "pto" in c_lower:
                            prior_leave_type = "pto"
                        elif "sick" in c_lower:
                            prior_leave_type = "sick"
                    except ValueError:
                        pass

            # Prior requested days from user
            if role in ("user", "human") and prior_days is None:
                day_match = re.search(r"\b(\d+)\s*(?:days?|pto|leaves?)?\b", c_lower)
                if day_match:
                    try:
                        prior_days = int(day_match.group(1))
                    except ValueError:
                        pass

        context.prior_leave_balance = prior_balance
        context.prior_leave_type = prior_leave_type
        context.prior_days_requested = prior_days
        context.active_subtopic = history_travel_subtopic

        # 2. Check if current question explicitly introduces a new topic
        current_explicit_topic = cls._detect_explicit_topic(q_lower)

        # 3. Classify transition & resolution
        is_short = len(q_lower.split()) <= 4
        is_follow_up_syntax = any(q_lower.startswith(p) for p in cls.FOLLOW_UP_TRIGGERS) or is_short

        if current_explicit_topic:
            if history_topic and current_explicit_topic == history_topic:
                context.transition = TopicTransition.SAME_TOPIC
                context.active_topic = history_topic
            else:
                context.transition = TopicTransition.NEW_TOPIC
                context.active_topic = current_explicit_topic
        elif is_follow_up_syntax and history_topic:
            context.transition = TopicTransition.SAME_TOPIC
            context.active_topic = history_topic
            context.is_elliptical = True
        else:
            context.transition = TopicTransition.NEW_TOPIC
            context.active_topic = history_topic or "general"

        return context

    @classmethod
    def _detect_explicit_topic(cls, text: str) -> str | None:
        for topic, keywords in cls.EXPLICIT_TOPIC_KEYWORDS.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", text) for kw in keywords):
                return topic
        return None
