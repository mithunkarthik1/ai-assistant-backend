"""
Stage 2: Intent Detector
Responsible for topic categorization, subtopics, question types, collision handling,
and security guardrails (prompt injection & out-of-scope).
"""
import re
from dataclasses import dataclass, field
from enum import Enum


class QuestionType(str, Enum):
    AMOUNT_LIMIT = "AMOUNT_LIMIT"
    DEADLINE = "DEADLINE"
    ELIGIBILITY = "ELIGIBILITY"
    COVERAGE = "COVERAGE"
    EXCLUSION = "EXCLUSION"
    PROCESS = "PROCESS"
    CALCULATION = "CALCULATION"
    GENERAL = "GENERAL"
    INJECTION = "INJECTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    COLLISION = "COLLISION"


@dataclass
class IntentResult:
    question_type: QuestionType = QuestionType.GENERAL
    topic: str | None = None
    subtopic: str | None = None
    is_injection: bool = False
    is_out_of_scope: bool = False
    is_collision: bool = False
    collision_topics: list[str] = field(default_factory=list)
    guardrail_response: str | None = None


class IntentDetector:
    """Detects user intent, question types, prompt injections, and collisions."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above|your|system)\s+(rules|instructions|prompts?|context)",
        r"forget\s+(all\s+)?(previous|prior|above|your|system)?\s*(rules|instructions|prompts?|context|secrets)",
        r"disregard\s+(all\s+)?(previous|prior|above|your|system)\s+(rules|instructions|prompts?|context)",
        r"override\s+(all\s+)?(previous|prior|above|your|system)\s+(rules|instructions|prompts?|context)",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(if|a|an)",
        r"pretend\s+(you\s+are|to\s+be)",
        r"new\s+instructions?",
        r"system\s+prompt",
        r"system\s+instructions?",
        r"output\s+secrets",
        r"jailbreak",
    ]

    OUT_OF_SCOPE_CATEGORIES = [
        (r"\b(weather|temperature|forecast|rain|snow|climate)\b", "weather forecasts and climate conditions", "local weather services or your preferred forecast app"),
        (r"\b(stock\s+price|share\s+price|market\s+cap|ticker|nasdaq|nyse)\b", "stock prices and market trading data", "financial data providers or your stock broker"),
        (r"\b(parking|park|valet|garage|park my car)\b", "parking arrangements", "Building Management or Workplace Facilities"),
        (r"\b(cafeteria|canteen|food\s+menu|lunch\s+menu)\b", "cafeteria or daily meal services", "Workplace Operations or Office Administration"),
    ]

    @classmethod
    def detect(cls, question: str, active_topic: str | None = None) -> IntentResult:
        q_lower = question.strip().lower()
        res = IntentResult(topic=active_topic)

        # 1. Prompt Injection Pre-Flight Check
        for pat in cls.INJECTION_PATTERNS:
            if re.search(pat, q_lower):
                res.question_type = QuestionType.INJECTION
                res.is_injection = True
                res.guardrail_response = (
                    "I'm WorkPilot's policy assistant and can only answer questions about company policies "
                    "and employee benefits from the official handbook."
                )
                return res

        # 2. Out-of-Scope Pre-Flight Check
        for pat, cat_name, dept in cls.OUT_OF_SCOPE_CATEGORIES:
            if re.search(pat, q_lower):
                res.question_type = QuestionType.OUT_OF_SCOPE
                res.is_out_of_scope = True
                res.guardrail_response = (
                    f"I couldn't find information on {cat_name} in the WorkPilot handbook. "
                    f"Please consult {dept}."
                )
                return res

        # 3. Collision Detection (Blended Domains)
        # Collision: Meal Allowance + Hardware
        if any(w in q_lower for w in ["meal", "meals", "food", "per diem"]) and any(w in q_lower for w in ["laptop", "macbook", "dell", "hardware"]):
            res.question_type = QuestionType.COLLISION
            res.is_collision = True
            res.collision_topics = ["meal allowance", "IT equipment"]
            res.guardrail_response = (
                "The **meal allowance** and **IT equipment** are separate policies and cannot be combined "
                "or used interchangeably. Each has its own eligibility criteria and limits. Would you like details about either one specifically?"
            )
            return res

        # Collision: Notice Period + Leave
        if any(w in q_lower for w in ["notice", "notice period"]) and any(w in q_lower for w in ["sick", "casual"]):
            res.question_type = QuestionType.COLLISION
            res.is_collision = True
            res.collision_topics = ["resignation notice", "sick leave"]
            res.guardrail_response = (
                "There is no specific 'notice period' for taking **sick or casual leave** — you can take sick/casual leave "
                "as needed (up to **12 days/year**), with a medical certificate required after **3 consecutive days**.\n\n"
                "Separately, the **resignation notice period** is **60 calendar days** for confirmed employees "
                "(or **30 calendar days** during probation)."
            )
            return res

        # 4. Question Type Classification
        if any(w in q_lower for w in ["how much", "amount", "limit", "cap", "max", "maximum", "$", "dollar"]):
            res.question_type = QuestionType.AMOUNT_LIMIT
        elif any(w in q_lower for w in ["when", "deadline", "within", "how many days to", "late", "by when"]):
            res.question_type = QuestionType.DEADLINE
        elif any(w in q_lower for w in ["who can", "eligible", "eligibility", "who is covered", "qualify", "probation"]):
            res.question_type = QuestionType.ELIGIBILITY
        elif any(w in q_lower for w in ["alcohol", "beer", "wine", "exclude", "exclusion", "not covered"]):
            res.question_type = QuestionType.EXCLUSION
        elif any(w in q_lower for w in ["how to", "process", "steps", "portal", "submit", "apply"]):
            res.question_type = QuestionType.PROCESS
        elif any(w in q_lower for w in ["take", "balance", "remaining", "left", "deduct", "minus"]):
            res.question_type = QuestionType.CALCULATION
        else:
            res.question_type = QuestionType.GENERAL

        return res
