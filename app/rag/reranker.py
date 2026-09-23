"""
Stage 6: Intent-Aware Reranker
Responsible for scoring and reranking retrieved candidate chunks based on
the detected question type and intent, and enforcing topic-isolation to prevent wrong-topic retrieval.
"""
import re
from typing import Sequence
from langchain_core.documents import Document

from app.rag.intent_detector import QuestionType


class IntentReranker:
    """Reranks and filters candidate chunks to maximize evidence-intent alignment."""

    TOPIC_SECTION_PATTERNS = {
        "insurance": [r"health insurance", r"group health", r"medical coverage", r"dental", r"vision", r"hospitalization"],
        "travel": [r"travel", r"meal allowance", r"flight", r"hotel", r"business expense", r"reimbursement"],
        "leave": [r"leave policy", r"paid time off", r"pto", r"sick leave", r"casual leave", r"maternity", r"paternity", r"bereavement"],
        "notice": [r"resignation", r"notice period", r"probation", r"buyout", r"settlement", r"f&f"],
        "remote": [r"remote work", r"hybrid", r"home office", r"internet utility", r"stipend"],
        "hardware": [r"it equipment", r"hardware", r"laptop", r"macbook", r"workstation"],
        "security": [r"information security", r"password", r"mfa", r"wireguard", r"screen lock"],
        "appraisal": [r"performance appraisal", r"rating scale", r"promotion policy", r"salary increment"],
        "conduct": [r"code of conduct", r"anti-harassment", r"posh", r"internal complaints committee"],
        "hours": [r"working hours", r"core timings", r"flexible hours"],
    }

    @classmethod
    def rerank(
        cls,
        chunks: Sequence[Document],
        question_type: QuestionType,
        target_topic: str | None = None,
    ) -> list[Document]:
        if not chunks:
            return []

        scored_chunks: list[tuple[float, Document]] = []

        for doc in chunks:
            base_score = float(doc.metadata.get("score", 0.5))
            text = doc.page_content.lower()
            bonus = 0.0

            # 1. Strict Topic Isolation
            if target_topic and target_topic in cls.TOPIC_SECTION_PATTERNS:
                matching_patterns = cls.TOPIC_SECTION_PATTERNS[target_topic]
                is_matching_section = any(re.search(pat, text) for pat in matching_patterns)
                
                # Check if this chunk belongs to a DIFFERENT, conflicting policy section
                is_conflicting_section = False
                for other_topic, other_patterns in cls.TOPIC_SECTION_PATTERNS.items():
                    if other_topic != target_topic:
                        if any(re.search(pat, text) for pat in other_patterns):
                            is_conflicting_section = True
                            break

                if is_matching_section:
                    bonus += 1.0  # Massive boost for correct section
                elif is_conflicting_section:
                    bonus -= 1.5  # Heavy penalty for cross-topic contamination

            # 2. Question type alignment bonus
            if question_type == QuestionType.AMOUNT_LIMIT:
                if re.search(r"(\$|usd|\bcap\b|\blimit\b|\ballowance\b|\bmaximum\b)", text):
                    bonus += 0.3
                if re.search(r"\$\s*\d+", text):
                    bonus += 0.4
            elif question_type == QuestionType.DEADLINE:
                if re.search(r"(\bdays?\b|\bdeadline\b|\bwithin\b|\bsubmit\b|\bportal\b)", text):
                    bonus += 0.4
            elif question_type in (QuestionType.ELIGIBILITY, QuestionType.COVERAGE):
                if re.search(r"(\bcovers?\b|\bcoverage\b|\beligible\b|\bspouse\b|\bchildren\b|\bdependents?\b|\bprobation\b)", text):
                    bonus += 0.4
            elif question_type == QuestionType.EXCLUSION:
                if re.search(r"(\bexcluded?\b|\bnot\s+covered\b|\bprohibited\b|\bexception\b|\bwithout\b)", text):
                    bonus += 0.5
            elif question_type == QuestionType.CALCULATION:
                if re.search(r"(\b18\b|\b12\b|\bdays?\b|\baccrue\b|\bcarryover\b|\broll\s+over\b)", text):
                    bonus += 0.4

            final_score = base_score + bonus
            scored_chunks.append((final_score, doc))

        # Sort descending by final score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate pages while keeping top scoring
        seen_pages = set()
        deduped: list[Document] = []
        for _, doc in scored_chunks:
            page = doc.metadata.get("page", 1)
            chunk_idx = doc.metadata.get("chunk_index", -1)
            key = (page, chunk_idx)
            if key not in seen_pages:
                seen_pages.add(key)
                deduped.append(doc)

        return deduped
