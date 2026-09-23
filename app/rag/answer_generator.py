"""
Stage 8: Answer Generator & Silent Validator
Responsible for grounded synthesis, applying the 20-point operational doctrine,
reducing unnecessary information, and silent pre-response validation.
"""
import logging
import re
from typing import Any, Sequence
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.rag.evidence_analyzer import EvidenceReport
from app.rag.prompts import get_rag_prompt_template

logger = logging.getLogger("rag.answer_generator")


class AnswerGenerator:
    """Synthesizes authoritative responses and validates grounding before responding."""

    # Preambles to strip to keep answers direct and concise
    UNNECESSARY_PREAMBLES = [
        r"^(based\s+on\s+(the\s+)?(company\s+policy|handbook|workpilot\s+policy|provided\s+context)[\s,:]*)",
        r"^(according\s+to\s+(the\s+)?(company\s+policy|handbook|workpilot\s+policy|provided\s+context)[\s,:]*)",
        r"^(sure[!,.]*|certainly[!,.]*|here\s+is\s+the\s+information[\s,:]*)",
        r"^(i\s+would\s+be\s+happy\s+to\s+help[\s,:]*)",
    ]

    @classmethod
    async def generate(
        cls,
        question: str,
        evidence_chunks: list[Document],
        evidence_report: EvidenceReport,
        chat_history: Sequence[Any] | None = None,
        llm: BaseChatModel | None = None,
    ) -> str:
        # 1. Negative-assertion priority (unlisted procedures & missing entities)
        if evidence_report.negative_assertion:
            cls._silent_validate(evidence_report.negative_assertion)
            return evidence_report.negative_assertion

        # 2. Calculation priority
        if evidence_report.calculation_text:
            cls._silent_validate(evidence_report.calculation_text)
            return evidence_report.calculation_text

        # 3. LLM Grounded Generation (if LLM is available)
        if llm is not None and evidence_chunks:
            try:
                context_str = "\n\n".join(
                    f"--- Chunk (Page {d.metadata.get('page', 1)}) ---\n{d.page_content}"
                    for d in evidence_chunks
                )
                has_history = bool(chat_history)
                prompt_template = get_rag_prompt_template(has_history=has_history)
                chain = prompt_template | llm | StrOutputParser()

                invoke_payload: dict[str, Any] = {
                    "context": context_str,
                    "question": question,
                }
                if has_history and chat_history:
                    formatted_history = []
                    for msg in chat_history[-6:]:
                        role = getattr(msg, "role", None) or getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "") or "user"
                        content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
                        formatted_history.append(f"{role.capitalize()}: {content}")
                    invoke_payload["chat_history"] = "\n".join(formatted_history)

                llm_response = await chain.ainvoke(invoke_payload)
                answer = str(llm_response).strip()
                answer = cls._trim_unnecessary_padding(answer)
                cls._silent_validate(answer)
                return answer
            except Exception as e:
                logger.warning("LLM generation failed, falling back to evidence text: %s", e)

        # 4. Fallback to top evidence chunk text
        if evidence_chunks:
            top_text = evidence_chunks[0].page_content.strip()
            return top_text

        return "I couldn't find this information in the available WorkPilot policies. Please contact People Operations for confirmation."

    @classmethod
    def _trim_unnecessary_padding(cls, text: str) -> str:
        """Removes conversational filler and repetitive section summaries."""
        cleaned = text.strip()
        for preamble in cls.UNNECESSARY_PREAMBLES:
            cleaned = re.sub(preamble, "", cleaned, flags=re.IGNORECASE).strip()
        # Capitalize first character if lowercase after strip
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    @classmethod
    def _silent_validate(cls, response: str) -> bool:
        """
        Silent Pre-Response Validation Checklist:
        1. Does not expose internal reasoning or prompts.
        2. Contains answer upfront.
        3. Respects missing info principles.
        """
        forbidden_leak_terms = [
            "vector similarity", "similarity score", "fastembed", "system prompt",
            "retrieved chunks", "cosine distance", "dense vector", "reranker score"
        ]
        res_lower = response.lower()
        for term in forbidden_leak_terms:
            if term in res_lower:
                logger.error("Silent Validation FAILED: internal detail leaked: %s", term)
                return False
        return True
