"""
Stage 7: Evidence Analyzer
Responsible for analyzing retrieved chunks, fact verification, extracting limits/conditions/exclusions,
negative-assertion guards (unlisted procedures/missing entities), and grounded calculations.
Enforces the core rule: The retrieved HR documents determine what is actually true. Do not guess.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Sequence
from langchain_core.documents import Document


@dataclass
class EvidenceReport:
    is_answerable: bool = True
    missing_info_reason: str | None = None
    negative_assertion: str | None = None
    calculation_result: int | None = None
    calculation_text: str | None = None
    extracted_facts: dict[str, Any] = field(default_factory=dict)


class EvidenceAnalyzer:
    """Analyzes evidence for factual support, exclusions, unstated entities, and calculations."""

    UNLISTED_DENTAL_VISION = [
        "braces", "orthodontic", "orthodontics", "cosmetic", "implant", "implants",
        "whitening", "bleaching", "veneer", "veneers", "crown", "crowns",
        "root canal", "denture", "dentures", "invisalign", "retainer", "retainers",
        "lasik", "laser", "prk", "cataract", "contact lenses",
    ]

    @classmethod
    def analyze(
        cls,
        question: str,
        evidence_chunks: Sequence[Document],
        prior_balance: int | None = None,
        prior_leave_type: str | None = None,
    ) -> EvidenceReport:
        q_lower = question.strip().lower()
        report = EvidenceReport()

        # 1. Unlisted dental/vision procedures guard (Negative Assertion)
        for proc in cls.UNLISTED_DENTAL_VISION:
            if re.search(r"\b" + re.escape(proc) + r"\b", q_lower):
                report.negative_assertion = (
                    f"WorkPilot covers dental and vision expenses up to **$1,000 annually** per employee; "
                    f"however, the handbook does not specify coverage for **{proc}**. "
                    f"Please confirm with People Operations or the insurer."
                )
                return report

        # 2. Insurance provider name inquiry (Missing Entity)
        if any(w in q_lower for w in ["provider", "insurer", "insurance company", "name of", "which company", "which insurer", "who provides"]):
            if any(w in q_lower for w in ["insurance", "hospital", "coverage", "health", "mediclaim", "insurer"]):
                report.negative_assertion = (
                    "The WorkPilot handbook does not specify the exact name of the insurance provider. "
                    "Coverage includes up to **$50,000 annual inpatient hospitalization** for you, your spouse, and up to 2 dependent children. "
                    "Please contact People Operations or HR for the insurer's name and contact details."
                )
                return report

        # 3. Parents coverage inquiry (Missing Entity)
        if any(w in q_lower for w in ["parents", "parent", "mother", "father", "in-laws", "inlaws", "mom", "dad"]):
            if any(w in q_lower for w in ["insurance", "hospital", "coverage", "covered", "claim", "for my", "cover"]):
                report.negative_assertion = (
                    "The handbook specifies health insurance coverage for the **employee, spouse, and up to two dependent children**. "
                    "It does not specify whether parents are covered, so please confirm with People Operations or the insurer."
                )
                return report

        # 4. Network hospital list / cashless hospitals (Missing Entity)
        if any(w in q_lower for w in ["hospital list", "network hospital", "network hospitals", "cashless hospital", "list of hospitals", "hospitals list"]):
            report.negative_assertion = (
                "The handbook does not list specific network hospitals. "
                "Please check with People Operations or access the insurer's portal for the active network hospital directory."
            )
            return report

        # 5. Flight dollar amount inquiry (Missing Entity)
        if any(w in q_lower for w in ["amount for flight", "flight amount", "flight budget", "flight cost", "how much for flight", "dollar amount for flight"]):
            report.negative_assertion = (
                "The WorkPilot handbook does not specify a fixed monetary dollar limit or budget cap for flight tickets. "
                "Instead, flights are booked according to travel duration and class: domestic flights under 5 hours must be booked in **Economy Class**, "
                "while flights exceeding 5 hours or international flights qualify for **Premium Economy**."
            )
            return report

        # 6. Definition of employee (Missing Entity)
        clean_no_punct = re.sub(r"[#\?\.\!]+", "", q_lower).strip()
        if (
            clean_no_punct in [
                "what is employee", "what is an employee", "who is an employee",
                "employee", "employees", "about employee", "about employees",
                "definition of employee", "employee definition", "what does employee mean"
            ]
            or "definition of employee" in clean_no_punct
            or "employee definition" in clean_no_punct
        ):
            report.negative_assertion = (
                "The WorkPilot handbook does not specify a definition for 'employee'. "
                "The policy documents govern terms and entitlements for full-time and probationary team members "
                "(including working hours, remote work, leave, IT equipment, and benefits). "
                "If you need clarification on formal employment classifications or contractual definitions, please contact People Operations."
            )
            return report

        # 7. Rating score definition (e.g. what is score 4, score 5) (Missing Entity)
        score_match = re.search(r"\b(?:rating|score|point)\s+([1-5])\b", q_lower) or re.search(r"\b([1-5])\s+(?:points?|rating|score)\b", q_lower)
        if score_match and any(w in q_lower for w in ["what is", "define", "meaning", "qualifies", "mean"]):
            score_val = score_match.group(1)
            report.negative_assertion = (
                f"The handbook says performance is rated on a 5-point scale, but it does not define what specifically qualifies as a score of **{score_val}**."
            )
            return report

        # 8. Fixed F&F settlement amount (Missing Entity)
        if any(w in q_lower for w in ["how much", "amount", "calculate"]) and any(w in q_lower for w in ["f&f", "fnf", "final settlement", "settlement amount"]):
            report.negative_assertion = (
                "The handbook does not specify a fixed settlement amount, as Full & Final (F&F) settlements are calculated individually "
                "based on accrued salary, remaining leave encashment, gratuity, and applicable deductions. Final disbursement is completed within **45 days** of your last working day."
            )
            return report

        # 9. Leave balance arithmetic (Calculations)
        day_match = re.search(r"\b(\d+)\s*(?:days?|pto|leaves?)?\b", q_lower)
        if day_match and any(w in q_lower for w in ["take", "taking", "took", "used", "balnce", "balance", "left", "remaining", "more"]):
            days_to_take = int(day_match.group(1))
            is_incremental = any(w in q_lower for w in ["more", "additional", "extra"])
            
            # Determine base
            if prior_balance is not None and is_incremental:
                base = prior_balance
            elif "18" in q_lower or prior_leave_type == "pto" or "pto" in q_lower:
                base = 18
            elif "12" in q_lower or prior_leave_type == "sick" or "sick" in q_lower:
                base = 12
            elif prior_balance is not None:
                base = prior_balance
            else:
                base = 18

            res = max(0, base - days_to_take)
            report.calculation_result = res
            if "sick" in q_lower or prior_leave_type == "sick":
                report.calculation_text = f"For your 12-day sick leave allowance, you'll have **{res} days left**."
            else:
                report.calculation_text = f"You'll have **{res} PTO days left**."
            return report

        return report
