"""
Stage 3: Query Rewriter
Responsible for typo normalization, colloquial phrase translation to canonical HR terms,
and conversational prefix cleaning.
"""
import re
from dataclasses import dataclass


@dataclass
class RewrittenQuery:
    original: str
    normalized: str
    canonical: str
    cleaned: str


class QueryRewriter:
    """Normalizes and translates informal employee inquiries into canonical HR terminology."""

    # Common typographical & phonetic corrections
    TYPO_CORRECTIONS = [
        (r"\b(balnce|balence|balanc)\b", "balance"),
        (r"\bnoe\s+lwft\b", "now left"),
        (r"\blwft\b", "left"),
        (r"\bnoe\b", "now"),
        (r"\b(remaing|remainig)\b", "remaining"),
        (r"\bremian\b", "remain"),
        (r"\bemploye\b", "employee"),
        (r"\bratng\b", "rating"),
        (r"\bclaimabke\b", "claimable"),
        (r"\bcoverble\b", "coverable"),
        (r"\bresion\b", "resignation"),
        (r"\bresioning\b", "resigning"),
        (r"\bresiging\b", "resigning"),
        (r"\bfinace\b", "finance"),
        (r"\balchohol\b", "alcohol"),
        (r"\binsurnce\b", "insurance"),
        (r"\bhospitl\b", "hospital"),
        (r"\bvacaton\b", "vacation"),
        (r"\bparternity\b", "paternity"),
        (r"\bnotce\b", "notice"),
        (r"\bprobashun\b", "probation"),
        (r"\bdoc\s+note\b", "medical certificate"),
        (r"\bdoctor\s+note\b", "medical certificate"),
        (r"\bdoctors\s+note\b", "medical certificate"),
    ]

    # Colloquial workplace terminology mappings to canonical HR policy concepts
    COLLOQUIAL_MAPPINGS = [
        (r"\b(food\s+allowance|dinner\s+claim|lunch\s+claim|food\s+expense|food\s+reimbursement|eating\s+allowance)\b", "meal reimbursement allowance"),
        (r"\b(kids|children|wife|husband|spouse|family)\s+(coverage|covered|claim|insurance)\b", "dependent health insurance coverage"),
        (r"\b(quit|quitting|leave\s+job|leaving\s+the\s+company|exit\s+company)\b", "resignation notice period"),
        (r"\b(wifi|broadband|internet\s+bill)\b", "monthly internet utility allowance"),
        (r"\b(gym|workout|fitness|yoga|sports)\s+(allowance|reimbursement|money|benefit)\b", "gym and fitness reimbursement"),
        (r"\b(therapy|counseling|counselling|mental\s+health)\b", "employee assistance program therapy sessions"),
        (r"\b(laptop|macbook|dell|computer)\s+(upgrade|replace|new|refresh)\b", "it hardware 3-year refresh cycle"),
        (r"\b(bonus|raise|salary\s+hike|increment)\b", "annual performance appraisal compensation review"),
        (r"\b(flight\s+ticket|plane\s+ticket|airline\s+ticket)\b", "domestic flight economy premium economy class"),
        (r"\b(hotel\s+stay|hotel\s+room|lodging)\b", "hotel accommodation reimbursement limit tier-1 cities"),
    ]

    # Conversational fillers to strip from query start
    CONVERSATIONAL_PREFIXES = [
        r"^(so|and|then|also|ok|okay|now|well|please|tell\s+me|give\s+me|can\s+you\s+give|can\s+you\s+tell)\s+",
        r"^(so\s+what\s+about|what\s+about|how\s+about)\s+",
    ]

    @classmethod
    def rewrite(cls, question: str) -> RewrittenQuery:
        q_raw = question.strip()
        q_norm = q_raw.lower()

        # 1. Apply Typo Corrections
        for pattern, replacement in cls.TYPO_CORRECTIONS:
            q_norm = re.sub(pattern, replacement, q_norm)

        # 2. Strip Conversational Fillers for clean matching
        q_clean = q_norm
        for prefix_pat in cls.CONVERSATIONAL_PREFIXES:
            q_clean = re.sub(prefix_pat, "", q_clean).strip()

        # 3. Formulate Canonical HR Query
        canonical = q_clean
        for pattern, hr_concept in cls.COLLOQUIAL_MAPPINGS:
            if re.search(pattern, q_clean):
                canonical = re.sub(pattern, hr_concept, canonical)

        return RewrittenQuery(
            original=q_raw,
            normalized=q_norm,
            canonical=canonical,
            cleaned=q_clean,
        )
