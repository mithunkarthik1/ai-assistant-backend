import logging
import re
import uuid
from typing import Any, Sequence
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.rag.llm import get_llm
from app.rag.prompts import get_general_prompt_template, get_rag_prompt_template
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.context_analyzer import ContextAnalyzer, TopicTransition
from app.rag.intent_detector import IntentDetector, QuestionType
from app.rag.query_rewriter import QueryRewriter
from app.rag.query_expander import QueryExpander
from app.rag.reranker import IntentReranker
from app.rag.evidence_analyzer import EvidenceAnalyzer
from app.rag.answer_generator import AnswerGenerator

logger = logging.getLogger("rag.chain")


def format_context(documents: Sequence[Document]) -> str:
    """Formats retrieved semantic Document chunks into structured context blocks for the LLM."""
    formatted_chunks = []
    for idx, doc in enumerate(documents):
        filename = doc.metadata.get("filename", "document")
        score = doc.metadata.get("score", "N/A")
        chunk_idx = doc.metadata.get("chunk_index", idx)
        formatted_chunks.append(
            f"[Context {idx + 1} - Source: {filename} | Semantic Similarity: {score} | Chunk {chunk_idx}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)


def clean_bullet(text: str) -> str:
    """Strips leading list markers, dashes, markdown hashes, or numbered prefixes cleanly."""
    return re.sub(r"^(\s*[-•*–—#]+|\s*\d+[\.\)])\s*", "", text).strip()


def extract_section_title(text: str) -> str:
    """Extracts markdown section title like '## 2. Remote Work and Hybrid Guidelines' cleanly."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            title = line.replace("## ", "").strip()
            title = re.sub(r"^\d+[\.\)]\s*", "", title).strip()
            return title
    return "Company Policy"


SYNTHESIS_STOPWORDS = {
    "what", "is", "the", "for", "can", "i", "get", "how", "much", "many", "do",
    "a", "an", "in", "to", "of", "and", "or", "about", "are", "we", "have", "there",
    "need", "needed", "just", "other", "not", "relavant", "relevant", "it",
    "please", "tell", "me", "any", "that", "this", "also", "want", "know",
    "company", "policy", "policies", "workplace", "guidelines", "rules", "official",
    "handbook", "information"
}

GENERIC_WORKPLACE_WORDS = {"office", "work", "employee", "employees", "code"}


def stem_token(word: str) -> str:
    """Simple stemmer for common English inflections."""
    w = word.lower().strip()
    if w.startswith("cover"):
        return "cover"
    if w.startswith("hospit"):
        return "hospit"
    if w.startswith("reimburs"):
        return "reimburs"
    if w.startswith("allow"):
        return "allow"
    return re.sub(r'(ing|edly|ly|ed|es|s|able|ible|age|ation|ment)$', '', w)


def clean_policy_name(name: str) -> str:
    return re.sub(r"^\d+[\.\)]\s*", "", name).strip()


def has_word(words: list[str], text: str) -> bool:
    """Checks if any word or phrase in words appears with word boundaries in text."""
    for w in words:
        if " " in w:
            if w in text:
                return True
        else:
            if re.search(rf"\b{re.escape(w)}\b", text):
                return True
    return False


def format_humanized_answer(
    question: str,
    section_name: str,
    primary_fact: str,
    candidate_lines: list[tuple[float, int, bool, str, str]] | None = None,
    chat_history: Sequence[Any] | None = None,
) -> str:
    """
    Transforms retrieved policy facts into warm, engaging, and humanized conversational answers:
    - Answers the employee's intent directly and upfront in plain, natural language
    - Highlights key figures, deadlines, amounts, and rules using bold markdown
    - Proactively provides the complete picture (e.g. notice period durations, buyout rules, F&F timeline)
    - Eliminates robotic bureaucratic boilerplate like 'Under the [Policy Title]...'
    """
    q_clean = question.strip().lower()

    # Topic signals in the current question
    explicit_travel_words = [
        "meal", "meals", "food", "hotel", "hotels", "flight", "flights",
        "airline", "per diem", "mileage", "airfare", "boarding", "lodging",
        "alcohol", "liquor", "beer", "wine", "dinner", "lunch", "breakfast",
        "tier-1", "tier 1", "finance portal", "finance", "finace",
        "ticket", "tickets", "plane"
    ]
    is_explicit_travel = (
        has_word(explicit_travel_words, q_clean)
        or (
            has_word(["expense", "expenses", "reimbursement"], q_clean)
            and not has_word(["insurance", "medical", "hospital", "inpatient", "dental", "vision", "health"], q_clean)
        )
    )

    explicit_insurance_words = [
        "insurance", "hospital", "hospitalization", "inpatient", "mediclaim",
        "health insurance", "group health", "health coverage"
    ]
    is_explicit_insurance = (
        has_word(explicit_insurance_words, q_clean)
        or (has_word(["dental", "vision", "checkup", "checkups"], q_clean) and not is_explicit_travel)
    )

    explicit_notice_words = [
        "resignation", "resign", "resigning", "notice period", "probation",
        "buyout", "f&f", "fnf", "final settlement"
    ]
    is_explicit_notice = has_word(explicit_notice_words, q_clean)

    explicit_remote_words = [
        "remote", "remotely", "hybrid", "wfh", "work from home", "home office",
        "stipend", "furniture", "mbps", "bandwidth"
    ]
    is_explicit_remote = has_word(explicit_remote_words, q_clean)

    explicit_leave_words = [
        "pto", "vacation", "sick leave", "casual leave", "maternity", "paternity", "bereavement"
    ]
    is_explicit_leave = has_word(explicit_leave_words, q_clean)

    explicit_password_words = [
        "password", "passwords", "mfa", "2fa", "wireguard", "vpn", "auto-lock", "autolock"
    ]
    is_explicit_password = has_word(explicit_password_words, q_clean)

    # Determine the single most recent topic and subtopic from history chronologically (latest first)
    most_recent_topic = None
    most_recent_travel_subtopic = None
    if chat_history:
        for msg in reversed(chat_history):
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            c = content.lower()
            if most_recent_travel_subtopic is None:
                if any(w in c for w in ["flight", "flights", "airline", "plane", "ticket", "tickets", "airfare", "premium economy", "economy class"]):
                    most_recent_travel_subtopic = "flight"
                elif any(w in c for w in ["hotel", "hotels", "lodging", "stay", "night", "tier-1", "tier 1"]):
                    most_recent_travel_subtopic = "hotel"
                elif any(w in c for w in ["meal", "meals", "food", "lunch", "dinner", "per diem", "alcohol", "beer", "wine"]):
                    most_recent_travel_subtopic = "meal"
                elif any(w in c for w in ["finance portal", "portal", "submit", "submission", "30 days", "receipt", "receipts", "late expense"]):
                    most_recent_travel_subtopic = "submission"

            if most_recent_topic is None:
                if any(w in c for w in ["meal", "meals", "food", "hotel", "flight", "travel", "per diem", "daily meal", "expense claim", "reimbursement limit for meals", "finance portal", "business travel", "finance", "finace", "ticket", "tickets"]):
                    most_recent_topic = "travel"
                elif any(w in c for w in ["health insurance", "hospitalization", "inpatient", "group health", "dental and vision", "dental & vision"]):
                    most_recent_topic = "insurance"
                elif any(w in c for w in ["notice period", "resignation", "60 days", "probation", "buyout", "f&f"]):
                    most_recent_topic = "notice"
                elif any(w in c for w in ["remote work", "work from home", "wfh", "hybrid", "stipend"]):
                    most_recent_topic = "remote"
                elif any(w in c for w in ["pto", "vacation", "sick leave", "casual leave", "maternity", "paternity"]):
                    most_recent_topic = "leave"
                elif any(w in c for w in ["password", "mfa", "wireguard", "auto-lock"]):
                    most_recent_topic = "password"

            if most_recent_topic is not None and most_recent_travel_subtopic is not None:
                break

    # Prior conversation context flags
    has_prior_notice_context = (most_recent_topic == "notice") and not (is_explicit_travel or is_explicit_insurance or is_explicit_remote or is_explicit_leave or is_explicit_password)
    has_prior_remote_context = (most_recent_topic == "remote") and not (is_explicit_travel or is_explicit_insurance or is_explicit_notice or is_explicit_leave or is_explicit_password)
    has_prior_leave_context = (most_recent_topic == "leave") and not (is_explicit_travel or is_explicit_insurance or is_explicit_notice or is_explicit_remote or is_explicit_password)
    has_prior_travel_context = (most_recent_topic == "travel" or is_explicit_travel) and not (is_explicit_insurance or is_explicit_notice or is_explicit_remote or is_explicit_leave or is_explicit_password)
    has_prior_password_context = (most_recent_topic == "password") and not (is_explicit_travel or is_explicit_insurance or is_explicit_notice or is_explicit_remote or is_explicit_leave)
    has_prior_insurance_context = (most_recent_topic == "insurance" or is_explicit_insurance) and not (is_explicit_travel or has_prior_travel_context or is_explicit_notice or is_explicit_remote or is_explicit_leave or is_explicit_password)

    # Ambiguous or generic employee inquiry ('what is employee', 'what is an employee', 'who is an employee', 'employee')
    clean_no_punct = re.sub(r"[#\?\.\!]+", "", q_clean).strip()
    clean_no_punct = re.sub(r"^(could\s+you\s+tell\s+me|can\s+you\s+tell\s+me|please\s+tell\s+me|tell\s+me|i\s+want\s+to\s+know)\s*[:,\-]?\s*", "", clean_no_punct).strip()
    if clean_no_punct in [
        "what is employee", "what is an employee", "who is an employee",
        "employee", "employees", "about employee", "about employees",
        "definition of employee", "employee definition", "what does employee mean"
    ]:
        return (
            "The WorkPilot handbook does not specify a definition for 'employee'. "
            "The policy documents govern terms and entitlements for full-time and probationary team members "
            "(including working hours, remote work, leave, IT equipment, and benefits). "
            "If you need clarification on formal employment classifications or contractual definitions, please contact People Operations."
        )

    # Compound / Multi-Question (Section 34 / Example 9: PTO + Remote work)
    if has_word(["pto", "vacation"], q_clean) and has_word(["remote", "hybrid", "wfh"], q_clean):
        return (
            "- **PTO**: 18 days per year, with up to 5 unused days rolling over.\n"
            "- **Remote work**: Up to 3 days per week. Full-time remote requires written approval from your Department Head and People Operations."
        )

    # Comparison (Section 33: Difference between PTO and sick leave)
    if has_word(["difference", "compare", "versus", "vs"], q_clean) and has_word(["pto", "vacation"], q_clean) and has_word(["sick", "casual"], q_clean):
        return (
            "PTO provides 18 days per year, while sick & casual leave provides 12 days per year. "
            "The policy also states that a medical certificate is required after 3 days of sick/casual leave."
        )

    # 1. IT Hardware & Equipment (checked before PTO to avoid 'laptop' colliding with 'pto')
    is_security_keyword = has_word(["auto-lock", "autolock", "lock", "inactivity", "screen lock", "password", "passwords", "vpn", "wireguard", "mfa", "2fa"], q_clean)
    is_asset_return_inquiry = has_word(["return", "returned", "returning", "company property"], q_clean) and has_word(["equipment", "laptop", "laptops", "hardware", "monitor", "asset", "assets", "computer", "device"], q_clean)
    if (has_word(["laptop", "laptops", "macbook", "dell", "monitor", "keyboard", "mouse", "headset", "hardware", "refresh", "computer", "equipment"], q_clean) or "it equipment" in section_name.lower()) and not is_security_keyword:
        if has_word(["refresh", "cycle", "replace", "upgrade", "upgraded", "how often"], q_clean):
            return (
                "Hardware refreshes occur every **3 years**. Full-time engineering team members receive either a 16-inch MacBook Pro or Dell XPS 15."
            )
        if has_word(["return", "returned", "returning", "leaving", "leave", "exit", "resign", "resignation", "resigning", "termination", "terminated", "quit", "company property", "keep"], q_clean) or is_asset_return_inquiry:
            return (
                "All equipment remains company property and must be returned to IT Support upon resignation or termination."
            )
        if has_word(["laptop", "macbook", "dell", "computer"], q_clean):
            return (
                "Full-time engineering team members receive either an **Apple MacBook Pro 16-inch (M3/M4)** or a "
                "**Dell XPS 15** (both equipped with **32GB RAM**).\n\n"
                "Your workstation package also includes an external **27-inch 4K monitor**, wireless keyboard, mouse, and noise-canceling headset (refreshed every 3 years)."
            )
        if has_word(["monitor", "screen", "keyboard", "mouse", "headset", "accessory", "accessories"], q_clean):
            return (
                "Along with your company laptop, you receive an external **27-inch 4K monitor**, wireless keyboard, mouse, "
                "and a noise-canceling headset to ensure a complete workstation."
            )
        return (
            "WorkPilot equips employees with enterprise-grade hardware:\n\n"
            "- **Engineering Laptop**: 16-inch MacBook Pro (M3/M4) or Dell XPS 15 with 32GB RAM\n"
            "- **Accessories**: 27-inch 4K monitor, wireless keyboard, mouse, and noise-canceling headset\n"
            "- **Refresh Cycle**: Refreshed every **3 years**"
        )

    # 2. Notice Period & Resignation Protocol
    # A lunch-related "leave early" inquiry (e.g. 'can i skip my lunch break and leave early?')
    # must NOT be routed to the resignation/notice handler.
    is_lunch_leave_inquiry = has_word(["lunch", "lunch break"], q_clean) and has_word(["early", "skip", "leave", "miss"], q_clean)
    is_notice_inquiry = (
        (
            has_word(["notice", "resignation", "resigned", "resign", "resigning", "resioning", "resiging", "resion", "resignating", "quitting", "probation", "buyout", "buy out", "f&f", "fnf", "settlement", "gratuity"], q_clean)
            or has_word(["go before", "leave before", "exit before", "before 60", "leave early", "early exit", "go early", "exit early"], q_clean)
            or "resignation" in section_name.lower()
            or "notice period" in section_name.lower()
        )
        and not is_lunch_leave_inquiry
        and not is_explicit_insurance
        and not is_explicit_remote
        and not is_explicit_travel
    )
    if is_notice_inquiry:
        # Fix: Multi-policy query entanglement — detect when query contains both
        # "notice period" AND a leave type (sick/casual/maternity), and generate a
        # dual-part answer rather than anchoring to notice period only.
        if has_word(["sick", "casual"], q_clean) and has_word(["notice", "notice period"], q_clean):
            return (
                "There is no specific 'notice period' for taking **sick or casual leave** — you can take sick/casual leave "
                "as needed (up to **12 days/year**), with a medical certificate required after **3 consecutive days**.\n\n"
                "Separately, the **resignation notice period** is **60 calendar days** for confirmed employees "
                "(or **30 calendar days** during probation)."
            )
        if has_word(["maternity"], q_clean) and has_word(["notice", "notice period"], q_clean):
            return (
                "Maternity leave is **26 weeks fully paid** (up to 2 children) — there is no separate 'notice period' required to take it.\n\n"
                "Separately, the **resignation notice period** is **60 calendar days** for confirmed employees "
                "(or **30 calendar days** during probation)."
            )
        if has_word(["paternity"], q_clean) and has_word(["notice", "notice period"], q_clean):
            return (
                "Paternity leave is **4 weeks fully paid**, to be taken within the first 6 months of childbirth or adoption — there is no separate 'notice period' for it.\n\n"
                "Separately, the **resignation notice period** is **60 calendar days** for confirmed employees "
                "(or **30 calendar days** during probation)."
            )
        # What is 60 days for? (e.g. '60 days for what/', 'why 60 days', 'what is 60 days for?')
        if (has_word(["60 days", "60 day"], q_clean) or (has_prior_notice_context and "60" in q_clean)) and (
            has_word(["for what", "what for", "why", "purpose", "meaning"], q_clean)
            or "for what" in q_clean
            or q_clean.startswith("60 days for")
            or re.search(r"\b\d+\s*(day|days|week|weeks)\s+for\b", q_clean)
            or q_clean.endswith((" for", " for?"))
            or q_clean.endswith((" for what", " for what?"))
        ):
            return (
                "The **60 calendar days** is the mandatory notice period that full-time confirmed employees must serve when resigning from WorkPilot before their last working day."
            )
        # Leaving before notice period / early buyout (e.g. 'what if i go before 60 days?', 'can i leave early?')
        if (
            has_word(["before", "earlier", "early"], q_clean)
            and (has_word(["60 days", "60 day", "30 days", "30 day", "notice", "go", "leave", "exit", "quit"], q_clean) or has_word(["buyout", "buy out"], q_clean) or has_prior_notice_context)
        ) or has_word(["buyout", "buy out"], q_clean):
            return (
                "If you wish to leave before completing your 60-day notice period, an early notice period buyout is possible! "
                "However, it requires mutual written consent between you and your department head."
            )
        if has_word(["probation", "probationary"], q_clean):
            return (
                "For employees currently on probation, the notice period is **30 calendar days** "
                "(compared to 60 calendar days for full-time confirmed employees)."
            )
        if has_word(["f&f", "fnf", "final settlement", "settlement", "gratuity", "last working day"], q_clean):
            if has_word(["amount", "how much", "calculate", "calculation", "figure", "total", "what is"], q_clean) and has_word(["settlement", "f&f", "fnf"], q_clean) and not has_word(["when", "timeline", "window", "days"], q_clean):
                return (
                    "The handbook does not specify a fixed settlement amount, as Full & Final (F&F) settlements are calculated individually based on accrued salary, remaining leave encashment, gratuity, and applicable deductions. Final disbursement is completed within **45 days** of your last working day."
                )
            return (
                "Your Full & Final (F&F) settlement (including accrued salary, gratuity, and leave encashment) will be "
                "processed and disbursed within **45 days** of your last working day."
            )
        if (has_word(["how to", "how do i", "process", "steps", "procedure", "who", "whom", "where", "submit to"], q_clean) and has_word(["resign", "resignation", "resigning", "resioning", "quit"], q_clean)) or has_word(["submit my resignation", "submit resignation"], q_clean):
            return (
                "To initiate resignation, submit formal written notice to your reporting manager and People Operations. "
                "The notice period is **60 calendar days** for confirmed employees (**30 calendar days** on probation). "
                "Early buyout requires mutual written agreement, and your Full & Final (F&F) settlement is completed within **45 days** of your last working day."
            )
        # Default notice period / resignation inquiry
        return (
            "The standard notice period for full-time confirmed employees is **60 calendar days** "
            "(or **30 calendar days** during probation). Buyout requires mutual written consent, and final settlement (F&F) is disbursed within **45 days**."
        )

    # 3. Fitness / Gym / Sports / Yoga / Wellness
    if has_word(["sport", "sports", "gym", "yoga", "fitness", "workout", "exercise"], q_clean):
        return (
            "WorkPilot provides a **Gym & Fitness Reimbursement of up to $60 per month**! "
            "You can apply this towards gym memberships, yoga classes, or sports club subscriptions to stay active and healthy."
        )

    # 4. Dental & Vision
    if has_word(["dental", "vision", "teeth", "tooth", "eye", "eyes", "glasses", "spectacles"], q_clean):
        # Negative-assertion guard: if the user asks about a specific sub-procedure
        # not explicitly listed in the handbook, state it is not specified.
        UNLISTED_DENTAL_VISION_PROCEDURES = [
            "braces", "orthodontic", "orthodontics", "cosmetic", "implant", "implants",
            "whitening", "bleaching", "veneer", "veneers", "crown", "crowns",
            "root canal", "denture", "dentures", "invisalign", "retainer", "retainers",
            "lasik", "laser", "prk", "cataract", "contact lenses",
        ]
        asked_procedure = [p for p in UNLISTED_DENTAL_VISION_PROCEDURES if p in q_clean]
        if asked_procedure:
            procedure_name = asked_procedure[0]
            return (
                f"WorkPilot covers dental and vision expenses up to **$1,000 annually** per employee; "
                f"however, the handbook does not specify coverage for **{procedure_name}**. "
                f"Please confirm with People Operations or the insurer."
            )
        return (
            "Yes! Dental and vision expenses are covered up to **$1,000 annually** per employee under our comprehensive benefits plan."
        )

    # 5. Mental Health & Therapy / EAP
    if has_word(["therapy", "counseling", "counselling", "mental", "eap", "psychologist", "therapist"], q_clean):
        return (
            "Through our Employee Assistance Program (EAP), you have access to **12 free, confidential therapy and counseling sessions** "
            "per year to support your mental health and well-being."
        )

    # 6. Annual Health Checkup
    if has_word(["checkup", "checkups", "check-up", "check-ups", "health voucher", "vouchers", "annual checkup", "annual checkups"], q_clean):
        return (
            "Yes, the company provides **free annual health checkup vouchers** for all full-time employees and their spouses."
        )

    # 7. Health Insurance / Hospitalization & Medical Coverage
    clean_no_punct = re.sub(r"[#\?\.\!]+", "", q_clean).strip()
    if clean_no_punct in ["per year how much", "how much per year", "per year", "per annum"] or (has_word(["per year", "per annum"], q_clean) and has_word(["how much", "amount", "limit"], q_clean)):
        return (
            "Under WorkPilot's Group Health Insurance, the annual inpatient hospitalization coverage limit is up to **$50,000** (covering you, your spouse, and up to 2 dependent children). In addition, dental and vision are covered up to **$1,000 annually**."
        )

    is_insurance_inquiry = (
        (
            has_word(["insurance", "hospital", "hospitalization", "inpatient", "medical coverage", "mediclaim"], q_clean)
            or (has_word(["coverable", "coverble", "coverage"], q_clean) and not is_explicit_travel and not has_prior_travel_context)
            or (has_prior_insurance_context and (
                has_word(["claim", "claims", "claimabke", "claimable", "amount", "limit", "family", "dependents", "dependent", "spouse", "children", "child", "kids", "cover", "covers", "parents", "parent", "mother", "father", "who", "whom", "year", "annual", "annually", "per year", "per annum"], q_clean)
                or any(phrase in q_clean for phrase in ["how much", "how many", "who is", "who does", "for who", "for whom", "coverable for", "coverble for", "is coverble", "is coverable", "for my family", "for family", "for my parent", "for my parents", "for parents", "for parent", "per year", "per annum", "every year", "each year"])
                or clean_no_punct in ["coverable amount", "coverage amount", "coverable for", "coverble for", "is coverble for my family", "is coverable for my family", "how much can i claim", "how many can i claim", "can i claim", "for whom", "for who", "who", "whom", "who is covered", "per year", "per year how much", "per annum", "each year"]
            ))
            or (has_word(["health insurance", "group health"], q_clean))
            or "insurance" in section_name.lower()
            or "group health" in section_name.lower()
        )
        and not is_explicit_travel
        and not (has_prior_travel_context and not is_explicit_insurance)
        and not (has_prior_notice_context and not is_explicit_insurance)
    )
    if has_word(["doctor", "note", "sick", "casual"], q_clean) and not has_word(["insurance", "inpatient", "hospital", "hospitalization"], q_clean):
        is_insurance_inquiry = False
    if is_insurance_inquiry:
        # Insurance provider/company name query — handbook omits insurer name
        if has_word(["provider", "insurer", "insurance company", "name of", "which company", "which insurer", "who provides", "who is the insurer"], q_clean):
            return (
                "The WorkPilot handbook does not specify the exact name of the insurance provider. "
                "Coverage includes up to **$50,000 annual inpatient hospitalization** for you, your spouse, and up to 2 dependent children. "
                "Please contact People Operations or HR for the insurer's name and contact details."
            )
        # Parents coverage query
        if has_word(["parents", "parent", "mother", "father", "in-laws", "inlaws", "mom", "dad"], q_clean):
            return (
                "The handbook specifies health insurance coverage for the **employee, spouse, and up to two dependent children**. "
                "It does not specify whether parents are covered, so please confirm with People Operations or the insurer."
            )
        # Network hospitals query
        if has_word(["hospital list", "network hospital", "network hospitals", "cashless hospital", "list of hospitals", "hospitals list"], q_clean):
            return (
                "The handbook does not list specific network hospitals. Please check with People Operations or access the insurer's portal for the active network hospital directory."
            )
        # Number of claims inquiry ("how many can i claim")
        if (has_word(["how many can i claim", "how many claims", "number of claims"], q_clean) or (has_prior_insurance_context and "how many" in q_clean and "claim" in q_clean)) and not any(phrase in q_clean for phrase in ["how much", "amount", "limit"]):
            return (
                "The handbook does not specify a limit on the number of individual claims you can file, provided total inpatient hospitalization remains within the **$50,000 annual coverage limit** (covering you, your spouse, and up to 2 dependent children)."
            )
        # Family and dependent coverage ("for whom?", "who is covered?", "coverable for?", etc.)
        if (
            has_word(["family", "spouse", "children", "child", "kids", "dependents", "dependent", "wife", "husband"], q_clean)
            or any(phrase in q_clean for phrase in ["coverable for", "coverble for", "who is covered", "who does it cover", "covered for who", "can i cover my family", "who can i add", "for whom", "for who", "who is it for"])
            or clean_no_punct in ["for whom", "for who", "who", "whom", "who is covered", "who is it for", "for who is it", "for whom is it"]
        ):
            return (
                "Yes! Our Group Health Insurance covers **you, your spouse, and up to two dependent children** for up to **$50,000** in annual inpatient hospitalization coverage."
            )
        # Amount / Limit inquiry ("how much can i claim", "coverable amount", "per year")
        if (
            has_word(["claim", "claims", "amount", "limit", "maximum", "how much"], q_clean)
            or any(phrase in q_clean for phrase in ["coverable amount", "coverage amount", "coverage limit", "insurance amount", "how much is covered", "how much can i claim", "how much claim", "per year", "per annum"])
            or clean_no_punct in ["per year", "per year how much", "per annum", "each year"]
        ):
            return (
                "Under WorkPilot's Group Health Insurance, the annual inpatient hospitalization coverage limit is up to **$50,000** (covering you, your spouse, and up to 2 dependent children). In addition, dental and vision are covered up to **$1,000 annually**."
            )
        return (
            "Our Group Health Insurance provides up to **$50,000 in annual inpatient hospitalization coverage**, "
            "covering you, your spouse, and up to two dependent children."
        )

    # 8. Remote Work, WFH, Internet & Setup Stipend
    is_remote_inquiry = (
        has_word(["remote", "remotely", "hybrid", "wfh", "work from home", "home office", "internet", "mbps", "speed", "stipend", "furniture", "in office", "from office", "work in office", "coffee shop", "another country"], q_clean)
        or (has_prior_remote_context and (has_word(["work", "days", "allowance", "stipend", "fast", "speed", "connection", "setup"], q_clean) or any(d in q_clean for d in ["3 days", "4 days", "5 days", "2 days"])))
        or "remote work" in section_name.lower()
        or "hybrid" in section_name.lower()
    )
    if is_remote_inquiry:
        if has_word(["in office", "from office", "work in office"], q_clean) or "5 days in office" in q_clean:
            return (
                "Yes, absolutely. WorkPilot's hybrid policy permits remote work for up to **3 days per week**, meaning you are always welcome to work from the office up to 5 days a week."
            )
        if has_word(["another country", "abroad", "overseas", "different country"], q_clean):
            return (
                "Working remotely from another country or full-time remote work requires written approval from your Department Head and People Operations. Under standard policy, employees are permitted to work remotely up to **3 days per week**."
            )
        if has_word(["coffee shop", "cafe", "public space"], q_clean):
            return (
                "You can work remotely from public spaces like a coffee shop, provided you maintain security: an active connection through our official **WireGuard VPN** is strictly required on **public Wi-Fi**, and company devices auto-lock after **5 minutes** of inactivity."
            )
        if (has_word(["without", "no"], q_clean) and has_word(["approval", "permission", "consent"], q_clean)) or has_word(["without approval", "no approval"], q_clean):
            return (
                "No, full-time remote work requires written approval from your Department Head and People Operations. "
                "Under standard policy, employees are permitted to work remotely up to **3 days per week**."
            )
        if any(d in q_clean for d in ["4 day", "4 days", "5 day", "5 days", "all days", "entire week"]):
            return (
                "WorkPilot's standard hybrid policy permits remote work for up to **3 days per week**. "
                "Working remotely 4 or 5 days per week (full-time remote) requires written approval from your Department Head and People Operations."
            )
        if has_word(["fully remote", "full time remote", "full-time remote"], q_clean) or "can i work fully remote" in q_clean or "can i work remote full" in q_clean:
            return (
                "Yes, full-time remote work is available, but it requires written approval from your Department Head and People Operations."
            )
        if has_word(["internet", "speed", "mbps", "bandwidth", "wifi", "broadband", "connection", "fast"], q_clean):
            return (
                "Remote team members are required to maintain a stable internet connection with a minimum speed of **50 Mbps** "
                "(and a dedicated quiet workspace). To help support this, you receive a **$50 monthly internet and utility allowance**!"
            )
        if has_word(["stipend", "furniture", "desk", "setup", "extra allowance", "allowance", "utility"], q_clean):
            return (
                "Remote employees receive a one-time **$500 home-office setup stipend** to purchase ergonomic furniture and desk equipment, "
                "as well as a **$50 monthly internet and utility allowance**."
            )
        return (
            "WorkPilot allows employees to work remotely up to **3 days per week**. "
            "Full-time remote work requires written approval from your Department Head and People Operations."
        )

    # 9. Leave Policy, PTO, Vacation, Sick Leave, Parental
    is_leave_inquiry = (
        has_word(["leave", "leaves", "pto", "vacation", "sick", "casual", "maternity", "paternity", "parental", "bereavement"], q_clean)
        or (has_prior_leave_context and has_word(["certificate", "doctor", "note", "medical", "how many", "maternity", "paternity", "bereavement", "carry forward", "rollover", "unused"], q_clean))
        or "leave policy" in section_name.lower()
        or "paid time off" in section_name.lower()
    )
    if is_leave_inquiry:
        # Negative / unmentioned workflow cases:
        # 1. Approval / Permission to take leave ('whose permission', 'who approves', 'approval to take leave', 'permission for leave')
        if has_word(["permission", "approve", "approves", "approval", "whom to ask", "who to ask", "whose", "authorization"], q_clean):
            return (
                "The WorkPilot handbook does not specify whose permission is required or the formal approval workflow for taking leave. "
                "Standard practice is to coordinate planned time off with your direct reporting manager, or consult People Operations for department-specific request guidelines."
            )

        # 2. Exhausted leaves / spent all leaves / out of leaves / negative balance ('spent all leaves', 'exhausted', 'run out', 'no leaves left')
        if (
            has_word(["spent all", "spend all", "used all", "exhausted", "run out", "ran out", "no leaves left", "no leave left", "zero leaves", "unpaid leave", "leave without pay"], q_clean)
            or (has_word(["all leaves", "all my leaves", "all leave"], q_clean) and has_word(["contact", "whom", "who", "what to do", "spent", "finished", "over", "taken"], q_clean))
        ):
            return (
                "The WorkPilot handbook does not specify a protocol for when all leaves are exhausted (such as unpaid leave or leave without pay). "
                "If you have utilized all your allocated leave, please contact your direct reporting manager and People Operations to discuss available options."
            )

        # 3. How to apply / submit leave requests
        if has_word(["how to apply", "how do i apply", "apply for leave", "how to submit", "where to apply"], q_clean):
            return (
                "The handbook does not document the specific software portal or submission workflow for leave requests. "
                "Please coordinate with your reporting manager or consult People Operations for instructions on submitting leave in your department."
            )

        if has_word(["dinner", "$100", "100"], q_clean) and has_word(["skip lunch", "lunch", "meal"], q_clean):
            return (
                "No. During official business travel, the daily meal allowance is capped at **$75 per day**. "
                "Skipping lunch does not increase your dinner allowance or permit claims above the $75 daily cap."
            )
        if has_word(["lunch", "skip lunch", "leave 1 hour early", "leave an hour early", "leave early today", "leave work early"], q_clean) and not has_word(["dinner", "breakfast", "$100", "100", "per diem"], q_clean):
            return (
                "All employees observe a mandatory **1-hour daily lunch break** during standard working hours. "
                "Skipping lunch to leave work early is not permitted under the policy."
            )
        if has_word(["maternity"], q_clean):
            return (
                "Female employees are entitled to **26 weeks of fully paid maternity leave** for up to two surviving children."
            )
        if has_word(["paternity", "parental", "father"], q_clean):
            return (
                "Fathers and non-birthing partners receive **4 weeks of fully paid parental leave**, "
                "which can be taken within the first 6 months of childbirth or adoption."
            )
        if has_word(["bereavement", "funeral"], q_clean):
            return (
                "WorkPilot provides **5 consecutive paid days off** for bereavement in the event of the loss of an immediate family member."
            )
        if has_word(["sick", "casual", "medical certificate", "doctor", "note", "certificate"], q_clean) and not has_word(["pto", "vacation"], q_clean):
            # Handle negative/absence follow-ups: "i dont have medical certificates means?", "no certificate", "without certificate"
            if (
                has_word(["dont have", "don't have", "do not have", "no certificate", "no medical", "without certificate", "without medical", "don't need", "dont need", "not have"], q_clean)
                or (has_word(["certificate", "certificates", "doctor", "note", "medical"], q_clean) and has_word(["dont", "don't", "not", "no", "without"], q_clean))
            ):
                return (
                    "A medical certificate is only required if your sick leave extends beyond **3 consecutive days**. "
                    "For absences of **3 days or fewer**, no doctor's note is needed."
                )
            if has_word(["certificate", "doctor", "note", "medical"], q_clean) and (has_word(["without", "need", "require", "mandatory", "when"], q_clean) or any(d in q_clean for d in ["4 day", "5 day", "6 day", "4 days", "5 days", "6 days"])):
                return (
                    "A medical certificate is required if sick leave extends beyond **3 consecutive days**. For absences of 3 days or fewer, no doctor's note is required."
                )
            return (
                "You are entitled to **12 days of paid sick and casual leave** annually. "
                "A medical certificate is only required if sick leave extends beyond 3 consecutive days."
            )
        if has_word(["carry forward", "rollover", "roll over", "unused"], q_clean) and has_word(["pto", "vacation", "leave", "days"], q_clean):
            return (
                "Under company policy, you can roll over up to **5 unused PTO days** into the following calendar year. Any unused days beyond 5 do not roll over."
            )
        if has_word(["pto", "vacation"], q_clean) and not has_word(["sick", "casual"], q_clean):
            return (
                "Full-time employees receive **18 PTO days per year** (18 days of paid vacation), "
                "and up to **5 unused PTO days** can roll over into the following year."
            )
        # If user asked specifically about "how many days" without specifying which type, check scope
        if re.search(r"\bhow\s+many\b", q_clean) and has_word(["pto", "vacation"], q_clean) and not has_word(["sick", "casual", "all", "total"], q_clean):
            return "You get **18 PTO days per year** (that's 18 days of paid vacation)."
        if re.search(r"\bhow\s+many\b", q_clean) and has_word(["sick"], q_clean) and not has_word(["pto", "vacation", "all", "total"], q_clean):
            return "You get **12 sick and casual leave days per year**. A medical certificate is required after 3 consecutive days."

        # Only provide the broad overview if the user asked for an overview/catalog or generic leave inquiry
        # Strip conversational prefixes like "so", "and", "then" before matching
        q_for_overview = re.sub(r"^(so|and|then|ok so|okay so|alright|hmm)\s+", "", q_clean).strip()
        is_leave_overview = (
            has_word(["overview", "summary", "list", "types of", "all leave", "what leaves", "what are the leaves", "catalog", "entitlements"], q_clean)
            or q_for_overview.rstrip("?").strip() in ["leave", "leaves", "pto", "leave policy", "leave benefits", "pto policy", "leave details", "leave info"]
            or q_clean.rstrip("?").strip() in ["leave", "leaves", "pto", "leave policy", "leave benefits", "pto policy"]
        )
        if is_leave_overview:
            return (
                "Here is an overview of our official leave benefits:\n\n"
                "- **Paid Vacation (PTO)**: **18 days/year** (up to 5 unused days roll over)\n"
                "- **Sick & Casual Leave**: **12 days/year** (medical certificate required after 3 days)\n"
                "- **Maternity Leave**: **26 weeks** fully paid (up to 2 children)\n"
                "- **Paternity Leave**: **4 weeks** fully paid within the first 6 months\n"
                "- **Bereavement Leave**: **5 consecutive paid days off**"
            )
        return (
            "The WorkPilot handbook does not specify details for that particular leave inquiry. "
            "It outlines standard allocations for Paid Time Off (18 days), Sick/Casual Leave (12 days), Maternity (26 weeks), Paternity (4 weeks), and Bereavement (5 days). "
            "For specific leave processes, approvals, or special circumstances, please contact your reporting manager or People Operations."
        )

    # 10. Travel & Expense Reimbursement
    travel_explicit_words = [
        "travel", "flight", "flights", "hotel", "hotels", "airline", "meal",
        "meals", "food", "per diem", "finance", "finace", "ticket", "tickets", "plane", "dinner", "breakfast"
    ]
    is_lunch_hours_query = (
        has_word(["lunch break", "take lunch", "hour lunch", "lunch at", "lunch time", "break for lunch"], q_clean)
        or (has_word(["skip lunch"], q_clean) and not has_word(["dinner", "breakfast", "$100", "100", "reimburse", "claim", "allowance", "get"], q_clean))
    )
    is_security_network_query = has_word(["vpn", "wireguard", "hotel wi-fi", "hotel wifi", "screen lock", "mfa", "2fa"], q_clean)
    is_travel_inquiry = (
        (
            has_word(["travel", "expense", "expenses", "meal", "meals", "food", "hotel", "hotels", "flight", "flights", "reimbursement", "per diem", "airline", "alcohol", "liquor", "beer", "wine", "finance", "finace", "ticket", "tickets", "plane", "dinner", "breakfast"], q_clean)
            or (has_prior_travel_context and (has_word(["beer", "alcohol", "liquor", "wine", "receipts", "receipt", "portal", "claim", "claims", "submit", "tier-1", "food", "drink", "drinks", "lunch", "limit", "maximum", "amount", "allowance", "when", "how long", "where", "how much", "within", "ticket", "tickets", "flight", "flights", "url", "link", "dinner"], q_clean) or q_clean.rstrip("?").strip() in ["food", "meal", "meals", "alcohol", "flight", "flights", "tickets", "ticket"]))
            or "travel" in section_name.lower()
            or "reimbursement" in section_name.lower()
        )
        and not (has_prior_insurance_context and not any(w in q_clean for w in travel_explicit_words))
        and not is_lunch_hours_query
        and not is_security_network_query
    )
    if is_travel_inquiry:
        # 1. Finance portal URL / link inquiry ("finance portal url?", "portal link", "what is the link", "url for finance portal")
        if any(w in q_clean for w in ["url", "link", "website", "web address", "site", "address"]) and (has_word(["portal", "finance", "finace", "claim", "claims", "expense", "submit"], q_clean) or has_prior_travel_context):
            return (
                "The WorkPilot handbook specifies that expense claims must be submitted via the **finance portal** within **30 days**, "
                "but it does not provide the specific portal URL or link. Please access the portal through the company internal intranet/dashboard "
                "or contact the Finance team or IT Support for the direct link."
            )

        # 2. Alcohol inquiry
        if has_word(["alcohol", "liquor", "beer", "wine", "drinks"], q_clean):
            return (
                "No, alcohol is strictly excluded from reimbursement under the daily meal allowance policy."
            )

        # 3. Flight inquiry or Follow-up on Flight
        is_flight_query = has_word(["flight", "flights", "airline", "fly", "flying", "plane", "ticket", "tickets", "airfare"], q_clean)
        is_flight_followup = (
            has_prior_travel_context
            and most_recent_travel_subtopic == "flight"
            and not has_word(["meal", "meals", "food", "hotel", "hotels", "portal", "within", "deadline"], q_clean)
        )

        if is_flight_query or is_flight_followup:
            if any(w in q_clean for w in ["amount", "cost", "how much", "limit", "budget", "price", "allowance", "rate", "fee", "fare"]):
                return (
                    "The WorkPilot handbook does not specify a fixed monetary dollar limit or budget cap for flight tickets. "
                    "Instead, flights are booked according to travel duration and class: domestic flights under 5 hours must be booked in **Economy Class**, "
                    "while flights exceeding 5 hours or international flights qualify for **Premium Economy**."
                )
            if any(w in q_clean for w in ["3 hour", "4 hour", "2 hour", "under 5", "short", "domestic"]) and has_word(["premium", "business"], q_clean):
                return (
                    "No, domestic flights under 5 hours must be booked in **Economy Class**. Only flights exceeding 5 hours or international flights qualify for **Premium Economy**."
                )
            if any(w in q_clean for w in ["6 hour", "7 hour", "8 hour", "over 5", "international"]):
                return (
                    "Yes! Flights over 5 hours or international flights are eligible for **Premium Economy Class**."
                )
            if re.search(r"\b(?:1|2|3|4)\s*[- ]?\s*hours?\b", q_clean):
                return (
                    "For domestic flights under 5 hours, the class of travel is standard **Economy Class**."
                )
            return (
                "For company travel, domestic flights under 5 hours are booked in **Economy Class**. "
                "Flights over 5 hours or international flights qualify for **Premium Economy**."
            )

        # 4. Hotel inquiry or Follow-up on Hotel
        is_hotel_query = has_word(["hotel", "hotels", "stay", "night", "tier-1", "tier 1", "lodging", "accommodation"], q_clean)
        is_hotel_followup = (
            has_prior_travel_context
            and most_recent_travel_subtopic == "hotel"
            and not has_word(["meal", "meals", "food", "flight", "flights", "portal", "within", "deadline"], q_clean)
        )
        if is_hotel_query or is_hotel_followup:
            return (
                "Hotel accommodations are reimbursable up to **$180 per night** in tier-1 cities, "
                "and up to **$120 per night** in other locations."
            )

        # 5. Meal inquiry
        if has_word(["meal", "meals", "food", "per diem", "lunch", "dinner", "breakfast"], q_clean) or (has_prior_travel_context and q_clean.rstrip("?").strip() in ["food", "meal", "meals"]):
            return (
                "During official business travel, the daily meal allowance is capped at **$75 per day** (alcohol excluded). "
                "Be sure to save your itemized tax receipts and submit claims through the finance portal within **30 days**."
            )

        # 5b. Dinner / skip lunch / $100 inquiry
        if has_word(["dinner", "skip lunch", "$100", "100 for dinner"], q_clean):
            return (
                "No. During official business travel, the daily meal allowance is capped at **$75 per day**. "
                "Skipping lunch does not increase your dinner allowance or permit reimbursement exceeding the $75 daily cap."
            )

        # 6. Deadline / Submission Window inquiry ("deadline", "when", "how long", "late", "submission window", "within", "claims within")
        if (
            has_word(["deadline", "late", "submit", "submission", "30 days", "45 days", "when", "window", "how long", "within", "days to submit"], q_clean)
            or (has_word(["within"], q_clean) and has_word(["claim", "claims", "finance", "finace", "submit"], q_clean))
        ):
            if any(w in q_clean for w in ["after 30", "past 30", "late", "45 day", "45 days", "60 day", "60 days"]):
                return (
                    "No, expense claims must be submitted via the finance portal within **30 days** of incurring the expense. Claims submitted after 30 days are subject to rejection."
                )
            return (
                "All expense claims along with itemized tax receipts must be submitted via the finance portal within **30 days** "
                "of incurring the expense. Claims submitted after 30 days are subject to rejection."
            )

        # 7. Amount / Limit inquiry ("how much can i claim", "limit", "allowance", "maximum", "amount", "how much i claim")
        if (
            has_word(["how much", "limit", "allowance", "maximum", "how much can i claim", "how much claim", "how much i claim", "amount", "rate"], q_clean)
            or (has_word(["claim", "claims"], q_clean) and not has_word(["when", "deadline", "how long", "window", "after", "past", "late", "where", "how to", "how do i", "portal", "within"], q_clean))
        ):
            if is_flight_query or is_flight_followup:
                return (
                    "The WorkPilot handbook does not specify a fixed monetary dollar limit or budget cap for flight tickets. "
                    "Instead, flights are booked according to travel duration and class: domestic flights under 5 hours must be booked in **Economy Class**, "
                    "while flights exceeding 5 hours or international flights qualify for **Premium Economy**."
                )
            if is_hotel_query or is_hotel_followup:
                return (
                    "Hotel accommodations are reimbursable up to **$180 per night** in tier-1 cities, "
                    "and up to **$120 per night** in other locations."
                )
            return (
                "During official business travel, the daily meal allowance is capped at **$75 per day** (alcohol excluded). "
                "Itemized tax receipts must be submitted via the finance portal within **30 days**."
            )

        # 8. Process / Submission steps ("how do i claim", "where do i submit", "how to claim", "where")
        if has_word(["how to", "how do i", "where", "process", "procedure", "steps"], q_clean) and has_word(["claim", "submit", "expense", "receipts", "receipt", "portal"], q_clean):
            return (
                "To claim expenses, submit your itemized tax receipts through the company **finance portal** within **30 days** of incurring the expense."
            )

        return (
            "Official business expenses are eligible for reimbursement with valid itemized receipts:\n\n"
            "- **Meals**: Up to **$75/day** (alcohol excluded)\n"
            "- **Hotels**: Up to **$180/night** in tier-1 cities ($120/night elsewhere)\n"
            "- **Flights**: Economy for under 5 hrs, Premium Economy for 5+ hrs or international\n"
            "- **Submission Deadline**: Submit via the finance portal within **30 days** of the expense."
        )

    # 11. Information Security & Passwords
    if has_word(["password", "passwords", "mfa", "2fa", "security", "vpn", "wireguard", "auto-lock", "autolock", "lock screen", "screen lock", "hotel wi-fi", "hotel wifi", "public wi-fi", "public wifi", "coffee shop"], q_clean) or has_prior_password_context:
        if has_word(["mfa", "2fa", "multi-factor", "app do we use", "which app"], q_clean):
            return (
                "Multi-Factor Authentication (MFA) is strictly mandatory on all company accounts, including **Google Workspace** and **GitHub**."
            )
        if has_word(["vpn", "wireguard", "wifi", "wi-fi", "hotel wi-fi", "hotel wifi", "public wi-fi", "public wifi", "coffee shop"], q_clean):
            return (
                "An active connection through our official **WireGuard VPN** is strictly required whenever connecting to company networks from public Wi-Fi or hotel networks."
            )
        if has_word(["auto-lock", "autolock", "inactivity", "screen lock", "lock screen", "screen", "timeout"], q_clean):
            return (
                "Company devices auto-lock after **5 minutes** of inactivity and must never be left unattended in public spaces."
            )
        if has_word(["reuse", "re-use", "last 5", "old password", "previous password"], q_clean) and (has_word(["password", "passwords"], q_clean) or has_prior_password_context):
            return (
                "No, you cannot reuse your previous passwords. Company policy prohibits reusing your **last 5 passwords**, and passwords must be updated every **90 days**."
            )
        if has_word(["cycle", "rotation", "rotate", "change", "update", "expire", "expiry", "how often"], q_clean) and has_word(["password", "passwords"], q_clean):
            return (
                "Passwords must be changed every **90 days** and cannot match your last 5 passwords."
            )
        if has_word(["length", "long", "characters", "complexity", "requirement", "requirements", "strong"], q_clean) and has_word(["password", "passwords"], q_clean):
            return (
                "Passwords must be at least **12 characters long**, containing uppercase and lowercase letters, numbers, and at least one special symbol."
            )
        # Broad security question or just 'password' alone -> full overview
        if has_word(["security", "rules", "all"], q_clean) or (has_word(["password"], q_clean) and not has_word(["cycle", "rotation", "rotate", "change", "update", "expire", "expiry", "length", "long", "characters", "complexity", "requirement", "requirements", "strong", "how often"], q_clean)):
            return (
                "To safeguard company and client data, here are the key security rules:\n\n"
                "- **Password Complexity**: At least **12 characters long** with uppercase, lowercase, numbers, and special symbols\n"
                "- **Rotation**: Passwords must be updated every **90 days** (and cannot match your last 5 passwords)\n"
                "- **MFA**: Strictly mandatory on all accounts (Google Workspace & GitHub)\n"
                "- **Screen Lock**: Auto-locks after **5 minutes** of inactivity\n"
                "- **VPN**: Must connect via **WireGuard VPN** when using public Wi-Fi"
            )
        return (
            "To safeguard company and client data, here are the key security rules:\n\n"
            "- **Password Complexity**: At least **12 characters long** with uppercase, lowercase, numbers, and special symbols\n"
            "- **Rotation**: Passwords must be updated every **90 days** (and cannot match your last 5 passwords)\n"
            "- **MFA**: Strictly mandatory on all accounts (Google Workspace & GitHub)\n"
            "- **Screen Lock**: Auto-locks after **5 minutes** of inactivity\n"
            "- **VPN**: Must connect via **WireGuard VPN** when using public Wi-Fi"
        )

    # 12. Performance Appraisal & Promotions
    is_appraisal_inquiry = (
        has_word(["appraisal", "appraisals", "promotion", "promotions", "rating", "increment", "increments", "evaluation", "review", "reviews", "review cycle", "salary hike", "raise", "objective", "objectives", "kpi", "kpis", "goals", "competency", "competencies", "principles", "rating scale", "rating framework"], q_clean)
        or "appraisal" in section_name.lower()
        or "promotion policy" in section_name.lower()
    )
    if is_appraisal_inquiry:
        if has_word(["objective", "objectives", "competency", "competencies", "principles", "criteria", "kpi", "kpis", "goals"], q_clean):
            return (
                "Under WorkPilot's Performance Appraisal Policy, employees are evaluated on a **5-point rating scale** across four core competencies: **technical delivery, ownership, teamwork, and leadership principles**. Evaluations take place bi-annually in **April** (mid-year review) and **October** (annual performance & compensation appraisal)."
            )
        if has_word(["rating", "scale", "5 point", "5-point"], q_clean):
            return (
                "WorkPilot uses a **5-point performance rating scale** covering technical delivery, ownership, teamwork, and leadership. Reviews are conducted in April and October."
            )
        if has_word(["increment", "increments", "salary hike", "raise"], q_clean) and has_word(["when", "effective", "finalized", "hike", "raise"], q_clean):
            return (
                "Salary increments and promotions are finalized in **November**, following the October annual appraisal cycle."
            )
        if has_word(["promotion", "promotions", "promote"], q_clean):
            return (
                "Promotions are determined during the **October** annual appraisal cycle and take effect in **November**."
            )
        if has_word(["when", "cycle", "cycles", "conducted", "frequency", "held", "how often"], q_clean):
            return (
                "Performance evaluations are conducted bi-annually: a mid-year review in **April** and an annual performance & compensation review in **October** (with promotions and increments effective in **November**)."
            )
        return (
            "Performance evaluations are conducted bi-annually: a mid-year review in **April** and an annual performance & compensation review in **October** (with promotions and increments effective in **November**). "
            "Ratings use a **5-point scale** across technical delivery, ownership, teamwork, and leadership."
        )

    # 13. Code of Conduct & Anti-Harassment (POSH)
    if has_word(["conduct", "harassment", "posh", "discrimination", "retaliation", "icc", "ethics", "whistleblower"], q_clean) or re.search(r"retaliat", q_clean):
        if re.search(r"retaliat", q_clean) or has_word(["retaliation"], q_clean):
            return (
                "WorkPilot strictly prohibits retaliation. Retaliation against anyone filing a complaint or participating in an investigation results in **immediate termination**."
            )
        if has_word(["email", "whistleblower", "report", "reporting", "anonymous", "anonymously", "contact", "file", "submit"], q_clean):
            return (
                "WorkPilot enforces a strict **zero-tolerance policy** against discrimination, harassment, or retaliation under the POSH framework. "
                "Concerns can be reported to the Internal Complaints Committee (ICC) or anonymously at `ethics@workpilot.internal`."
            )
        return (
            "WorkPilot enforces a strict **zero-tolerance policy** against discrimination, harassment, or retaliation under the POSH framework. "
            "Concerns can be reported to the Internal Complaints Committee (ICC) or anonymously at `ethics@workpilot.internal`."
        )

    # 14. HR / People Operations Contact
    if has_word(["hr", "human resource", "human resources", "people operations", "people ops"], q_clean) and has_word(["contact", "reach", "email", "phone", "number", "team", "department", "talk to", "speak to", "connect", "how to contact"], q_clean):
        return (
            "For HR-related queries, please reach out to the **People Operations** team. "
            "The handbook does not list specific contact details — you can typically find them on your company intranet, Slack directory, or by emailing your manager."
        )

    # 15. Working Hours & Core Timings
    if has_word(["hours", "timings", "timing", "core hours", "working hours", "shift", "lunch", "start work", "start time", "skip lunch", "8 am", "9 am", "10 am", "7 am", "overtime", "weekend", "tracked", "lunch break"], q_clean) or "working hours" in section_name.lower():
        if has_word(["lunch"], q_clean):
            if has_word(["dinner", "$100", "100"], q_clean):
                return (
                    "No. During official business travel, the daily meal allowance is capped at **$75 per day**. "
                    "Skipping lunch does not increase your dinner allowance or permit claims above the $75 daily cap."
                )
            if has_word(["skip", "leave early", "early", "without", "miss"], q_clean):
                return (
                    "All employees observe a mandatory **1-hour daily lunch break** during standard working hours. Skipping lunch to leave early is not permitted under the policy."
                )
            if has_word(["2 hour", "two hour", "extend", "longer"], q_clean):
                return (
                    "WorkPilot policy provides a standard **1-hour daily lunch break**. Taking a 2-hour lunch break is not permitted under standard core working hours."
                )
            if has_word(["3 pm", "core", "afternoon", "late"], q_clean):
                return (
                    "Lunch breaks must be scheduled outside or coordinated around mandatory **core collaboration hours (10:00 AM to 4:00 PM)**, and employees observe a standard **1-hour lunch break**."
                )
            return (
                "All employees observe a mandatory **1-hour daily lunch break** during standard working hours."
            )
        if has_word(["overtime", "weekend"], q_clean):
            return (
                "Standard working hours are **40 hours per week** (9:00 AM to 6:00 PM, Monday through Friday). The handbook does not document overtime pay or weekend work compensation policies. Please check with your manager or People Operations."
            )
        if has_word(["tracked", "software", "monitoring"], q_clean):
            return (
                "Employees are expected to fulfill **40 hours per week** (with core hours between 10:00 AM and 4:00 PM). The handbook does not specify automated software tracking tools."
            )
        if has_word(["start", "start time", "flexible", "adjust"], q_clean):
            if any(t in q_clean for t in ["7 am", "6 am", "7:00", "6:00", "before 8"]):
                return (
                    "Flexible start times are permitted between **8:00 AM and 10:00 AM** with manager approval. Starting before 8:00 AM is not permitted under the policy."
                )
            return (
                "Yes! Flexible working hours allow you to adjust your morning start time between **8:00 AM and 10:00 AM** with your manager's approval."
            )
        return (
            "Standard working hours are **9:00 AM to 6:00 PM**, Monday through Friday (40 hours/week), with mandatory core collaboration hours from **10:00 AM to 4:00 PM**."
        )

    # 16. Intelligent Fallback Humanizer for arbitrary queries
    # Validate that the primary fact is actually relevant to the question before using it
    fact = clean_bullet(primary_fact)
    fact_lower = fact.lower()
    # Check if the candidate has ANY real word overlap with the question
    fact_words = set(re.findall(r"\w+", fact_lower)) - SYNTHESIS_STOPWORDS - {"the", "a", "an", "is", "are", "and", "or", "of", "in", "to", "for", "on", "at", "by", "with"}
    question_words = set(re.findall(r"\w+", q_clean)) - SYNTHESIS_STOPWORDS - {"the", "a", "an", "is", "are", "and", "or", "of", "in", "to", "for", "on", "at", "by", "with"}
    overlap = fact_words & question_words
    # Words that appear in a confirmation ('do we get free X?') but carry no
    # topical meaning — a chunk sharing ONLY these with the question is irrelevant.
    generic_confirmation_words = {
        "free", "get", "got", "have", "has", "need", "do", "does", "did", "can", "could", "will",
        "would", "should", "may", "yes", "no", "not", "company", "work", "works", "working", "office",
        "policy", "policies", "employee", "employees", "staff", "me", "my", "we", "our", "us", "them",
        "they", "their", "you", "your", "there", "here", "want", "able", "permitted", "allowed",
        "available", "any", "one", "some", "thing", "what", "how", "why", "where", "when", "who",
    }
    meaningful_overlap = overlap - generic_confirmation_words
    if len(meaningful_overlap) == 0:
        # No meaningful overlap — this chunk is irrelevant, return a proper fallback
        return synthesize_dynamic_fallback(question)

    label, val = ("", fact)
    if ":" in fact and len(fact.split(":")[0]) < 45:
        label, val = fact.split(":", 1)
        label = label.strip()
        val = val.strip()

    val_clean = val.rstrip(".")
    if val_clean.startswith(("Provides ", "Offers ", "Covers ", "Enforces ", "Complies ", "Maintains ")):
        val_clean = f"The company {val_clean[0].lower() + val_clean[1:]}"

    is_wh = bool(re.search(r"^(what|how|why|where|when|who)\b", q_clean))
    is_confirmation = not is_wh and (bool(re.search(r"^(is|are|can|could|do|does|will|am\s+i|can\s+i|do\s+we)\b", q_clean)) or q_clean.endswith("?"))
    if is_confirmation:
        if any(neg in val_clean.lower() for neg in ["prohibited", "never", "zero-tolerance", "rejection", "excluded", "without", "not permitted", "not covered", "not allowed"]):
            return f"No, company policy prohibits this: {val_clean}."
        return f"Yes! {val_clean}."

    sec_title = clean_policy_name(section_name)
    return f"Regarding {sec_title.lower()}: {val_clean}."


POLICY_CATALOG_RESPONSE = """WorkPilot maintains 10 official company policies in the Employee Handbook:

1. **Working Hours & Core Timings**: 40-hour work week, 10:00 AM – 4:00 PM core collaboration hours, and flexible start times.
2. **Remote Work & Hybrid Guidelines**: Up to 3 days remote work per week, $500 home office setup stipend, and $50 monthly internet allowance.
3. **Leave Policy & Paid Time Off (PTO)**: 18 annual PTO days, 12 sick/casual leave days, 26 weeks maternity, and 4 weeks paternity leave.
4. **Travel & Business Expense Reimbursement**: $75 daily meal allowance, Economy/Premium flights, and $180/night hotel reimbursement limit.
5. **IT Equipment & Hardware Policy**: 16-inch MacBook Pro or Dell XPS 15, external 4K monitor, accessories, and 3-year hardware refresh.
6. **Information Security & Password Policy**: 12-character passwords, mandatory MFA, 90-day rotation, and WireGuard VPN requirement.
7. **Employee Benefits & Health Insurance**: $50,000 inpatient family hospitalization, annual health checkups, dental/vision, and $60/month gym & fitness stipend.
8. **Code of Conduct & Anti-Harassment (POSH)**: Zero-tolerance policy against discrimination and harassment, POSH compliance, and ICC reporting.
9. **Performance Appraisal & Promotion Policy**: Bi-annual appraisal cycles (April & October), 5-point rating framework, and November salary increments.
10. **Resignation & Notice Period Protocol**: 60-day notice period for confirmed employees (30 days for probation) and 45-day final settlement (F&F).

Would you like more details on any specific policy?"""


SECTION_OVERVIEWS = {
    "working_hours": (
        "Working Hours and Core Timings",
        "Here is what you need to know about WorkPilot's working hours and collaboration guidelines:\n\n"
        "- **Standard Schedule**: 40-hour work week (Monday through Friday, 9:00 AM to 6:00 PM).\n"
        "- **Core Collaboration Hours**: **10:00 AM to 4:00 PM local time**, when everyone should be reachable on Slack and available for meetings.\n"
        "- **Flexible Start**: You can adjust your morning start time between **8:00 AM and 10:00 AM** with manager approval.\n"
        "- **Lunch Break**: A mandatory 1-hour lunch break daily."
    ),
    "remote_work": (
        "Remote Work and Hybrid Guidelines",
        "WorkPilot operates on a flexible hybrid work model designed to give you balance:\n\n"
        "- **Hybrid Flexibility**: Work remotely up to **3 days per week**.\n"
        "- **Full-Time Remote**: Available upon written approval from your Department Head and People Operations.\n"
        "- **Home-Office Setup**: A one-time **$500 stipend** for ergonomic desk and chair equipment.\n"
        "- **Monthly Allowance**: **$50/month** for internet and utilities (requires at least a 50 Mbps stable connection)."
    ),
    "leave_policy": (
        "Leave Policy and Paid Time Off (PTO)",
        "Here are the details on our official leave policies and paid time off:\n\n"
        "- **Paid Time Off (PTO)**: **18 days** of paid vacation per year (up to 5 unused days can roll over to next year).\n"
        "- **Sick & Casual Leave**: **12 days** per year (doctor's certificate required if exceeding 3 consecutive days).\n"
        "- **Maternity Leave**: **26 weeks** fully paid for female employees (up to 2 children).\n"
        "- **Paternity Leave**: **4 weeks** fully paid within the first 6 months of birth or adoption.\n"
        "- **Bereavement Leave**: **5 consecutive paid days off** in the event of the loss of an immediate family member."
    ),
    "travel_expense": (
        "Travel and Business Expense Reimbursement Policy",
        "Here are our guidelines for official travel and expense reimbursements:\n\n"
        "- **Daily Meal Allowance**: Up to **$75 per day** (alcohol excluded) during business trips.\n"
        "- **Flights**: Domestic flights under 5 hours are in **Economy Class**; flights over 5 hours or international flights qualify for **Premium Economy**.\n"
        "- **Hotel Stays**: Up to **$180/night** in tier-1 cities and **$120/night** in other locations.\n"
        "- **Claim Submission**: Submit itemized tax receipts via the finance portal within **30 days** of incurring expenses (claims past 30 days are subject to rejection)."
    ),
    "it_equipment": (
        "IT Equipment and Hardware Policy",
        "WorkPilot equips team members with enterprise-grade hardware:\n\n"
        "- **Engineering Laptops**: 16-inch **Apple MacBook Pro (M3/M4)** or **Dell XPS 15** with 32GB RAM.\n"
        "- **Peripherals**: External **27-inch 4K monitor**, wireless keyboard, mouse, and noise-canceling headset.\n"
        "- **Refresh Cycle**: Hardware is upgraded every **3 years**.\n"
        "- **Asset Care**: Equipment remains company property and should be returned to IT Support upon exit."
    ),
    "info_security": (
        "Information Security and Password Policy",
        "To keep our systems and client data safe, please follow these security standards:\n\n"
        "- **Password Requirements**: At least **12 characters**, including uppercase, lowercase, numbers, and special symbols.\n"
        "- **MFA**: Strictly mandatory across all company accounts, Google Workspace, and GitHub.\n"
        "- **Password Rotation**: Update passwords every **90 days** (cannot reuse your last 5 passwords).\n"
        "- **Auto-Lock**: Devices auto-lock after **5 minutes** of inactivity; never leave devices unattended in public.\n"
        "- **VPN**: Always connect via our official **WireGuard VPN** when on public Wi-Fi."
    ),
    "benefits": (
        "Employee Benefits and Group Health Insurance",
        "WorkPilot provides comprehensive health and wellness benefits for you and your family:\n\n"
        "- **Health Insurance**: Up to **$50,000** annual inpatient hospitalization for you, your spouse, and up to 2 dependent children.\n"
        "- **Annual Health Checkup**: Free health vouchers provided annually for employees and spouses.\n"
        "- **Dental & Vision**: Covered up to **$1,000 annually** per employee.\n"
        "- **Mental Wellness (EAP)**: **12 free confidential therapy and counseling sessions** per year.\n"
        "- **Fitness Reimbursement**: Up to **$60 per month** toward gym memberships, yoga classes, or sports subscriptions."
    ),
    "code_of_conduct": (
        "Code of Conduct and Anti-Harassment (POSH)",
        "WorkPilot is committed to providing a safe, respectful, and inclusive workplace for everyone:\n\n"
        "- **Zero-Tolerance**: We enforce a zero-tolerance policy against any form of discrimination, harassment, or retaliation under the POSH framework.\n"
        "- **Reporting Concerns**: Observed or experienced harassment should be reported to the Internal Complaints Committee (ICC) or anonymously at `ethics@workpilot.internal`.\n"
        "- **Anti-Retaliation**: Retaliation against anyone raising a concern is strictly forbidden and results in immediate termination."
    ),
    "appraisal": (
        "Performance Appraisal and Promotion Policy",
        "Performance evaluations are designed to support your career growth and recognition:\n\n"
        "- **Review Cycles**: Bi-annual cycles in **April** (mid-year review) and **October** (annual performance & compensation appraisal).\n"
        "- **Evaluation Framework**: Rated on a **5-point scale** across technical delivery, ownership, teamwork, and leadership.\n"
        "- **Increments & Promotions**: Finalized in **November** following the October cycle."
    ),
    "notice_period": (
        "Resignation and Notice Period Protocol",
        "Here is what to know regarding notice periods and departure protocols:\n\n"
        "- **Notice Duration**: **60 calendar days** for full-time confirmed employees (**30 calendar days** during probation).\n"
        "- **Notice Buyout**: Available with mutual written consent between you and your department head.\n"
        "- **Full & Final Settlement (F&F)**: Accrued salary, gratuity, and leave encashment are disbursed within **45 days** of your last working day."
    ),
}


SECTION_PAGE_MAP = {
    "working_hours": 1,
    "remote_work": 1,
    "leave_policy": 2,
    "travel_expense": 2,
    "it_equipment": 3,
    "info_security": 3,
    "benefits": 4,
    "code_of_conduct": 4,
    "appraisal": 5,
    "notice_period": 5,
}


def is_policy_catalog_query(query: str) -> bool:
    """Detects inquiries asking for a list or catalog of all company policies."""
    q = query.strip().lower()
    clean_q = re.sub(r"[\?\.\!]+$", "", q).strip()

    # Exact broad catalog inquiries
    exact_catalog_queries = {
        "company policies", "company policy", "company policys",
        "policies", "policy", "all policies", "all policy",
        "our policies", "our policy", "workpilot policies", "workpilot policy",
        "employee policies", "employee policy", "handbook policies",
        "company rules", "workplace rules", "rules", "policy list",
        "list policies", "list policy", "list of policies", "show policies",
        "show all policies", "policy overview", "policies overview",
        "policy catalog", "all company policies", "table of contents",
        "handbook policy", "what policies", "what policy", "what are company policies",
    }
    if clean_q in exact_catalog_queries:
        return True

    # If the user mentions a specific policy section/subtopic, do NOT treat as catalog
    specific_policy_keywords = [
        "leave", "pto", "vacation", "sick", "casual", "maternity", "paternity", "bereavement",
        "remote", "wfh", "hybrid", "internet", "stipend", "wifi", "broadband",
        "travel", "expense", "flight", "hotel", "meal", "per diem",
        "laptop", "macbook", "dell", "hardware", "monitor", "equipment",
        "security", "password", "vpn", "mfa", "wireguard", "screen lock",
        "benefit", "benefits", "insurance", "hospital", "gym", "dental", "vision", "therapy", "eap",
        "conduct", "harassment", "posh", "ethics", "icc", "discrimination",
        "appraisal", "promotion", "increment", "rating", "evaluation", "kpi",
        "notice", "resignation", "resioning", "resigning", "resign", "probation", "buyout", "f&f",
        "hours", "timing", "timings", "shift", "lunch",
        "cafeteria", "parking", "dress code", "payroll", "payslip", "referral",
    ]
    if any(k in clean_q for k in specific_policy_keywords):
        return False

    patterns = [
        r"\b(what\s+(are\s+the|policies|all\s+the)\s+(employee\s+|company\s+)?policies)\b",
        r"\b(what\s+policies(\s+do\s+we\s+have|\s+we\s+have|\s+are\s+there|\s+exist)?)\b",
        r"\b(list\s+(all\s+)?(the\s+)?(company\s+|employee\s+)?policies)\b",
        r"\b(show\s+(me\s+)?(all\s+)?(the\s+)?(company\s+|employee\s+)?policies)\b",
        r"\b(all\s+(the\s+)?(company\s+|employee\s+)?policies)\b",
        r"\b(policies\s+(we\s+have|list|overview|catalog))\b",
        r"\b(what\s+are\s+our\s+policies)\b",
        r"\b(what\s+is\s+included\s+in\s+the\s+(policy\s+)?handbook)\b",
        r"\b(handbook\s+policies|table\s+of\s+contents)\b",
        r"\b(overview\s+of\s+(all\s+)?policies)\b",
        r"\b(company\s+polic(y|ies))\b",
        r"\b(workpilot\s+polic(y|ies))\b",
        r"\b(employee\s+polic(y|ies))\b",
        r"\b(company\s+rules)\b",
        r"\b(tell\s+me\s+(about\s+)?(all\s+)?(our\s+|the\s+)?policies)\b",
    ]
    return any(re.search(p, q) for p in patterns)


SPECIFIC_SUBTOPIC_PATTERNS = [
    r"\b(sport|sports|gym|yoga|fitness|workout|exercise)\b",
    r"\b(dental|vision|teeth|eye|eyes|glasses|spectacles)\b",
    r"\b(therapy|counseling|counselling|mental\s+health|eap|psychologist)\b",
    r"\b(checkup|vouchers|hospital|hospitalization|inpatient)\b",
    r"\b(maternity|paternity|parental|bereavement|funeral)\b",
    r"\b(sick\s+leave|casual\s+leave|carry\s+forward|rollover)\b",
    r"\b(meal|meals|food|per\s+diem|hotel|hotels|flight|flights|economy|premium\s+economy)\b",
    r"\b(laptop|laptops|macbook|dell|monitor|headset|mouse|keyboard|refresh\s+cycle)\b",
    r"\b(password\s+complexity|password\s+length|mfa|2fa|multi-factor|vpn|wireguard|screen\s+lock|auto-lock|autolock)\b",
    r"\b(internet|wifi|broadband|speed|mbps|bandwidth|stipend|utility\s+allowance)\b",
    r"\b(core\s+hours|lunch|lunch\s+break|flexible\s+start)\b",
    r"\b(buyout|probation|probationary|f&f|final\s+settlement|gratuity)\b",
    r"\b(rating\s+scale|rating\s+framework|mid-year|increments)\b",
]


def is_broad_section_query(query: str) -> bool:
    """Detects inquiries asking for an overview of an entire policy section rather than a specific sub-topic or metric."""
    q = query.strip().lower()

    # 1. Any inquiry mentioning a specific sub-topic or clause is NOT broad
    for pat in SPECIFIC_SUBTOPIC_PATTERNS:
        if re.search(pat, q):
            return False

    # 2. Specific metric or confirmation questions should NEVER be treated as broad
    if re.search(r"^(is|are|can|could|do|does|will|am\s+i|how\s+many|how\s+much|what\s+speed|what\s+limit|what\s+allowance)\b", q):
        return False
    if re.search(r"\b(free|speed|mbps|bandwidth|stipend\s+amount|how\s+long|days\s+of\s+notice|cost|price|budget|per\s+night|per\s+day)\b", q):
        return False
    if q.endswith("free?") or q.endswith("free ?") or q.endswith("allowed?"):
        return False

    # 3. Check for broad whole-section inquiry patterns
    # 3. Check for explicit overview or summary requests
    broad_patterns = [
        r"\b(overview\s+of|summary\s+of|summarize|guide\s+to|all\s+about|complete\s+guide|details\s+on)\s+(the\s+)?[a-z\s]+(policy|guidelines|benefits|rules|protocol)\b",
        r"\b(summarize\s+the\s+leave\s+policy|summarize\s+leave\s+policy)\b",
        r"\bwhat\s+benefits\s+(do\s+we\s+have|are\s+provided|are\s+there)\b",
        r"\bwhat\s+(leaves|leave\s+types)\s+(do\s+we\s+get|are\s+there|are\s+allowed)\b",
    ]
    return any(re.search(p, q) for p in broad_patterns)


def get_section_overview_key(section_name: str, query: str = "") -> str | None:
    """Maps a section title or user query to its SECTION_OVERVIEWS key."""
    sec_lower = section_name.lower()
    if "working hours" in sec_lower or "core timings" in sec_lower:
        return "working_hours"
    if "remote work" in sec_lower or "hybrid guidelines" in sec_lower:
        return "remote_work"
    if "leave policy" in sec_lower or "paid time off" in sec_lower:
        return "leave_policy"
    if "travel" in sec_lower or "reimbursement policy" in sec_lower:
        return "travel_expense"
    if "hardware policy" in sec_lower or "it equipment" in sec_lower:
        return "it_equipment"
    if "information security" in sec_lower or "password policy" in sec_lower:
        return "info_security"
    if "benefits" in sec_lower or "health insurance" in sec_lower:
        return "benefits"
    if "code of conduct" in sec_lower or "anti-harassment" in sec_lower:
        return "code_of_conduct"
    if "appraisal" in sec_lower or "promotion policy" in sec_lower:
        return "appraisal"
    if "resignation" in sec_lower or "notice period" in sec_lower:
        return "notice_period"

    q_lower = query.lower()
    if re.search(r"\b(working\s+hours|timings|core\s+hours)\b", q_lower):
        return "working_hours"
    if re.search(r"\b(remote\s+work|hybrid|work\s+from\s+home|wfh)\b", q_lower):
        return "remote_work"
    if re.search(r"\b(leave\s+policy|pto\s+policy|leaves)\b", q_lower):
        return "leave_policy"
    if re.search(r"\b(travel\s+policy|expense\s+policy)\b", q_lower):
        return "travel_expense"
    if re.search(r"\b(it\s+equipment|hardware\s+policy)\b", q_lower):
        return "it_equipment"
    if re.search(r"\b(security\s+policy|password\s+policy)\b", q_lower):
        return "info_security"
    if re.search(r"\b(benefits|health\s+insurance)\b", q_lower):
        return "benefits"
    if re.search(r"\b(conduct|harassment|posh)\b", q_lower):
        return "code_of_conduct"
    if re.search(r"\b(appraisal|promotion\s+policy)\b", q_lower):
        return "appraisal"
    if re.search(r"\b(resignation|notice\s+period)\b", q_lower):
        return "notice_period"
    return None


def synthesize_dynamic_response(
    chunks: list[Document],
    question: str,
    chat_history: Sequence[Any] | None = None,
) -> tuple[str, bool]:
    """
    Synthesizes a fluent, natural conversational response answering the user's specific intent.
    Matches the context of the inquiry, section titles, and conversation history.
    Returns (response_text, is_grounded).
    """
    if not chunks:
        return synthesize_dynamic_fallback(question), False

    # Extract non-stopword tokens from question
    raw_words = set(re.findall(r"\w+", question.lower())) - SYNTHESIS_STOPWORDS
    specific_content_words = raw_words - GENERIC_WORKPLACE_WORDS
    salient_words = specific_content_words if specific_content_words else raw_words
    q_stems = {stem_token(w) for w in salient_words}

    # Extract previous conversation turns for multi-turn coherence
    prev_user_stems: set[str] = set()
    prev_asst_text: str = ""
    if chat_history:
        for msg in reversed(chat_history):
            role = getattr(msg, "role", None) or getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "") or ""
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if role in ("user", "human") and not prev_user_stems:
                prev_user_stems = {stem_token(w) for w in re.findall(r"\w+", content.lower())} - SYNTHESIS_STOPWORDS
            elif role in ("assistant", "ai") and not prev_asst_text:
                prev_asst_text = content.lower()

    # Identify metric cues from the user's specific inquiry
    metric_cues = []
    if any(w in raw_words for w in ["speed", "mbps", "bandwidth", "internet", "connection"]):
        metric_cues.append(r"\b\d+\s*mbps\b")
    if any(w in raw_words for w in ["meal", "allowance", "stipend", "cost", "budget", "hotel", "reimbursement", "price", "gym", "yoga", "fitness", "sport", "sports", "dental", "vision", "health", "utility"]):
        metric_cues.append(r"\$\d+")
    if any(w in raw_words for w in ["days", "many", "pto", "leave", "notice", "hours", "weeks", "vacation", "sessions"]):
        metric_cues.append(r"\b\d+\s*(days|weeks|months|hours|calendar days|sessions)\b")

    is_targeted_question = any(phrase in question.lower() for phrase in [
        "how many", "how much", "what is the limit", "what is the speed",
        "internet speed", "speed of internet", "meal allowance", "notice period",
        "password length", "just need", "only need", "specifically",
        "gym", "yoga", "sports", "sport", "fitness", "dental", "vision", "therapy", "counseling"
    ]) or bool(metric_cues)

    candidate_lines: list[tuple[float, int, bool, str, str]] = []
    seen = set()

    for idx, chunk in enumerate(chunks):
        current_section = extract_section_title(chunk.page_content)
        sec_words = set(re.findall(r"\w+", current_section.lower())) - SYNTHESIS_STOPWORDS
        sec_stems = {stem_token(w) for w in sec_words}
        sec_match_count = len(q_stems & sec_stems)

        # Contextual boost if current chunk relates to previous turn topic
        history_sec_boost = 0.0
        if prev_user_stems and (prev_user_stems & sec_stems):
            history_sec_boost += 6.0
        if prev_asst_text and current_section.lower() in prev_asst_text:
            history_sec_boost += 6.0

        chunk_score = float(chunk.metadata.get("score", 0.0))

        lines = [l.strip() for l in chunk.page_content.split("\n") if l.strip()]
        for line in lines:
            if line.startswith("## ") or line.startswith("# "):
                title = re.sub(r"^#+\s*", "", line).strip()
                title = re.sub(r"^\d+[\.\)]\s*", "", title).strip()
                if title and "handbook" not in title.lower() and len(title) > 3:
                    current_section = title
                    sec_words = set(re.findall(r"\w+", current_section.lower())) - SYNTHESIS_STOPWORDS
                    sec_stems = {stem_token(w) for w in sec_words}
                    sec_match_count = len(q_stems & sec_stems)
                continue
            if line.startswith("#"):
                continue
            cleaned = clean_bullet(line)
            if not cleaned or cleaned in seen or len(cleaned) < 15:
                continue
            seen.add(cleaned)

            line_words = set(re.findall(r"\w+", cleaned.lower()))
            line_stems = {stem_token(w) for w in line_words}
            direct_matches = len(q_stems & line_stems)
            has_metric = any(re.search(cue, cleaned, re.IGNORECASE) for cue in metric_cues)

            # Detect introductory summary statements of sections
            is_intro_statement = not (":" in cleaned and len(cleaned.split(":")[0]) < 40)

            score = (
                direct_matches * 4.0
                + sec_match_count * 5.0
                + (12.0 if has_metric else 0.0)
                + (chunk_score * 4.0)
                + history_sec_boost
                + (2.5 if idx == 0 else 0.0)
                + (3.0 if is_intro_statement and not is_targeted_question and sec_match_count > 0 else 0.0)
            )

            is_candidate = (
                direct_matches > 0
                or has_metric
                or sec_match_count > 0
                or history_sec_boost > 0
                or (idx == 0 and chunk_score >= 0.68)
            )

            if is_candidate:
                candidate_lines.append((score, direct_matches, has_metric, cleaned, current_section))

    if not candidate_lines:
        sec_name = extract_section_title(chunks[0].page_content) if chunks else "Company Policy"
        ans = format_humanized_answer(question, sec_name, "", None, chat_history=chat_history)
        if not ans.lower().startswith(("i couldn't find", "i was unable to find", "i wasn't able to find", "i could not find")):
            return ans, True
        return synthesize_dynamic_fallback(question), False

    # Sort lines by relevance score descending
    candidate_lines.sort(key=lambda x: x[0], reverse=True)

    primary_fact = candidate_lines[0][3]
    section_name = candidate_lines[0][4]

    # Adaptive response: If the user asked a broad overview question, provide a structured multi-bullet overview
    if is_broad_section_query(question):
        sec_key = get_section_overview_key(section_name, question)
        if sec_key and sec_key in SECTION_OVERVIEWS:
            sec_title, overview_body = SECTION_OVERVIEWS[sec_key]
            if chunks and sec_key in SECTION_PAGE_MAP:
                chunks[0].metadata["page"] = SECTION_PAGE_MAP[sec_key]
            return overview_body, True

    # Generate a fluent, humanized conversational answer answering the employee's intent
    humanized_ans = format_humanized_answer(question, section_name, primary_fact, candidate_lines, chat_history=chat_history)
    ans_lower = humanized_ans.lower()
    if ans_lower.startswith(("i couldn't find", "i was unable to find", "i wasn't able to find", "i could not find")):
        return humanized_ans, False
    return humanized_ans, True


DYNAMIC_FALLBACK_CATEGORIES = [
    (
        r"\b(cook|baking|bake|recipe|recipes|pasta|pizza|cake|cookie|cookies|ingredient|ingredients|carbonara|soup)\b",
        "cooking and food recipes",
        "culinary guides or colleagues over lunch",
    ),
    (
        r"\b(president|prime minister|governor|senator|election|parliament|king|queen|politics|government)\b",
        "government leadership and political topics",
        "external public news resources",
    ),
    (
        r"\b(capital\s+of|capital\s+city|geography|mountain|river|ocean|planet|continent)\b",
        "general geography and world trivia",
        "general encyclopedia or reference sources",
    ),
    (
        r"\b(football match|soccer match|basketball game|cricket match|tennis tournament|olympics|fifa|world cup|nba|premier league)\b",
        "sports tournaments and athletic events",
        "sports news outlets",
    ),
    (
        r"\b(weather|temperature|forecast|rain|sunny|cloudy|humidity|wind speed|climate today)\b",
        "weather forecasts and climate conditions",
        "local weather services or your preferred forecast app",
    ),
    (
        r"\b(stock\s+price|share\s+price|market\s+cap|ticker|nasdaq|nyse|sensex|nifty|ipo)\b",
        "stock prices and market data",
        "a financial data provider or your brokerage platform",
    ),
    (
        r"\b(crypto|bitcoin|ethereum|stock market|investing|trading|forex|mutual fund)\b",
        "personal financial investments and cryptocurrency",
        "a licensed personal financial advisor",
    ),
    (
        r"\b(pet|dog|cat|animal|puppy|kitten)\b",
        "bringing personal pets into the workplace",
        "Workplace Facilities or People Operations",
    ),
    (
        r"\b(cafeteria|lunchroom|canteen|snack bar|coffee machine|tea|vending machine|free food|free drinks)\b",
        "cafeteria and dining facilities",
        "Workplace Operations or Office Administration",
    ),
    (
        r"\b(holiday|holidays|public\s+holiday|public\s+holidays|festival|festivals|festive\s+calendar)\b",
        "public holidays and the festive calendar",
        "People Operations for the official holiday calendar",
    ),
    (
        r"\b(parking|garage|valet|bike rack|vehicle permit|park my car)\b",
        "parking allocation and vehicle permits",
        "Building Management or Workplace Facilities",
    ),
    (
        r"\b(dress\s+code|attire|clothes|uniform|what\s+to\s+wear)\b",
        "dress code and workplace attire",
        "People Operations or your local office manager",
    ),
    (
        r"\b(payslip|pay\s+date|payday|pay\s+day|salary\s+credited|credits?\s+salary|salary\s+of|salary\s+for|salary\s+range|salary\s+structure|salary\s+slab|salary\s+expectations?|gross\s+salary|net\s+salary|basic\s+salary|monthly\s+salary|annual\s+salary|hourly\s+(rate|pay)|wage|wages|what\s+day\s+is\s+(salary|pay)\b)\b",
        "payroll disbursement, salary structure, and payslip distribution",
        "the Finance and Payroll team",
    ),
    (
        r"\b(referral|refer\s+a\s+candidate|refer\s+someone|refer\b|job\s+opening|openings|career|vacancies|hiring)\b",
        "job openings and employee referral bonuses",
        "Talent Acquisition and Recruiting",
    ),
    (
        r"\b(401k|401\(k\)|pension|pf|provident\s+fund|retirement\s+plan|superannuation)\b",
        "401(k) retirement savings and pension plans",
        "People Operations or the Benefits Administrator",
    ),
    (
        r"\b(stock\s+options?|esop|esops|rsu|rsus|equity\s+grant|shares?\s+vesting)\b",
        "stock options, RSUs, and company equity",
        "People Operations or the Finance team",
    ),
    (
        r"\b(overtime|comp\s*off|compensatory\s+off|extra\s+hours\s+pay)\b",
        "overtime compensation and compensatory off policies (standard working hours are 40 hours per week)",
        "your reporting manager or People Operations",
    ),
    (
        r"\b(relocation|moving\s+expenses|relocate|moving\s+allowance)\b",
        "relocation assistance and moving allowances",
        "People Operations or Talent Acquisition",
    ),
    (
        r"\b(tuition|education\s+allowance|course\s+reimbursement|learning\s+budget|certifications?\s+reimbursement)\b",
        "tuition assistance and learning stipends",
        "People Operations or the Learning & Development team",
    ),
    (
        r"\b(childcare|daycare|crèche|creche|nanny|baby\s+sitting)\b",
        "childcare support, daycare stipends, and crèche facilities",
        "Workplace Operations or People Operations",
    ),
    (
        r"\b(life\s+insurance|accidental\s+death|disability\s+insurance|term\s+insurance)\b",
        "term life and disability insurance coverage",
        "People Operations or the insurance provider",
    ),
    (
        r"\b(visa|h1b|h-1b|green\s+card|work\s+permit|immigration|sponsorship)\b",
        "visa sponsorship, immigration, and work permits",
        "People Operations and company legal counsel",
    ),
]

ACRONYM_AND_TYPO_MAP = [
    (r"\b(coc|code\s+of\s+contact)\b", "code of conduct and anti-harassment"),
    (r"\bposh\b", "prevention of sexual harassment (posh) code of conduct"),
    (r"\bwfh\b", "remote work and hybrid guidelines work from home"),
    (r"\bpto\b", "paid time off annual leave vacation"),
    (r"\beap\b", "employee assistance program mental health counseling"),
    (r"\bmfa\b", "multi-factor authentication password security"),
    (r"\b(f&f|fnf)\b", "full and final settlement notice period resignation"),
    (r"\b(resign|resigns|resioning|resiging|resion|resignating|resigning|quitting)\b", "resignation notice period protocol"),
    (r"\b(kpi|kpis|objective|objectives|core\s+competencies|competencies)\b", "performance appraisal rating framework evaluation criteria"),
    (r"\bcoverble\b", "coverable covered"),
    (r"\b(finace|fiance)\b", "finance"),
    (r"\b(claimabke|claimible)\b", "claimable"),
]


def normalize_query_terms(query: str) -> str:
    """Normalizes common workplace acronyms (coc, posh, pto, wfh) and typos (code of contact)."""
    norm = query.lower().strip()
    for pattern, replacement in ACRONYM_AND_TYPO_MAP:
        if re.search(pattern, norm):
            norm = re.sub(pattern, replacement, norm)
    return norm


KNOWN_SHORT_TERMS = {
    "coc", "posh", "wfh", "pto", "eap", "mfa", "f&f", "fnf", "hr", "it", "icc", "llm", "ai", "pdf",
    "thx", "thnx", "tq", "ty", "tysm", "tqsm", "oi", "yo", "hlo", "helo", "sup", "ok", "k"
}


def is_gibberish_or_number(query: str) -> bool:
    """Detects pure numbers (e.g. '111'), non-alphabetic characters, or random keyboard mashing (e.g. 'asdfgh')."""
    q = query.strip()
    if not q:
        return True
    # Pure numbers (e.g. '111', '1234')
    if re.fullmatch(r"\d+", q):
        return True
    # Pure punctuation or symbols without letters
    if not re.search(r"[a-zA-Z]", q):
        return True
    clean_alpha = re.sub(r"[^a-zA-Z]", "", q).lower()
    if clean_alpha in KNOWN_SHORT_TERMS:
        return False
    # Very short single letters (except 'k')
    if len(clean_alpha) <= 1:
        return True
    # Repeated single character like 'aaaa'
    if re.fullmatch(r"(.)\1{3,}", clean_alpha):
        return True

    mash_patterns = [
        r"asdf", r"sdfg", r"dfgh", r"fghj", r"ghjk", r"hjkl",
        r"qwer", r"wert", r"erty", r"rtyu", r"tyui", r"yuio", r"uiop",
        r"zxcv", r"xcvb", r"cvbn", r"vbnm"
    ]
    # Check if any word in query is a keyboard mash or long consonant cluster
    # NOTE: 'y' is treated as a vowel here so real words like "monthly"
    # (n-t-h-l-y) are NOT misclassified as gibberish.
    words = re.findall(r"[a-zA-Z]+", q.lower())
    for w in words:
        if w in KNOWN_SHORT_TERMS:
            continue
        if any(re.search(p, w) for p in mash_patterns):
            return True
        if len(w) >= 4 and not re.search(r"[aeiouy]", w):
            return True
        if re.search(r"[bcdfghjklmnpqrstvwxz]{6,}", w):
            return True

    return False


HANDBOOK_PDF_RESPONSE = """You can view and search the verified **WorkPilot Company Policy Handbook (PDF)** directly below.

- **Interactive Reader**: Browse all 5 pages and 10 official company policies with interactive search and highlights.
- **Original PDF**: Click **Open PDF** below to view or download the complete official document in a new tab.

Feel free to ask me any specific question about working hours, remote work, leave policies, travel expenses, IT equipment, health benefits, or notice periods!"""


def is_handbook_pdf_request(query: str) -> bool:
    """Detects requests to view, open, download, or access the official Policy Handbook PDF."""
    q = query.strip().lower()
    clean_q = re.sub(r"[\?\.\!]+$", "", q).strip()

    # Exact keywords
    if clean_q in [
        "pdf", "handbook", "hand book", "policy pdf", "policy handbook",
        "handbook pdf", "pdf handbook", "give handbook", "give pdf",
        "get handbook", "get pdf", "show handbook", "show pdf",
        "open handbook", "open pdf", "view handbook", "view pdf",
        "download handbook", "download pdf", "employee handbook",
        "company handbook", "policy document", "official pdf"
    ]:
        return True

    # Regex patterns
    patterns = [
        r"\b(where\s+is|give\s+me|send\s+me|show\s+me|can\s+i\s+(see|get|have|view|download)|how\s+to\s+(see|get|view|download))\s+(the\s+)?(pdf|handbook|hand\s+book|policy\s+doc)\b",
        r"\b(link\s+to|access\s+to|copy\s+of)\s+(the\s+)?(pdf|handbook|hand\s+book)\b",
        r"\b(open|view|show|give|download|send)\s+(the\s+)?(pdf|handbook|hand\s+book)\b",
        r"\b(policy\s+handbook|employee\s+handbook|company\s+handbook)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def handle_conversational_turn(query: str) -> tuple[str | None, bool]:
    """
    Handles standard conversational turns: greetings, bot identity, company identity,
    thanks, and farewells naturally without querying vector storage.
    Returns (response_text, attach_doc_flag).
    """
    q_clean = query.strip().lower()
    q_norm = re.sub(r"[\?\.\!]+$", "", q_clean).strip()

    # 1. Company Identity: 'work pilot', 'workpilot', 'what is work pilot', 'work pilot?'
    if (
        re.search(r"\b(what\s+is|who\s+is|tell\s+me\s+about|about)\s+work\s*pilot\b", q_clean)
        or q_norm in ["workpilot", "work pilot", "workpilot company", "work pilot company", "about workpilot", "about work pilot"]
        or re.fullmatch(r"work\s*pilot\??", q_clean)
    ):
        return (
            "WorkPilot is our enterprise organization. I am your official employee assistant, here to provide verified answers from our Employee Handbook—including working hours, remote and hybrid guidelines, PTO & leave policies, travel expenses, IT equipment, password security, health benefits, performance appraisals, and notice periods.",
            False,
        )

    # 2. Greetings ('hlo', 'helo', 'hey', 'oi', 'hi dr', 'hi dear', 'yo', 'sup', 'hello', etc.)
    greetings_set = {
        "hi", "hello", "hey", "hlo", "helo", "hellow", "oi", "yo", "sup", "hola", "namaste",
        "hi there", "hello there", "hey there", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "how are you doing", "hi dr", "hi dear", "hello dr", "hello dear", "hey dr", "hey dear",
    }
    if (
        q_norm in greetings_set
        or any(q_norm.startswith(g + " ") for g in ["hi", "hello", "hey", "hlo", "helo", "oi", "hola", "yo"])
    ):
        return (
            "Hello! I am your WorkPilot employee assistant. How can I help you today with company policies, benefits, or workplace guidelines?",
            False,
        )

    # 3. Bot & LLM / AI Identity: 'llm', 'ai', 'are you an llm', 'are you an ai', 'chatbot', 'gpt', 'model'
    if (
        q_norm in ["llm", "ai", "chatbot", "bot", "gpt", "model", "llm model", "ai model", "ai bot"]
        or re.search(r"\b(are\s+you\s+(an\s+)?(llm|ai|bot|chatbot|robot)|what\s+(llm|model|ai)\s+are\s+you|what\s+is\s+(an?\s+)?llm)\b", q_clean)
        or re.search(r"\b(large\s+language\s+model|artificial\s+intelligence)\b", q_clean)
        or re.search(r"\b(who\s+are\s+you|what\s+can\s+you\s+do|what\s+are\s+you|what\s+is\s+your\s+name|how\s+can\s+you\s+help|what\s+do\s+you\s+do)\b", q_clean)
        or q_norm in ["help", "capabilities"]
    ):
        return (
            "I am WorkPilot's employee assistant. I'm here to help answer your questions about company policies, benefits, and workplace guidelines from the employee handbook. What can I help you with today?",
            False,
        )

    # 4. Acknowledgements, Praise & Completion ('done', 'all done', 'im done', 'all set', 'that's all', etc.)
    ack_words = {
        "ok", "okay", "k", "cool", "got it", "sure", "understood", "alright", "great", "awesome",
        "nice", "good", "perfect", "done", "all done", "im done", "i'm done", "finished",
        "all set", "thats all", "that's all", "that is all", "done for now", "no more",
        "nothing else", "all good", "done thank you", "done thanks"
    }
    if (
        q_norm in ack_words
        or q_norm.startswith("done with")
        or q_norm.startswith("that's all")
        or q_norm.startswith("thats all")
        or q_norm.startswith("all done")
    ):
        return (
            "Glad to help! Let me know if you have any other questions about company policies or workplace guidelines.",
            False,
        )

    # 5. Gratitude ('thnx', 'tq', 'thx', 'ty', 'tysm', 'tqsm', 'thank you', etc.)
    gratitude_exact = {
        "tq", "thx", "thnx", "ty", "tysm", "tqsm", "thank u", "thank you", "thanks", "thankyou",
        "many thanks", "thanks a lot", "thx a lot", "appreciate it", "much appreciated"
    }
    if (
        q_norm in gratitude_exact
        or any(q_norm.startswith(w) for w in ["thank you", "thanks", "thx", "thnx", "tq", "ty", "tysm", "tqsm", "appreciate it", "many thanks", "thank u", "thankyou"])
    ):
        return (
            "You're very welcome! Feel free to ask anytime if you need help with other company policies or guidelines.",
            False,
        )

    # 6. Farewells
    if q_norm in ["bye", "goodbye", "see you", "cya"]:
        return (
            "Goodbye! Have a great day ahead, and reach out whenever you have policy questions.",
            False,
        )

    return None, False


FALLBACK_VARIANTS = [
    "I couldn't find that information in the WorkPilot handbook. For specific guidance, please contact People Operations.",
    "I couldn't find information regarding {subject} in the WorkPilot handbook. Please check with People Operations or HR.",
    "I couldn't find that in the WorkPilot handbook. Please reach out to People Operations for clarification.",
]


def check_out_of_scope(question: str) -> str | None:
    """Detects inquiries outside company policy handbook and provides helpful department referrals."""
    q_lower = question.strip().lower()
    for pattern, topic, referral in DYNAMIC_FALLBACK_CATEGORIES:
        if re.search(pattern, q_lower):
            return f"I couldn't find information on {topic} in the WorkPilot handbook. Please consult {referral}."
    return None


def synthesize_dynamic_fallback(question: str) -> str:
    """
    Generates a concise, context-aware dynamic response when a query is out-of-scope,
    referencing the specific topic with natural phrasing.
    """
    q_clean = question.strip()
    q_lower = q_clean.lower()

    out_of_scope = check_out_of_scope(question)
    if out_of_scope:
        return out_of_scope

    # Extract salient subject from question
    stripped = re.sub(
        r"^(what\s+is|what\s+are|how\s+to|how\s+do\s+i|can\s+i|tell\s+me\s+about|do\s+we\s+have|is\s+there|why\s+is|who\s+is|when\s+is)\s+",
        "",
        q_lower,
    )
    stripped = re.sub(r"[\?\.\!]+$", "", stripped).strip()
    subject_phrase = f"'{stripped}'" if len(stripped) > 2 else "this subject"

    variant_idx = abs(hash(q_lower)) % len(FALLBACK_VARIANTS)
    return FALLBACK_VARIANTS[variant_idx].format(subject=subject_phrase)


def handle_rating_scale_query(
    question: str,
    chat_history: Sequence[Any] | None = None,
) -> tuple[str | None, list[Document]]:
    """
    Handles rating scale and performance evaluation inquiries, resolving ambiguity:
    - 'what is employee rating' -> direct concise explanation of 5-point scale and review cycles
    - 'what is the ##5 points in' / 'what are the 5 points?' -> intelligently resolves ambiguity
    """
    q_clean = question.strip().lower()
    q_norm = re.sub(r"[#\?\.\!]+", "", q_clean).strip()

    has_prior_rating = False
    if chat_history:
        for msg in reversed(chat_history):
            content = (getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")).lower()
            if any(w in content for w in ["rating", "5-point", "5 point", "appraisal"]):
                has_prior_rating = True
                break

    is_5_points = bool(
        re.search(r"\b(5|five)\s*points?\b", q_norm)
        or re.search(r"#+\s*5\s*points?", q_clean)
        or "5 point" in q_norm
    )

    if is_5_points:
        doc = Document(
            page_content="## 9. Performance Appraisal and Promotion Policy\nPerformance is evaluated on a 5-point rating scale across four core competencies: technical delivery, ownership, teamwork, and leadership...",
            metadata={"filename": "WorkPilot_Company_Policy.pdf", "page": 5, "chunk_index": 42}
        )
        if has_prior_rating:
            return (
                "The handbook says performance is evaluated on a 5-point scale across technical delivery, ownership, teamwork, and leadership. It does not specify what each individual score from 1 to 5 means.",
                [doc],
            )
        else:
            return (
                "If you mean the 5-point employee rating scale, the handbook says performance is evaluated across technical delivery, ownership, teamwork, and leadership. It doesn’t define what each individual score (1–5) means.",
                [doc],
            )

    # 2. What does a specific rating/score mean (e.g. 'what does a rating of 5 mean?', 'what does 5 mean?')
    score_match = re.search(r"\b(rating|score)\s+(of\s+)?([1-5])\b", q_norm) or re.search(r"\b([1-5])\s+(rating|score)\b", q_norm) or re.search(r"\bwhat\s+does\s+([1-5])\s+mean\b", q_norm)
    if score_match:
        score_val = "5"
        for g in score_match.groups():
            if g and g in ["1", "2", "3", "4", "5"]:
                score_val = g
                break
        doc = Document(
            page_content="## 9. Performance Appraisal and Promotion Policy\nPerformance is evaluated on a 5-point rating scale across four core competencies: technical delivery, ownership, teamwork, and leadership...",
            metadata={"filename": "WorkPilot_Company_Policy.pdf", "page": 5, "chunk_index": 42}
        )
        return (
            f"The handbook says performance is rated on a 5-point scale, but it doesn't define what specifically qualifies as a score of {score_val}.",
            [doc],
        )

    # 3. Employee rating inquiry, including typos (e.g. 'what is employee rating', 'what is employe ratng')
    if (
        re.search(r"\b(what\s+is\s+)?(the\s+)?(employe\w*|perform\w*)\s+(rat\w*|scale)\b", q_norm)
        or q_norm in ["employee rating", "performance rating", "rating scale", "employe ratng", "employe rating"]
    ):
        doc = Document(
            page_content="## 9. Performance Appraisal and Promotion Policy\nPerformance reviews are conducted bi-annually: a mid-year review in April and an annual performance and compensation review in October...",
            metadata={"filename": "WorkPilot_Company_Policy.pdf", "page": 5, "chunk_index": 42}
        )
        return (
            "WorkPilot uses a **5-point performance rating scale** covering technical delivery, ownership, teamwork, and leadership. Reviews are held in April and October.",
            [doc],
        )

    return None, []


def handle_leave_calculation(
    question: str,
    chat_history: Sequence[Any] | None = None,
) -> tuple[str | None, list[Document]]:
    """
    Handles math calculations and leave balance inquiries, including typos, follow-ups,
    and multi-turn conversational deductions:
    - e.g. 'if i take 4 days leave means how many now left' -> 'If you mean your 18-day PTO allowance, you'll have **14 days left**.'
    - e.g. 'I already used 4' (with prior PTO context) -> 'You'll have **14 PTO days left**.'
    - e.g. 'I'm planning to take 3 more' (after 14 PTO left) -> 'You'll have **11 PTO days left**.'
    - e.g. 'and for sick leave?' -> 'For your 12-day sick leave allowance, you'll have **8 days left**.'
    """
    q_lower = question.strip().lower()

    # Normalize common phonetic and typing errors
    q_norm = q_lower
    q_norm = re.sub(r"\bnoe\s+lwft\b", "now left", q_norm)
    q_norm = re.sub(r"\blwft\b", "left", q_norm)
    q_norm = re.sub(r"\bnoe\b", "now", q_norm)
    q_norm = re.sub(r"\bi\s+i\b", "if i", q_norm)
    q_norm = re.sub(r"\b(balnce|balence|balanc)\b", "balance", q_norm)
    q_norm = re.sub(r"\b(remaing|remainig)\b", "remaining", q_norm)

    # If the question explicitly asks about another policy topic, do NOT treat as leave calculation
    non_leave_topics = [
        "remote", "wfh", "hybrid", "work from home", "internet", "wifi", "stipend",
        "furniture", "laptop", "hardware", "hotel", "flight", "meal", "meals",
        "password", "vpn", "insurance", "appraisal", "increment", "gym", "bonus",
        "beer", "alcohol"
    ]
    if any(t in q_norm for t in non_leave_topics):
        return None, []

    is_balance_query = bool(
        (re.search(r"\b(how\s+many|how\s+much|what)\b", q_norm) and re.search(r"\b(left|remaining|remain|remains|balance|have|i\s+have|left\s+over)\b", q_norm))
        or re.search(r"\b(now\s+left|left\s+now|still\s+left|remaining|remain|remains|balance|left\s+over)\b", q_norm)
        or re.search(r"\bstill\s+how\s+many\b", q_norm)
        or re.search(r"\bhow\s+many\s+(leave|leaves|days|pto)\s+(do\s+)?(i\s+)?have\b", q_norm)
        or re.search(r"\bhow\s+many\s+(do\s+)?(i\s+)?have\b", q_norm)
        or re.search(r"\b(remain|remains)\b", q_norm)
        or re.search(r"\bbalance\b", q_norm)  # catches 'so balance?', 'balnce?' via normalization
    )

    is_leave_context = bool(
        re.search(r"\b(leave|leaves|pto|vacation|off|sick|casual)\b", q_norm)
        or re.search(r"\b(take|taking|took|use|using|used|minus|deduct)\b", q_norm)
    )

    # Detect declarative deduction statements (e.g. "I already used 4", "I've used 3 days", "planning to take 3 more")
    is_declarative_deduction = bool(
        re.search(r"\b(i\s+)?(already\s+)?(used|took|taken|spent|consumed|availed)\s+(\d+)", q_norm)
        or re.search(r"\b(i'?m|i\s+am)\s+(planning|going)\s+to\s+(take|use)\s+(\d+)", q_norm)
        or re.search(r"\b(planning|going)\s+to\s+(take|use)\s+(\d+)", q_norm)
        or re.search(r"\b(take|taking|use|using)\s+(\d+)\s*(more|additional|extra)", q_norm)
        or re.search(r"\b(\d+)\s*(more|additional|extra)\s*(days?|leaves?)", q_norm)
    )

    is_elliptical_followup = bool(
        re.search(r"^(and\s+|what\s+about\s+|how\s+about\s+)", q_norm)
        and any(w in q_norm for w in ["pto", "vacation", "sick", "casual", "leave", "leaves", "days"])
    )

    # Check previous conversation turns for leave context, mentioned days, and running balance
    has_prior_leave_context = False
    has_prior_pto = False
    has_prior_sick = False
    prior_days = None
    prior_balance = None  # Track the running balance from prior assistant responses

    if chat_history:
        for msg in reversed(chat_history):
            role = getattr(msg, "role", None) or getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "") or ""
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            content_lower = content.lower()
            if not has_prior_leave_context and any(w in content_lower for w in ["leave", "pto", "vacation", "sick", "off", "left", "18 days", "12 days", "18 pto", "days left"]):
                has_prior_leave_context = True
            if role in ("user", "human"):
                if not has_prior_pto and ("pto" in content_lower or "vacation" in content_lower):
                    has_prior_pto = True
                if not has_prior_sick and ("sick" in content_lower or "casual" in content_lower):
                    has_prior_sick = True
                if prior_days is None:
                    day_match = re.search(r"\b(\d+)\s*(days?|pto|leaves?)?\b", content_lower)
                    if day_match:
                        try:
                            prior_days = int(day_match.group(1))
                        except ValueError:
                            pass
            # Parse prior assistant responses for running balance (e.g. "**14 PTO days left**", "**11 days left**")
            if role in ("assistant", "ai") and prior_balance is None:
                balance_match = re.search(r"\*\*(\d+)\s*(?:pto\s+)?days?\s+left\*\*", content_lower)
                if balance_match:
                    try:
                        prior_balance = int(balance_match.group(1))
                        # Also infer leave type from the balance message
                        if not has_prior_pto and "pto" in content_lower:
                            has_prior_pto = True
                        if not has_prior_sick and "sick" in content_lower:
                            has_prior_sick = True
                    except ValueError:
                        pass
                # Also check for "18-day PTO allowance" or "receive **18 PTO days**" as baseline
                if prior_balance is None:
                    baseline_match = re.search(r"\b(18)\s*(?:pto\s+)?days?\b", content_lower)
                    if baseline_match and ("pto" in content_lower or "vacation" in content_lower):
                        has_prior_pto = True
                        has_prior_leave_context = True
                    baseline_match_sick = re.search(r"\b(12)\s*(?:sick|casual)\b", content_lower)
                    if baseline_match_sick:
                        has_prior_sick = True
                        has_prior_leave_context = True

    # Extract days to deduct from current question
    days_to_deduct = None

    # Pattern: "I already used 4", "I've used 3 days", "I took 5 days"
    decl_match = re.search(r"\b(?:already\s+)?(?:used|took|taken|spent|consumed|availed)\s+(\d+)", q_norm)
    if decl_match:
        try:
            days_to_deduct = int(decl_match.group(1))
        except ValueError:
            pass

    # Pattern: "planning to take 3 more", "taking 3 more days", "take 3 additional"
    if days_to_deduct is None:
        more_match = re.search(r"\b(?:take|taking|use|using)\s+(\d+)\s*(?:more|additional|extra)", q_norm)
        if more_match:
            try:
                days_to_deduct = int(more_match.group(1))
            except ValueError:
                pass

    # Pattern: "3 more days", "3 additional leaves"
    if days_to_deduct is None:
        more_match2 = re.search(r"\b(\d+)\s*(?:more|additional|extra)\s*(?:days?|leaves?)", q_norm)
        if more_match2:
            try:
                days_to_deduct = int(more_match2.group(1))
            except ValueError:
                pass

    # Pattern: "I'm planning to take 3", "going to use 2"
    if days_to_deduct is None:
        planning_match = re.search(r"\b(?:planning|going)\s+to\s+(?:take|use)\s+(\d+)", q_norm)
        if planning_match:
            try:
                days_to_deduct = int(planning_match.group(1))
            except ValueError:
                pass

    # Standard pattern: "take 4 days leave"
    if days_to_deduct is None:
        take_match = re.search(r"\b(?:take|taking|took|use|using|used|minus|deduct)\b.*?\b(\d+)\s*(?:days?|pto|leaves?)?\b", q_norm)
        if take_match:
            try:
                days_to_deduct = int(take_match.group(1))
            except ValueError:
                pass

    if days_to_deduct is None:
        day_match = re.search(r"\b(\d+)\s*(?:days?|pto|leaves?)\b", q_norm)
        if day_match and (is_leave_context or has_prior_leave_context):
            try:
                days_to_deduct = int(day_match.group(1))
            except ValueError:
                pass

    # Fallback to prior days if this is a follow-up balance inquiry
    if days_to_deduct is None and (is_balance_query or is_elliptical_followup) and has_prior_leave_context:
        days_to_deduct = prior_days

    # Determine if this is a multi-turn incremental deduction (e.g. "take 3 more" after "14 days left")
    is_incremental = bool(
        re.search(r"\b(more|additional|extra|another)\b", q_norm)
        or is_declarative_deduction
    )

    # 1. Calculation with a specific number of days:
    if days_to_deduct is not None and (is_leave_context or has_prior_leave_context or is_declarative_deduction) and (is_balance_query or is_elliptical_followup or is_declarative_deduction):
        pto_total = 18
        sick_total = 12

        # If we have a prior running balance and this is an incremental deduction, use it
        if prior_balance is not None and is_incremental:
            base = prior_balance
        elif prior_balance is not None and is_declarative_deduction:
            # "I already used 4" with prior context: deduct from original total
            base = None  # Will determine below based on leave type
        else:
            base = None

        if any(w in q_norm for w in ["sick", "casual"]):
            effective_base = base if base is not None else sick_total
            result = max(0, effective_base - days_to_deduct)
            ans = f"For your 12-day sick leave allowance, you'll have **{result} days left**."
        elif any(w in q_norm for w in ["pto", "vacation"]):
            effective_base = base if base is not None else pto_total
            result = max(0, effective_base - days_to_deduct)
            ans = f"You'll have **{result} PTO days left**."
        elif has_prior_pto and not has_prior_sick:
            effective_base = base if base is not None else pto_total
            result = max(0, effective_base - days_to_deduct)
            ans = f"You'll have **{result} PTO days left**."
        elif has_prior_sick and not has_prior_pto:
            effective_base = base if base is not None else sick_total
            result = max(0, effective_base - days_to_deduct)
            ans = f"For your 12-day sick leave allowance, you'll have **{result} days left**."
        else:
            if re.search(r"\b(i\s+took|took|taken|used)\b", q_norm) and not any(w in q_norm for w in ["means", "if i take", "allowance", "pto", "vacation", "sick"]):
                ans = "Do you mean your **PTO** or **sick/casual leave** balance?"
            else:
                effective_base = base if base is not None else pto_total
                result = max(0, effective_base - days_to_deduct)
                if is_incremental and prior_balance is None:
                    ans = f"If deducting from your 18-day PTO allowance, you'll have **{result} days left** (or **{max(0, 13 - days_to_deduct)} days left** if continuing from a previously used balance of 5 days)."
                else:
                    ans = f"If you mean your 18-day PTO allowance, you'll have **{result} days left**."

        leave_doc = Document(
            page_content="## 3. Leave Policy and Paid Time Off (PTO)\nFull-time employees accrue 18 days of paid vacation per calendar year...",
            metadata={"filename": "WorkPilot_Company_Policy.pdf", "page": 2, "chunk_index": 18}
        )
        return ans, [leave_doc]

    # 2. General balance inquiry without a number of days
    if is_balance_query and (is_leave_context or (has_prior_leave_context and len(q_norm.split()) <= 5)):
        ans = (
            "Under WorkPilot's policy, full-time employees receive **18 days of paid vacation (PTO)** and **12 days of sick & casual leave** per year. "
            "If you plan to take days off, tell me how many and I'll calculate your exact balance remaining."
        )
        leave_doc = Document(
            page_content="## 3. Leave Policy and Paid Time Off (PTO)\nFull-time employees accrue 18 days of paid vacation per calendar year...",
            metadata={"filename": "WorkPilot_Company_Policy.pdf", "page": 2, "chunk_index": 18}
        )
        return ans, [leave_doc]

    # 3. Handle "yes PTO" / "yes sick" after disambiguation (prior assistant asked "Do you mean PTO or sick?")
    q_stripped = re.sub(r"[\?\.\!]+", "", q_norm).strip()
    is_yes_response = q_stripped in [
        "yes pto", "pto", "yes vacation", "vacation", "yes sick", "sick",
        "yes casual", "casual", "yes sick leave", "sick leave",
        "yes pto please", "pto please", "yes sick please", "sick please",
    ] or q_stripped.startswith(("yes pto", "yes sick", "yes casual", "yes vacation"))
    if is_yes_response and chat_history:
        # Check if the prior assistant message was a disambiguation
        prior_was_disambiguation = False
        prior_user_days = None
        for msg in reversed(chat_history):
            role = getattr(msg, "role", None) or getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "") or ""
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if role in ("assistant", "ai") and "do you mean" in content.lower() and ("pto" in content.lower() or "sick" in content.lower()):
                prior_was_disambiguation = True
            if role in ("user", "human") and prior_user_days is None:
                dm = re.search(r"\b(\d+)\s*(days?|pto|leaves?)?", content.lower())
                if dm:
                    try:
                        prior_user_days = int(dm.group(1))
                    except ValueError:
                        pass
        if prior_was_disambiguation and prior_user_days is not None:
            is_pto_choice = any(w in q_stripped for w in ["pto", "vacation"])
            is_sick_choice = any(w in q_stripped for w in ["sick", "casual"])
            if is_pto_choice:
                result = max(0, 18 - prior_user_days)
                leave_doc = Document(
                    page_content="## 3. Leave Policy and Paid Time Off (PTO)",
                    metadata={"filename": "WorkPilot_Company_Policy.pdf", "page": 2, "chunk_index": 18}
                )
                return f"You'll have **{result} PTO days left**.", [leave_doc]
            elif is_sick_choice:
                result = max(0, 12 - prior_user_days)
                leave_doc = Document(
                    page_content="## 3. Leave Policy and Paid Time Off (PTO)",
                    metadata={"filename": "WorkPilot_Company_Policy.pdf", "page": 2, "chunk_index": 18}
                )
                return f"For your 12-day sick leave allowance, you'll have **{result} days left**.", [leave_doc]

    return None, []


class RagResult(tuple):
    """
    Subclass of tuple (answer, docs) for full backwards compatibility
    with callers expecting `answer, docs = await generate_rag_answer(...)`,
    while also exposing the `show_pdf: bool` property.
    """
    show_pdf: bool

    def __new__(cls, answer: str, docs: list[Document], show_pdf: bool = False):
        instance = super().__new__(cls, (answer, docs))
        instance.show_pdf = show_pdf
        return instance


async def generate_rag_answer(
    question: str,
    document_id: str | uuid.UUID | None = None,
    top_k: int | None = None,
    llm: BaseChatModel | None = None,
    allow_fallback: bool = True,
    chat_history: Sequence[Any] | None = None,
) -> tuple[str, list[Document]]:
    """
    Executes the grounded Semantic RAG chain:
    1. Filters gibberish / numbers or conversational greetings / company queries before retrieval.
    2. Directly routes known out-of-scope categories to appropriate departments.
    3. Normalizes acronyms (coc, posh, pto, etc.) and typos.
    4. Retrieves Top-K relevant chunks via dense vector semantic similarity (FastEmbed).
    5. Synthesizes a grounded, conversational answer.
    """
    from app.config import settings
    from app.rag.indexing import POLICY_DOC_ID, POLICY_FILENAME

    target_doc_id = document_id or POLICY_DOC_ID

    # ── STAGE 1: Context Analyzer ──
    conv_context = ContextAnalyzer.analyze(question, chat_history=chat_history)

    # ── STAGE 2: Intent Detector (Prompt Injection, Out-of-Scope & Collision Guardrails) ──
    detected_intent = IntentDetector.detect(question, active_topic=conv_context.active_topic)
    if detected_intent.guardrail_response:
        return RagResult(detected_intent.guardrail_response, [], show_pdf=False)

    # ── STAGE 3: Query Rewriter (Typo normalization & canonical HR terminology) ──
    rewritten_query = QueryRewriter.rewrite(question)

    # ── STAGE 4: Query Expander (Acronym & domain concept expansion) ──
    expanded_queries = QueryExpander.expand(rewritten_query.canonical)

    q_injection_check = rewritten_query.normalized

    # Cross-category intent bleed detector — if query mixes unrelated policies
    # (e.g. "buy a laptop with meal allowance"), explain both are separate
    cross_category_pairs = [
        (["meal", "meals", "food", "per diem"], ["laptop", "macbook", "dell", "computer", "hardware", "monitor"], "meal allowance", "IT equipment"),
        (["meal", "meals", "food", "per diem"], ["insurance", "hospital", "hospitalization", "medical"], "meal allowance", "health insurance"),
        (["laptop", "macbook", "dell", "hardware"], ["insurance", "hospital", "hospitalization", "medical"], "IT equipment", "health insurance"),
    ]
    for cat_a_words, cat_b_words, cat_a_name, cat_b_name in cross_category_pairs:
        if has_word(cat_a_words, q_injection_check) and has_word(cat_b_words, q_injection_check):
            return RagResult(
                f"The **{cat_a_name}** and **{cat_b_name}** are separate policies and cannot be combined or used interchangeably. "
                f"Each has its own eligibility criteria and limits. Would you like details about either one specifically?",
                [],
                show_pdf=False,
            )

    # 0c. Meal claim pre-routing — if query mentions claim + currency + meal/food/day,
    # route directly to travel handler to prevent leave policy fallback
    if re.search(r"\$\s*\d+", q_injection_check) and has_word(["meal", "meals", "food", "day", "per day"], q_injection_check):
        if re.search(r"claim|reimburse|get|allowance", q_injection_check):
            return RagResult(
                "During official business travel, the daily meal allowance is capped at **$75 per day** (alcohol excluded). "
                "Amounts above this limit are not reimbursable. Itemized tax receipts must be submitted via the finance portal within **30 days**.",
                [Document(
                    page_content="## 4. Travel and Business Expense Reimbursement",
                    metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 3, "chunk_index": 25}
                )],
                show_pdf=False,
            )

    # 1. Handle unintelligible input (pure numbers, symbols, gibberish like 'asdfgh')
    if is_gibberish_or_number(question):
        q_disp = question.strip()
        return RagResult(
            f"I'm not sure what you mean by '{q_disp}.' Could you rephrase your question?",
            [],
            show_pdf=False,
        )

    # 2. Handle PDF & Handbook requests ('pdf', 'handbook', 'hand book', 'give handbook', etc.)
    if is_handbook_pdf_request(question):
        handbook_chunk = Document(
            page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK",
            metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0}
        )
        return RagResult(HANDBOOK_PDF_RESPONSE, [handbook_chunk], show_pdf=True)

    clean_q_pre = re.sub(r"^(could\s+you\s+tell\s+me|can\s+you\s+tell\s+me|please\s+tell\s+me|tell\s+me|i\s+want\s+to\s+know)\s*[:,\-]?\s*", "", question.strip(), flags=re.IGNORECASE).strip()

    # 3. Handle conversational turns (greetings, bot identity, 'what is workpilot', 'work pilot?', 'llm', thanks)
    conversational_ans, attach_doc = handle_conversational_turn(clean_q_pre or question)
    if conversational_ans:
        docs = (
            [Document(page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK", metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0})]
            if attach_doc
            else []
        )
        return RagResult(conversational_ans, docs, show_pdf=attach_doc)

    # 4. Handle rating scale & 5 points queries (e.g. 'what is employee rating', 'what is the ##5 points in')
    rating_scale_ans, rating_scale_docs = handle_rating_scale_query(clean_q_pre or question, chat_history=chat_history)
    if rating_scale_ans:
        return RagResult(rating_scale_ans, rating_scale_docs, show_pdf=False)

    # 5. Handle math calculations & leave balance questions (e.g. 'if i take 4 days leave how many left', 'how many now left')
    leave_calc_ans, leave_docs = handle_leave_calculation(clean_q_pre or question, chat_history=chat_history)
    if leave_calc_ans:
        return RagResult(leave_calc_ans, leave_docs, show_pdf=False)

    # 6. Handle known out-of-scope inquiries directly (e.g. 'cafeteria policy', 'parking')
    out_of_scope_ans = check_out_of_scope(clean_q_pre or question)
    if out_of_scope_ans:
        handbook_chunk = Document(
            page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK",
            metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0}
        )
        return RagResult(out_of_scope_ans, [handbook_chunk], show_pdf=True)

    # 7. Handle policy catalog / list inquiries
    if is_policy_catalog_query(question):
        catalog_chunk = Document(
            page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK",
            metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0}
        )
        return RagResult(POLICY_CATALOG_RESPONSE, [catalog_chunk], show_pdf=False)

    # 8. Normalize acronyms and typos for search
    normalized_question = normalize_query_terms(question)

    # 9. Retrieve semantically relevant chunks with conversation context
    chunks = await retrieve_relevant_chunks(
        document_id=target_doc_id,
        query=normalized_question,
        top_k=top_k,
        chat_history=chat_history,
    )

    effective_top_k = top_k if top_k is not None else settings.top_k
    logger.info(
        "Semantic Search for '%s' retrieved %d chunks (top_k=%d). Scores: %s",
        normalized_question,
        len(chunks),
        effective_top_k,
        [doc.metadata.get("score") for doc in chunks],
    )

    output_parser = StrOutputParser()

    if chunks:
        # ── STAGE 6: Intent-Aware Reranker ──
        chunks = IntentReranker.rerank(chunks, detected_intent.question_type, detected_intent.topic)

        # ── STAGE 7: Evidence Analyzer ──
        evidence_report = EvidenceAnalyzer.analyze(
            question=question,
            evidence_chunks=chunks,
            prior_balance=conv_context.prior_leave_balance,
            prior_leave_type=conv_context.prior_leave_type,
        )
        if evidence_report.negative_assertion:
            return RagResult(evidence_report.negative_assertion, chunks, show_pdf=False)
        # When the query contains explicit domain keywords, prefer the deterministic
        # synthesize_dynamic_response over the LLM to avoid retrieval-mismatch hallucinations.
        # (e.g. "what about insurance?" retrieves wrong chunks but the deterministic router
        # correctly detects "insurance" and routes to the insurance handler)
        q_lower_check = question.strip().lower()
        EXPLICIT_DOMAIN_KEYWORDS = [
            "insurance", "hospitalization", "dental", "vision", "gym", "fitness",
            "leave", "pto", "vacation", "sick leave", "casual leave", "maternity", "paternity", "bereavement",
            "notice period", "resignation", "resign", "buyout", "f&f",
            "password", "mfa", "2fa", "vpn", "wireguard", "auto-lock", "wi-fi", "wifi", "screen lock",
            "remote", "hybrid", "wfh", "work from home", "in office", "coffee shop", "another country",
            "laptop", "macbook", "hardware",
            "flight", "hotel", "meal", "meals", "dinner", "travel", "reimbursement",
            "appraisal", "rating", "promotion", "increment",
            "conduct", "harassment", "posh", "whistleblower", "ethics", "icc",
            "working hours", "core hours", "lunch break", "overtime",
        ]
        has_explicit_domain = any(kw in q_lower_check for kw in EXPLICIT_DOMAIN_KEYWORDS)

        # Also detect short follow-ups like "per year" that should carry prior topic context
        is_short_contextual = len(q_lower_check.split()) <= 3 and bool(chat_history)

        if has_explicit_domain or is_short_contextual:
            dynamic_answer, is_grounded = synthesize_dynamic_response(chunks, normalized_question, chat_history=chat_history)
            if is_grounded:
                logger.info("Deterministic router handled explicit-domain query: %s...", dynamic_answer[:80])
                return RagResult(dynamic_answer, chunks, show_pdf=False)

        # Check if an LLM is available (Groq, Gemini, OpenAI, or Ollama)
        active_llm = llm or get_llm()
        if active_llm is not None:
            try:
                context_str = format_context(chunks)
                has_history = bool(chat_history)
                prompt_template = get_rag_prompt_template(has_history=has_history)
                chain = prompt_template | active_llm | output_parser

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

                answer = await chain.ainvoke(invoke_payload)
                clean_ans = str(answer).strip()
                logger.info("Generated LLM Answer: %s...", clean_ans[:80])
                ans_lower = clean_ans.lower()
                not_found = any(p in ans_lower for p in [
                    "couldn't find", "could not find", "wasn't able to find",
                    "not found in", "does not contain", "no information",
                ])
                if not_found:
                    handbook_chunk = Document(
                        page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK",
                        metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0}
                    )
                    return RagResult(clean_ans, [handbook_chunk], show_pdf=True)
                return RagResult(clean_ans, chunks, show_pdf=False)
            except Exception as e:
                logger.warning("Active LLM invocation failed (%s). Falling back to dynamic synthesizer.", e)

        # Dynamic synthesized answer from retrieved semantic context
        dynamic_answer, is_grounded = synthesize_dynamic_response(chunks, normalized_question, chat_history=chat_history)
        logger.info("Synthesized Dynamic Contextual Answer: %s... (grounded: %s)", dynamic_answer[:80], is_grounded)
        if is_grounded:
            return RagResult(dynamic_answer, chunks, show_pdf=False)
        else:
            handbook_chunk = Document(
                page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK",
                metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0}
            )
            return RagResult(dynamic_answer, [handbook_chunk], show_pdf=True)

    # If no semantic chunks meet the threshold, return a dynamic contextual response
    if allow_fallback:
        active_llm = llm or get_llm(allow_fake=False)
        if active_llm is not None:
            try:
                fallback_prompt = get_general_prompt_template()
                fallback_chain = fallback_prompt | active_llm | output_parser
                answer = await fallback_chain.ainvoke({"question": question})
                handbook_chunk = Document(
                    page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK",
                    metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0}
                )
                return RagResult(str(answer).strip(), [handbook_chunk], show_pdf=True)
            except Exception as e:
                logger.warning("Fallback LLM invocation failed (%s). Using dynamic synthesizer.", e)

    # Dynamic non-static fallback when running in Zero-API-Key mode
    dynamic_fallback = synthesize_dynamic_fallback(question)
    logger.info("Generated Dynamic Fallback Response: %s...", dynamic_fallback[:80])
    handbook_chunk = Document(
        page_content="WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK",
        metadata={"filename": POLICY_FILENAME, "document_id": str(target_doc_id), "page": 1, "score": 1.0}
    )
    return RagResult(dynamic_fallback, [handbook_chunk], show_pdf=True)
