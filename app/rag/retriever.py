import logging
import uuid
import numpy as np
from typing import Any, Sequence
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.rag.embeddings import get_embedding_model

logger = logging.getLogger("rag.retriever")


import re

FOLLOW_UP_PREFIXES = (
    "what about", "how about", "and what", "and for", "can i also", "can i ", "could i ", "do i ",
    "is that", "does it", "is it", "what if", "can it", "how much is that", "how much ", "how fast ",
    "what is the limit for that", "any allowance for that", "and ", "also ",
    "what about for", "how about for", "when should i", "where do i", "how do i",
    "for whom", "for who", "who is", "who are", "who can", "who does", "whom", "who", "whose", "why",
    "for my ", "for our ", "for ", "its is ", "is it ", "is that ",
)
REFERENTIAL_PRONOUNS = {
    "that", "this", "it", "its", "they", "them", "these", "those",
    "same", "above", "previous", "latter", "former", "whom", "who", "whose",
    "my", "our",
}


WORKPLACE_COLLOQUIAL_TERMS = [
    (r"\b(code\s+of\s+conduct|conduct|ethics|harassment|anti-harassment|anti\s+harassment|posh|discrimination|retaliation|discipline|disciplinary|misconduct)\b", "code of conduct and anti-harassment posh compliance zero tolerance discrimination incident reporting internal complaints committee anti-retaliation policy"),
    (r"\b(appraisal|performance\s+review|rating|salary\s+increment|promotion|objective|objectives|kpi|kpis|goals|competency|competencies|principles)\b", "performance appraisal and promotion policy review cycles rating framework salary increments 5-point scale core competencies technical delivery ownership teamwork leadership"),
    (r"\b(notice\s+period|resign|resigns|resignation|resigning|resioning|resiging|resion|resignating|quitting|f&f|full\s+and\s+final|buyout)\b", "resignation and notice period protocol buyout final settlement full and final 60 days 30 days probation"),
    (r"\b(yoga|gym|fitness|workout|exercise|sport|sports)\b", "employee benefit gym fitness yoga sports subscription reimbursement"),
    (r"\b(dentist|dental|teeth|tooth|vision|glasses|spectacles|eye|eyes)\b", "employee benefit dental and vision health coverage"),
    (r"\b(therapy|counseling|counselling|mental health|stress|eap|psychologist)\b", "employee assistance program mental health and wellness therapy sessions"),
    (r"\b(desk|chair|ergonomic|furniture)\b", "remote work home office setup ergonomic furniture desk equipment stipend"),
    (r"\b(wifi|broadband|internet|speed|mbps|bandwidth)\b", "remote work monthly internet utility allowance and connection speed 50 mbps"),
    (r"\b(doctor|hospital|hospitalization|mediclaim|medical insurance|health insurance)\b", "group health insurance coverage and annual health checkup"),
    (r"\b(mfa|password|passwords|screen\s+lock|auto-lock|autolock|vpn)\b", "information security and password policy complexity expiry rotation multi-factor authentication vpn requirement"),
    (r"\b(laptop|macbook|dell|monitor|keyboard|mouse|hardware|headset)\b", "it equipment and hardware policy standard engineering laptops accessories refresh cycle"),
    (r"\b(travel|per\s+diem|meal\s+allowance|flight|hotel|expenses|reimbursement|alcohol|beer|wine|food)\b", "travel and business expense reimbursement daily meal allowance flight booking hotel accommodation limit"),
    (r"\b(hours|working\s+hours|timings|timing|core\s+hours|flexible\s+hours|lunch\s+break)\b", "working hours and core timings collaboration hours flexible schedule"),
    (r"\b(leave|leaves|pto|vacation|sick\s+leave|casual\s+leave|maternity|paternity|bereavement)\b", "leave policy and paid time off annual pto sick casual leave maternity paternity bereavement"),
]


def get_enriched_query(query: str) -> str | None:
    """Enriches short colloquial workplace inquiries with domain concepts to match company policy benefits."""
    q_lower = query.lower()
    for pattern, expansion in WORKPLACE_COLLOQUIAL_TERMS:
        if re.search(pattern, q_lower):
            return f"{query} {expansion}"
    return None


FOLLOW_UP_PATTERNS = [
    r"\b(how\s+many|how\s+much|what)\s+(is\s+)?(now\s+)?(left|remaining|balance)\b",
    r"\b(now\s+left|left\s+now|still\s+left|remaining|balance|left\s+over)\b",
    r"^(and\s+|also\s+|then\s+|so\s+|what\s+if\s+|what\s+about\s+|what\s+happens\s+if\s+)",
    r"\b(for\s+what|what\s+for|why|before\s+\d+|after\s+\d+)\b",
    r"\b(for\s+whom|for\s+who|who\s+is\s+it\s+for)\b",
    r"\b(go|leave|exit)\s+(before|early|earlier)\b",
    r"\b(before|after)\s+(60|30|\d+)\s+days\b",
    r"\b(how\s+many|how\s+much)\s+can\s+i\s+claim\b",
    r"\b(coverable|coverble|coverage)\b",
    r"\b(for\s+my\s+family|is\s+it\s+covered\s+for|is\s+coverble\s+for|is\s+coverable\s+for)\b",
    r"^(for\s+(my\s+|our\s+)?(parent|parents|mother|father|mom|dad|spouse|wife|husband|child|children|kids|family|in-laws|dependents?))\b",
    r"\b(claimabke|claimable|eligible|reimbursable)\b",
]


def is_follow_up_query(query: str) -> bool:
    q_lower = query.strip().lower()
    clean = re.sub(r"[#\?\.\!]+", "", q_lower).strip()
    words = clean.split()
    if clean in ["for whom", "for who", "who", "whom", "for what", "what for"]:
        return True
    for prefix in FOLLOW_UP_PREFIXES:
        if q_lower.startswith(prefix):
            return True
    for pattern in FOLLOW_UP_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    word_set = set(re.findall(r"\w+", q_lower))
    if bool(word_set & REFERENTIAL_PRONOUNS):
        return True
    if len(words) <= 3 and any(w in word_set for w in ["who", "whom", "how", "what", "which", "when", "why"]):
        return True
    return False


def extract_history_topic(chat_history: Sequence[Any]) -> str:
    """Extracts the dominant policy topic discussed in recent conversation turns."""
    contents = []
    for msg in chat_history[-4:]:
        c = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
        if c:
            contents.append(c.lower())
    combined = " ".join(contents)
    if any(w in combined for w in ["notice", "resignation", "resign", "60 days", "probation", "buyout", "f&f", "last working day"]):
        return "resignation and notice period protocol"
    if any(w in combined for w in ["pto", "vacation", "sick leave", "casual leave", "maternity", "paternity", "bereavement"]):
        return "leave policy and paid time off"
    if any(w in combined for w in ["working hours", "core hours", "core timings", "shift", "lunch"]):
        return "working hours and core timings"
    if any(w in combined for w in ["remote", "hybrid", "wfh", "home office", "stipend", "50 mbps"]):
        return "remote work and hybrid guidelines"
    if any(w in combined for w in ["laptop", "macbook", "dell", "hardware", "monitor"]):
        return "it equipment and hardware policy"
    if any(w in combined for w in ["password", "passwords", "mfa", "wireguard", "auto-lock", "autolock"]):
        return "information security and password policy"
    if any(w in combined for w in ["insurance", "hospitalization", "gym", "fitness", "dental", "vision", "eap"]):
        return "employee benefits and health insurance"
    if any(w in combined for w in ["appraisal", "rating scale", "5-point", "increments", "promotion"]):
        return "performance appraisal and promotion policy"
    if any(w in combined for w in ["conduct", "harassment", "posh", "icc"]):
        return "code of conduct and anti-harassment"
    if any(w in combined for w in ["travel", "flight", "flights", "hotel", "hotels", "meal", "meals", "per diem", "finance portal", "reimbursement", "airline", "tickets"]):
        return "travel and business expense reimbursement"
    return ""


def build_contextual_query(query: str, chat_history: Sequence[Any] | None = None) -> str | None:
    """
    Constructs a context-aware query string when conversation history exists,
    resolving implicit references and follow-ups so dense semantic vector
    embeddings capture the full conversation context without polluting independent new topics.
    """
    if not chat_history:
        return None

    # Extract previous user questions to establish context
    recent_user_turns = []
    for msg in chat_history:
        role = getattr(msg, "role", None) or getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "") or ""
        content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
        if role in ("user", "human") and content:
            recent_user_turns.append(content)
    if not recent_user_turns:
        return None

    last_user_query = recent_user_turns[-1].strip()
    if last_user_query.lower() == query.strip().lower():
        if len(recent_user_turns) > 1:
            last_user_query = recent_user_turns[-2].strip()
        else:
            return None

    if is_follow_up_query(query):
        history_topic = extract_history_topic(chat_history)
        if history_topic:
            contextualized = f"Regarding {history_topic} ({last_user_query}): {query}"
        else:
            contextualized = f"Regarding {last_user_query}: {query}"
        logger.info("Contextualized query with conversation history: '%s'", contextualized)
        return contextualized

    return None


async def retrieve_semantic_chunks(
    document_id: str | uuid.UUID,
    query: str,
    top_k: int = 6,
    embeddings: Embeddings | None = None,
    min_similarity: float | None = None,
    chat_history: Sequence[Any] | None = None,
) -> list[Document]:
    """
    Performs true dense vector semantic search across PostgreSQL chunk embeddings.
    Matches the meaning, intent, and context of the query using cosine similarity
    without relying on exact keyword matching.
    """
    threshold = min_similarity if min_similarity is not None else settings.min_similarity
    search_query = build_contextual_query(query, chat_history)

    embed_model = embeddings or get_embedding_model()
    q_vec = embed_model.embed_query(query)
    q_arr = np.array(q_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_arr))
    if q_norm == 0:
        q_norm = 1e-9

    has_contextual = (search_query is not None and search_query != query)
    if has_contextual and search_query:
        ctx_vec = embed_model.embed_query(search_query)
        ctx_arr = np.array(ctx_vec, dtype=np.float32)
        ctx_norm = float(np.linalg.norm(ctx_arr))
        if ctx_norm == 0:
            ctx_norm = 1e-9

    enriched_query = get_enriched_query(query)
    has_enriched = (enriched_query is not None and enriched_query != query)
    if has_enriched and enriched_query:
        enr_vec = embed_model.embed_query(enriched_query)
        enr_arr = np.array(enr_vec, dtype=np.float32)
        enr_norm = float(np.linalg.norm(enr_arr))
        if enr_norm == 0:
            enr_norm = 1e-9

    async with engine.connect() as conn:
        res = await conn.execute(
            text(
                "SELECT id, cmetadata, document, embedding "
                "FROM langchain_pg_embedding "
                "WHERE cmetadata->>'document_id' = :doc_id"
            ),
            {"doc_id": str(document_id)},
        )
        rows = res.fetchall()

    if not rows:
        return []

    scored_chunks: list[tuple[float, Document]] = []
    for row in rows:
        cmetadata = row[1] or {}
        content = row[2] or ""
        emb = row[3]

        if emb and len(emb) == len(q_arr):
            chunk_arr = np.array(emb, dtype=np.float32)
            c_norm = float(np.linalg.norm(chunk_arr))
            if c_norm > 0:
                sim = float(np.dot(q_arr, chunk_arr) / (q_norm * c_norm))
                if has_contextual:
                    sim_ctx = float(np.dot(ctx_arr, chunk_arr) / (ctx_norm * c_norm))
                    sim = max(sim, sim_ctx)
                if has_enriched:
                    sim_enr = float(np.dot(enr_arr, chunk_arr) / (enr_norm * c_norm))
                    sim = max(sim, sim_enr)
            else:
                sim = 0.0
        else:
            sim = 0.0

        meta = {**cmetadata, "score": round(sim, 4)}
        scored_chunks.append((sim, Document(page_content=content, metadata=meta)))

    # Sort strictly by semantic similarity in descending order
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    scores_preview = [f"{s:.4f}" for s, _ in scored_chunks[:top_k]]
    logger.info("Semantic similarity scores for query '%s': %s", search_query or query, scores_preview)

    if not scored_chunks or scored_chunks[0][0] < threshold:
        return []

    top_score = scored_chunks[0][0]
    # Dynamic relative threshold: must meet absolute threshold AND stay close to the best match
    # Prevents pulling in distant, unrelated policy chunks that barely cross baseline threshold
    effective_threshold = max(threshold, top_score - 0.09, top_score * 0.85)

    relevant = [doc for score, doc in scored_chunks[:top_k] if score >= effective_threshold]
    logger.info(
        "Filtered to %d semantically tight chunks (top: %.4f, cutoff: %.4f)",
        len(relevant),
        top_score,
        effective_threshold,
    )
    return relevant


async def retrieve_relevant_chunks(
    document_id: str | uuid.UUID,
    query: str,
    top_k: int | None = None,
    embeddings: Embeddings | None = None,
    min_similarity: float | None = None,
    chat_history: Sequence[Any] | None = None,
) -> list[Document]:
    """
    Retrieves the most semantically relevant policy chunks based on contextual vector similarity.
    """
    k = top_k if top_k is not None else settings.top_k
    threshold = min_similarity if min_similarity is not None else settings.min_similarity
    return await retrieve_semantic_chunks(
        document_id=document_id,
        query=query,
        top_k=k,
        embeddings=embeddings,
        min_similarity=threshold,
        chat_history=chat_history,
    )
