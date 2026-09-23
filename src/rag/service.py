"""
Comprehensive RAG Service module for WorkPilot AI Assistant.
Contains the entire LangChain pipeline, FastEmbed embedding generation,
dual-mode vector search (PostgreSQL pgvector + local embedded fallback),
LLM answer synthesis, and chat persistence.
"""
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import engine, settings
from src.rag.model import ChatMessage, DEFAULT_DOC_ID
from src.rag.schema import ChatRequest, ChatResponse, SourceChunk

logger = logging.getLogger("src.rag.service")

POLICY_DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
POLICY_FILENAME = "company_policy.txt"
LOCAL_VECTOR_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "local_vector_store.json"

# ============================================================
# 1. EMBEDDING & LLM FACTORIES
# ============================================================

_embedding_instance: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    """Returns singleton local FastEmbed model (CPU ONNX, zero external service required)."""
    global _embedding_instance
    if _embedding_instance is None:
        logger.info("Initializing FastEmbed embedding model: %s", settings.embedding_model)
        _embedding_instance = TextEmbedding(model_name=settings.embedding_model)
    return _embedding_instance


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Generates dense vector embeddings using local FastEmbed."""
    model = get_embedding_model()
    return [vec.tolist() for vec in model.embed(list(texts))]


def embed_query(query: str) -> list[float]:
    """Generates a dense vector embedding for a single text query."""
    return embed_texts([query])[0]


def get_llm(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel | None:
    """
    Returns an OpenAI-compatible Chat model instance.
    Supports Groq, OpenAI, xAI Grok, Ollama, and OpenRouter.
    """
    key = api_key or settings.llm_api_key or ""
    b_url = base_url or settings.llm_base_url
    model_name = model or settings.llm_model

    if key.startswith("gsk_") or (b_url and "groq.com" in b_url):
        b_url = b_url or "https://api.groq.com/openai/v1"
        if not model_name or "llama-3.3-70b-versatile" in model_name:
            model_name = "openai/gpt-oss-20b"
        logger.info("Initializing Groq LLM (model=%s)", model_name)
    elif key.startswith("xai-") or (b_url and "x.ai" in b_url):
        b_url = b_url or "https://api.x.ai/v1"
        model_name = model_name if (model_name and "grok" in model_name.lower()) else "grok-2-latest"
    elif key.startswith("sk-or-") or (b_url and "openrouter.ai" in b_url):
        b_url = b_url or "https://openrouter.ai/api/v1"
        model_name = model_name or "meta-llama/llama-3.3-70b-instruct"
    elif (b_url and "11434" in b_url) or (not key and b_url):
        b_url = b_url or "http://localhost:11434/v1"
        key = key or "ollama"
        model_name = model_name or "llama3"
    elif key.startswith("sk-"):
        b_url = b_url or "https://api.openai.com/v1"
        model_name = model_name or "gpt-4o-mini"
    elif not key:
        logger.warning("No LLM API key configured (LLM_API_KEY).")
        return None

    return ChatOpenAI(
        api_key=key,
        base_url=b_url,
        model=model_name,
        temperature=temperature,
        timeout=30.0,
    )


# ============================================================
# 2. DOCUMENT SPLITTING
# ============================================================

def split_policy_document(text_content: str, base_metadata: dict) -> list[Document]:
    """Splits policy handbook markdown text into structure-aware section and clause chunks."""
    chunks: list[Document] = []
    try:
        raw_sections = text_content.split("## ")
        chunk_idx = 0

        header_intro = raw_sections[0].strip()
        if header_intro:
            chunks.append(Document(page_content=header_intro, metadata={**base_metadata, "chunk_index": chunk_idx}))
            chunk_idx += 1

        for section in raw_sections[1:]:
            lines = [l.strip() for l in section.strip().split("\n") if l.strip()]
            if not lines:
                continue
            section_title = lines[0].strip()
            section_body = section.strip()

            sec_match = re.match(r"^(\d+)", section_title)
            page_num = min(5, (int(sec_match.group(1)) - 1) // 2 + 1) if sec_match else 1

            chunk_meta = {
                **base_metadata,
                "filename": "WorkPilot_Company_Policy.pdf",
                "section": section_title,
                "page": page_num,
            }

            chunks.append(Document(page_content=f"## {section_body}", metadata={**chunk_meta, "chunk_index": chunk_idx}))
            chunk_idx += 1

            for line in lines[1:]:
                if line.startswith("-"):
                    item_text = line.lstrip("-").strip()
                    chunks.append(
                        Document(
                            page_content=f"## {section_title}\n- {item_text}",
                            metadata={**chunk_meta, "chunk_index": chunk_idx},
                        )
                    )
                    chunk_idx += 1

        return chunks
    except Exception as e:
        logger.error("Error splitting policy document: %s", e, exc_info=True)
        return []


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """Splits LangChain Document list using structure-aware or recursive text splitting."""
    if not documents:
        return []
    if len(documents) == 1 and ("## " in documents[0].page_content):
        return split_policy_document(documents[0].page_content, documents[0].metadata)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))
    for idx, c in enumerate(chunks):
        c.metadata["chunk_index"] = idx
    return chunks


# ============================================================
# 3. VECTOR STORAGE & INDEXING (DUAL-MODE / SERVICE-FREE)
# ============================================================

def get_local_vector_store() -> list[dict[str, Any]]:
    """Reads local embedded vector store (zero DB service needed)."""
    if not LOCAL_VECTOR_STORE_PATH.exists():
        return []
    try:
        return json.loads(LOCAL_VECTOR_STORE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to read local vector store: %s", e)
        return []


def save_local_vector_store(entries: list[dict[str, Any]]) -> None:
    """Saves chunks and embeddings to local file."""
    try:
        LOCAL_VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_VECTOR_STORE_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d vector records to local embedded store.", len(entries))
    except Exception as e:
        logger.error("Failed to save local vector store: %s", e)


async def store_documents(documents: Sequence[Document]) -> list[str]:
    """Stores chunks into local JSON store and syncs to PostgreSQL pgvector if reachable."""
    if not documents:
        return []

    texts = [d.page_content for d in documents]
    vectors = embed_texts(texts)

    doc_ids = []
    local_entries = []
    for doc, vec in zip(documents, vectors):
        chunk_id = str(uuid.uuid4())
        doc_ids.append(chunk_id)
        local_entries.append({
            "id": chunk_id,
            "document": doc.page_content,
            "cmetadata": doc.metadata,
            "embedding": vec,
        })

    # Always persist locally for service-free fallback
    save_local_vector_store(local_entries)

    # Attempt PostgreSQL sync
    collection_id = uuid.UUID("3a896d38-6cd0-4856-834b-12e07cff388e")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO langchain_pg_collection (uuid, name) VALUES (:uuid, :name) ON CONFLICT (uuid) DO NOTHING"),
                {"uuid": collection_id, "name": "rag_documents"},
            )
            for entry in local_entries:
                vec_str = "[" + ",".join(str(f) for f in entry["embedding"]) + "]"
                try:
                    await conn.execute(
                        text("""
                        INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                        VALUES (:id, :col_id, CAST(:vec AS vector), :doc, CAST(:meta AS jsonb))
                        ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding, document = excluded.document, cmetadata = excluded.cmetadata
                        """),
                        {
                            "id": entry["id"],
                            "col_id": collection_id,
                            "vec": vec_str,
                            "doc": entry["document"],
                            "meta": json.dumps(entry["cmetadata"]),
                        },
                    )
                except Exception:
                    await conn.execute(
                        text("""
                        INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                        VALUES (:id, :col_id, CAST(:vec AS float8[]), :doc, CAST(:meta AS jsonb))
                        ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding, document = excluded.document, cmetadata = excluded.cmetadata
                        """),
                        {
                            "id": entry["id"],
                            "col_id": collection_id,
                            "vec": entry["embedding"],
                            "doc": entry["document"],
                            "meta": json.dumps(entry["cmetadata"]),
                        },
                    )
        logger.info("Synchronized %d chunks to PostgreSQL pgvector.", len(doc_ids))
    except Exception as e:
        logger.debug("PostgreSQL storage skipped (%s). Using local embedded store.", e)

    return doc_ids


def get_policy_file_path() -> Path:
    """Resolves policy file path across environments."""
    raw = Path(settings.policy_file_path)
    if raw.is_absolute() and raw.exists():
        return raw
    if raw.exists():
        return raw
    backend_root = Path(__file__).resolve().parent.parent.parent
    candidate = backend_root / raw
    return candidate if candidate.exists() else raw


async def index_company_policy(force_reindex: bool = False) -> int:
    """Indexes company policy on application startup."""
    path = get_policy_file_path()
    if not path.exists():
        logger.error("Policy file not found at: %s", path)
        return 0

    policy_text = path.read_text(encoding="utf-8")

    # Check local store first
    local_chunks = get_local_vector_store()
    if not force_reindex and len(local_chunks) >= 30:
        logger.info("Policy already indexed in local embedded store (%d chunks).", len(local_chunks))
        return len(local_chunks)

    # Check PostgreSQL
    try:
        async with engine.connect() as conn:
            res = await conn.execute(
                text("SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id"),
                {"doc_id": str(POLICY_DOC_ID)},
            )
            count = res.scalar() or 0
            if not force_reindex and count >= 30:
                logger.info("Policy already indexed in PostgreSQL (%d chunks).", count)
                return count
    except Exception:
        pass

    raw_policy = Document(
        page_content=policy_text,
        metadata={"document_id": str(POLICY_DOC_ID), "filename": POLICY_FILENAME, "page": 1},
    )
    chunks = split_documents([raw_policy])
    if chunks:
        await store_documents(chunks)
        return len(chunks)
    return 0


# ============================================================
# 4. CONTEXTUAL RETRIEVAL & VECTOR SEARCH
# ============================================================

def contextualize_query(query: str, chat_history: Sequence[Any] | None = None) -> str:
    """Contextualizes brief follow-up queries with recent conversation context."""
    if not chat_history:
        return query

    cleaned = query.strip()
    words = cleaned.split()
    is_short = len(words) <= 5
    starts_with_connector = cleaned.lower().startswith(
        ("and ", "also ", "what about", "how about", "what if", "can i also", "does it", "is it")
    )
    if not (is_short or starts_with_connector):
        return query

    last_user_query = None
    for msg in reversed(chat_history):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "") or "user"
        content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
        if role in ("user", "human") and content.strip():
            last_user_query = content.strip()
            break

    if last_user_query and last_user_query.lower() != cleaned.lower():
        return f"{last_user_query} - {cleaned}"
    return query


async def retrieve_relevant_chunks(
    query: str,
    top_k: int | None = None,
    chat_history: Sequence[Any] | None = None,
) -> list[Document]:
    """
    Retrieves semantically relevant policy chunks using dense vector embeddings + cosine similarity.
    Queries PostgreSQL pgvector when reachable; falls back to local embedded store seamlessly.
    """
    threshold = settings.min_similarity
    k = top_k or settings.top_k
    search_query = contextualize_query(query, chat_history)

    try:
        q_vec = embed_query(search_query)
        q_arr = np.array(q_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr)) or 1e-9
    except Exception as e:
        logger.error("Failed to generate query embedding: %s", e)
        return []

    scored_chunks: list[tuple[float, Document]] = []

    # 1. Try PostgreSQL pgvector
    try:
        async with engine.connect() as conn:
            res = await conn.execute(
                text("SELECT id, cmetadata, document, embedding FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id"),
                {"doc_id": str(POLICY_DOC_ID)},
            )
            rows = res.fetchall()
            for row in rows:
                cmetadata, content, emb = row[1] or {}, row[2] or "", row[3]
                if isinstance(emb, str):
                    try:
                        emb = [float(x.strip()) for x in emb.strip("[]").split(",") if x.strip()]
                    except Exception:
                        emb = []
                if emb and len(emb) == len(q_arr):
                    c_arr = np.array(emb, dtype=np.float32)
                    c_norm = float(np.linalg.norm(c_arr)) or 1e-9
                    sim = float(np.dot(q_arr, c_arr) / (q_norm * c_norm))
                else:
                    sim = 0.0
                scored_chunks.append((sim, Document(page_content=content, metadata={**cmetadata, "score": round(sim, 4)})))
    except Exception as e:
        logger.debug("PostgreSQL query skipped (offline or not ready): %s", e)

    # 2. Fallback to local embedded store (no database service needed!)
    if not scored_chunks:
        local_entries = get_local_vector_store()
        for item in local_entries:
            cmetadata, content, emb = item.get("cmetadata", {}), item.get("document", ""), item.get("embedding", [])
            if emb and len(emb) == len(q_arr):
                c_arr = np.array(emb, dtype=np.float32)
                c_norm = float(np.linalg.norm(c_arr)) or 1e-9
                sim = float(np.dot(q_arr, c_arr) / (q_norm * c_norm))
            else:
                sim = 0.0
            scored_chunks.append((sim, Document(page_content=content, metadata={**cmetadata, "score": round(sim, 4)})))

    if not scored_chunks:
        return []

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    if not scored_chunks or scored_chunks[0][0] < threshold:
        return []

    top_score = scored_chunks[0][0]
    effective_threshold = max(threshold, top_score - 0.12)
    return [doc for score, doc in scored_chunks[:k] if score >= effective_threshold]


# ============================================================
# 5. PROMPT TEMPLATES & LANGCHAIN LCEL CHAIN
# ============================================================

RAG_SYSTEM_PROMPT = """
You are the intelligent reasoning layer of WorkPilot's HR Policy Assistant.
Your job is to understand the employee's intent, map terminology to HR concepts, and provide an accurate, grounded, and concise answer based strictly on the retrieved knowledge base.

Guidelines:
- The retrieved HR documents determine what is actually true.
- Never invent HR policies, figures, limits, or deadlines. Bold key numbers and limits.
- If information is not mentioned, state that it is not specified in available documents and refer to People Operations.

============================================================
RETRIEVED KNOWLEDGE BASE
============================================================
{context}

Answer the user's question directly adhering to all guidelines.
"""

GENERAL_SYSTEM_PROMPT = """
You are WorkPilot's intelligent HR Policy Assistant. The requested information is not documented in the available WorkPilot policies.
Keep your response concise (1-2 sentences). State that the information is not specified in the WorkPilot policies and direct the employee to People Operations for confirmation.
"""


def format_context(documents: Sequence[Document]) -> str:
    """Formats retrieved Document chunks into structured context blocks."""
    if not documents:
        return "No specific policy documents retrieved."
    blocks = []
    for doc in documents:
        filename = doc.metadata.get("filename", POLICY_FILENAME)
        page = doc.metadata.get("page", 1)
        score = doc.metadata.get("score", "N/A")
        blocks.append(f"--- [Document: {filename} | Page: {page} | Similarity: {score}] ---\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def format_chat_history(chat_history: Sequence[Any] | None) -> str:
    """Formats recent conversation turns into a clear dialogue string."""
    if not chat_history:
        return "No prior conversation."
    lines = []
    for msg in chat_history[-6:]:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "") or "user"
        content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


async def generate_rag_answer(
    question: str,
    chat_history: Sequence[Any] | None = None,
) -> tuple[str, list[Document], bool]:
    """Executes pure LangChain LCEL RAG chain."""
    chunks = await retrieve_relevant_chunks(query=question, chat_history=chat_history)
    llm = get_llm()

    if llm is None:
        return (
            "⚠️ **No LLM is configured.** Please set a valid `LLM_API_KEY` (e.g. Groq, OpenAI, xAI Grok, or local Ollama) in `.env`.",
            chunks,
            True,
        )

    output_parser = StrOutputParser()

    if chunks:
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Conversation History:\n{history}\n\nQuestion: {question}"),
        ])
        chain = prompt | llm | output_parser
        try:
            answer = await chain.ainvoke({
                "context": format_context(chunks),
                "history": format_chat_history(chat_history),
                "question": question,
            })
            clean_answer = str(answer).strip()
            show_pdf = any(p in clean_answer.lower() for p in ["not specified", "not documented", "people operations", "policy handbook"])
            return clean_answer, chunks, show_pdf
        except Exception as e:
            logger.error("LLM RAG invocation error: %s", e)
            return f"⚠️ Error generating answer from LLM: {str(e)}", chunks, True

    # Fallback when out of scope
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    fallback_chain = fallback_prompt | llm | output_parser
    try:
        answer = await fallback_chain.ainvoke({"question": question})
        return str(answer).strip(), [], True
    except Exception as e:
        logger.error("Fallback LLM error: %s", e)
        return (
            "I couldn't find information regarding that in the official company policy documents. Please consult People Operations.",
            [],
            True,
        )


# ============================================================
# 6. MAIN CHAT SERVICE & DATABASE PERSISTENCE
# ============================================================

class ChatService:
    """
    Coordinates chat message persistence, RAG context retrieval,
    and LLM answer generation.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_message(
        self,
        role: str,
        content: str,
        document_id: uuid.UUID | None = None,
    ) -> ChatMessage:
        """Inserts and commits a new chat message into the database."""
        try:
            msg = ChatMessage(
                document_id=document_id or DEFAULT_DOC_ID,
                role=role,
                content=content,
            )
            self.session.add(msg)
            await self.session.commit()
            await self.session.refresh(msg)
            return msg
        except Exception as e:
            await self.session.rollback()
            logger.error("Failed to commit chat message to database: %s", e)
            raise

    async def get_history(
        self,
        document_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> Sequence[ChatMessage]:
        """Fetches recent message history for a given document in chronological order."""
        try:
            doc_id = document_id or DEFAULT_DOC_ID
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.document_id == doc_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            messages = list(result.scalars().all())
            messages.reverse()
            return messages
        except Exception as e:
            logger.error("Failed to fetch chat history: %s", e)
            return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Processes a chat inquiry:
        1. Records user message.
        2. Executes LangChain RAG pipeline with FastEmbed vector search.
        3. Persists assistant answer.
        4. Returns response with citations.
        """
        raw_history = request.chat_history if request.chat_history is not None else request.history
        recent_history = raw_history[-6:] if raw_history else []

        # 1. Record user message
        try:
            await self.add_message(role="user", content=request.message)
        except Exception as e:
            logger.warning("Failed to save incoming user message: %s", e)

        # 2. Generate answer via LangChain RAG
        try:
            answer, matching_docs, show_pdf = await generate_rag_answer(
                question=request.message,
                chat_history=recent_history,
            )
        except Exception as e:
            logger.error("RAG pipeline failed: %s", e, exc_info=True)
            answer = "⚠️ An error occurred while generating the answer. Please try again shortly."
            matching_docs = []
            show_pdf = True

        seen_pages: set[int] = set()
        sources: list[SourceChunk] = []
        for d in matching_docs:
            page = d.metadata.get("page", 1)
            if page not in seen_pages:
                seen_pages.add(page)
                sources.append(
                    SourceChunk(
                        filename=d.metadata.get("filename", POLICY_FILENAME),
                        page=page,
                        chunk_index=d.metadata.get("chunk_index", 0),
                    )
                )

        if show_pdf and not sources:
            sources.append(SourceChunk(filename=POLICY_FILENAME, page=1, chunk_index=0))

        # 3. Record assistant answer
        try:
            await self.add_message(role="assistant", content=answer)
        except Exception as e:
            logger.warning("Failed to save assistant message: %s", e)

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id,
            show_pdf=show_pdf,
        )


# ============================================================
# 7. POLICY HANDBOOK PDF STRUCTURE & GENERATION
# ============================================================

POLICY_PAGES = [
    {
        "page": 1,
        "sections": [
            {
                "num": "1",
                "title": "Working Hours and Core Timings",
                "intro": "Standard working hours are Monday through Friday, 9:00 AM to 6:00 PM local time.",
                "bullets": [
                    "The company maintains a 40-hour work week with a mandatory 1-hour lunch break daily.",
                    "Core collaboration hours are 10:00 AM to 4:00 PM, during which team members must be reachable on Slack and available for scheduled meetings.",
                    "Flexible working hours permit employees to adjust their start time between 8:00 AM and 10:00 AM upon manager approval."
                ]
            },
            {
                "num": "2",
                "title": "Remote Work and Hybrid Guidelines",
                "intro": "The company operates on a flexible hybrid work model allowing up to 3 days of remote work per week.",
                "bullets": [
                    "Full-time remote work requires prior written approval from the Department Head and People Operations.",
                    "Remote employees receive a one-time home-office setup stipend of $500 to purchase ergonomic furniture and desk equipment.",
                    "A monthly internet and utility allowance of $50 is provided to eligible remote employees.",
                    "Employees working remotely must maintain a dedicated quiet workspace and stable internet connection of at least 50 Mbps."
                ]
            }
        ]
    },
    {
        "page": 2,
        "sections": [
            {
                "num": "3",
                "title": "Leave Policy and Paid Time Off (PTO)",
                "intro": "The annual leave year runs from January 1 to December 31.",
                "bullets": [
                    "Paid Time Off (PTO): Full-time employees accrue 18 days of paid vacation per calendar year. Up to 5 unused PTO days can be carried forward to the following year.",
                    "Sick and Casual Leave: Employees are entitled to 12 days of paid sick and casual leave annually. Medical certificates are required for sick leave extending beyond 3 consecutive days.",
                    "Maternity Leave: Female employees are entitled to 26 weeks of fully paid maternity leave for up to two surviving children.",
                    "Paternity Leave: Male employees and non-birthing partners are entitled to 4 weeks of fully paid parental leave to be taken within the first 6 months of childbirth or adoption.",
                    "Bereavement Leave: 5 consecutive paid days off are provided in the event of the loss of an immediate family member."
                ]
            },
            {
                "num": "4",
                "title": "Travel and Expense Reimbursement Policy",
                "intro": "Business-related expenses incurred on behalf of the company are eligible for reimbursement.",
                "bullets": [
                    "Daily Meal Allowance: Capped at $75 per day without alcohol during official business travel.",
                    "Flight Booking Policy: Domestic flights under 5 hours must be booked in Economy Class; flights over 5 hours or international flights qualify for Premium Economy.",
                    "Hotel Accommodation Limit: Reimbursable up to $180 per night in tier-1 cities and $120 per night in other locations.",
                    "Expense Submission Deadline: All expense reports and receipts must be submitted within 30 days of expense incurrence via the employee portal."
                ]
            }
        ]
    },
    {
        "page": 3,
        "sections": [
            {
                "num": "5",
                "title": "Health Insurance and Wellness Benefits",
                "intro": "Comprehensive group health insurance is provided to all full-time permanent employees effective from day one of employment.",
                "bullets": [
                    "Coverage: Medical, surgical, and hospitalization coverage up to $50,000 per policy year.",
                    "Dependents: Policy covers the employee, spouse, and up to two dependent children.",
                    "Dental and Vision: An annual benefit of $500 per covered member is provided for preventive dental checkups, cleaning, and corrective eyewear.",
                    "Mental Health Support: Up to 8 confidential sessions per year with licensed counselors through our Employee Assistance Program (EAP).",
                    "Wellness Stipend: $50 per month toward gym memberships, yoga classes, or fitness subscriptions."
                ]
            },
            {
                "num": "6",
                "title": "Code of Conduct and Anti-Harassment",
                "intro": "WorkPilot is committed to providing a safe, inclusive, and harassment-free workplace for everyone.",
                "bullets": [
                    "Zero Tolerance: Harassment, discrimination, or bullying based on race, gender, religion, sexual orientation, disability, or age will result in immediate disciplinary action up to termination.",
                    "Reporting: Incidents can be reported directly to People Operations, a designated HR partner, or anonymously via our confidential whistle-blower helpline.",
                    "Non-Retaliation: Retaliation against any employee reporting a violation in good faith is strictly prohibited."
                ]
            }
        ]
    },
    {
        "page": 4,
        "sections": [
            {
                "num": "7",
                "title": "Device Security, Data Protection, and Acceptable Use",
                "intro": "All company-provided laptops and equipment are monitored for security compliance.",
                "bullets": [
                    "Multi-Factor Authentication (MFA): Mandatory for all company accounts, SSO, VPN, and email access.",
                    "Password Policy: Minimum 12 characters with a mix of uppercase, lowercase, numbers, and symbols, rotated every 90 days.",
                    "Data Classification: Customer data and source code are classified as Confidential and must never be copied to personal devices, USB drives, or unapproved cloud storage.",
                    "Incident Reporting: Lost or stolen laptops must be reported to the IT Security Team within 2 hours of discovery for immediate remote wipe."
                ]
            },
            {
                "num": "8",
                "title": "Performance Reviews, Promotions, and Appraisals",
                "intro": "Performance appraisals follow a structured bi-annual review cycle in June and December.",
                "bullets": [
                    "Self-evaluation followed by 360-degree peer feedback and manager review.",
                    "Performance ratings range from 1 (Needs Improvement) to 5 (Exceeds Expectations).",
                    "Promotion Eligibility: Requires minimum 12 months in current role and sustained rating of 4 or above in the previous two evaluation cycles.",
                    "Annual Merit Increases: Effective annually on April 1 based on overall company performance and individual ratings."
                ]
            }
        ]
    },
    {
        "page": 5,
        "sections": [
            {
                "num": "9",
                "title": "Learning, Development, and Certifications",
                "intro": "Continuous learning and professional growth are core values at WorkPilot.",
                "bullets": [
                    "Annual Learning Budget: $1,200 per full-time employee per calendar year for courses, books, workshops, and conferences.",
                    "Professional Certifications: Examination fees for relevant technical or domain certifications are 100% reimbursed upon passing.",
                    "Study Leave: Up to 3 days of paid study leave per year for approved certification examinations."
                ]
            },
            {
                "num": "10",
                "title": "Separation, Resignation, and Exit Process",
                "intro": "Guidelines for a smooth offboarding process when an employee leaves the company.",
                "bullets": [
                    "Notice Period: Standard notice period is 30 days for individual contributors and 60 days for lead and managerial roles.",
                    "Notice Buyout: Permissible only with written approval from the Department Head and People Operations.",
                    "Asset Return: All company property including laptops, monitors, access cards, and company credit cards must be returned by the last working day.",
                    "Full and Final Settlement: Processed within 30 days of the last working day, including encashment of eligible unused PTO days."
                ]
            }
        ]
    }
]


def generate_company_policy_pdf(target_path: Path | str | None = None) -> Path:
    """
    Generates an official enterprise-grade WorkPilot Company Policy Handbook PDF.
    """
    if target_path is None:
        target_path = Path(__file__).resolve().parent.parent.parent / "data" / "WorkPilot_Company_Policy.pdf"
    else:
        target_path = Path(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        logger.warning("ReportLab is not installed; writing fallback PDF placeholder")
        target_path.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n108\n%%EOF\n")
        return target_path

    doc = SimpleDocTemplate(str(target_path), pagesize=letter, rightMargin=45, leftMargin=45, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    doc_header_style = ParagraphStyle("DocHeader", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#0F172A"), spaceAfter=2)
    doc_sub_style = ParagraphStyle("DocSub", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.HexColor("#2563EB"), spaceAfter=6)
    sec_title_style = ParagraphStyle("SecTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor("#1E293B"), spaceBefore=8, spaceAfter=4)
    intro_style = ParagraphStyle("IntroStyle", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=9, leading=12.5, textColor=colors.HexColor("#334155"), spaceAfter=5)
    bullet_style = ParagraphStyle("BulletStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#1E293B"), leftIndent=14, spaceAfter=4)
    page_footer_style = ParagraphStyle("PageFooter", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#94A3B8"), alignment=1)

    story = []
    for page_idx, page_data in enumerate(POLICY_PAGES):
        page_num = page_data["page"]
        story.append(Paragraph("WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK", doc_header_style))
        story.append(Paragraph("Official Human Resources & Operations Guidelines | Confidential & Proprietary", doc_sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))

        for sec in page_data["sections"]:
            story.append(Paragraph(f"## {sec['num']}. {sec['title']}", sec_title_style))
            if sec.get("intro"):
                story.append(Paragraph(sec["intro"], intro_style))

            for bullet in sec.get("bullets", []):
                if ":" in bullet:
                    parts = bullet.split(":", 1)
                    bullet_text = f"• <b>{parts[0].strip()}:</b> {parts[1].strip()}"
                else:
                    bullet_text = f"• {bullet.strip()}"
                story.append(Paragraph(bullet_text, bullet_style))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 14))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=6))
        story.append(Paragraph(f"Page {page_num} of 5 — WorkPilot Policy Documentation", page_footer_style))

        if page_idx < len(POLICY_PAGES) - 1:
            story.append(PageBreak())

    doc.build(story)
    return target_path
