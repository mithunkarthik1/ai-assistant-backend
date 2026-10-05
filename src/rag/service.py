"""
Comprehensive RAG Service module for WorkPilot AI Assistant.
Contains the entire LangChain pipeline, FastEmbed embedding generation,
dual-mode vector search (PostgreSQL pgvector + local embedded fallback),
LLM answer synthesis, and chat persistence.
"""
import hashlib
import io
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.database.connection import engine
from src.rag.model import ChatMessage, DEFAULT_DOC_ID, Document as DocumentModel, DocumentChunk
from src.rag.schema import (
    ChatRequest,
    ChatResponse,
    DocumentDetailResponse,
    DocumentInfoResponse,
    DocumentUploadResponse,
    SourceChunk,
)

logger = logging.getLogger("src.rag.service")

POLICY_DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
POLICY_FILENAME = "WorkPilot_Company_Policy.pdf"
LOCAL_VECTOR_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "local_vector_store.json"
COLLECTION_ID = uuid.UUID("3a896d38-6cd0-4856-834b-12e07cff388e")

# ============================================================
# 1. DOCUMENT EXTRACTION UTILITIES (PDF, DOCX, TXT)
# ============================================================

class ExtractionError(Exception):
    """Raised when text extraction from a file fails."""
    pass


def extract_text_from_pdf(file_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extracts text page-by-page from PDF bytes using pypdf.
    Returns list of dicts: [{"page_number": 1, "text": "..."}, ...]
    """
    try:
        import pypdf
    except ImportError as e:
        logger.error("pypdf is required for PDF extraction: %s", e)
        raise ExtractionError("pypdf library is not installed.") from e

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise ExtractionError("Encrypted PDF could not be decrypted.") from e

        pages_data: list[dict[str, Any]] = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages_data.append({
                "page_number": idx + 1,
                "text": page_text,
            })

        total_extracted = sum(len(p["text"].strip()) for p in pages_data)
        if total_extracted == 0:
            logger.warning("PDF extracted 0 text characters across %d pages (may be scanned images).", len(pages_data))

        return pages_data
    except Exception as e:
        logger.error("Failed to extract text from PDF: %s", e, exc_info=True)
        raise ExtractionError(f"Failed to extract text from PDF: {str(e)}") from e


def extract_text_from_docx(file_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extracts text from DOCX bytes using python-docx.
    Detects heading paragraphs to preserve document structure.
    """
    try:
        import docx
    except ImportError as e:
        logger.error("python-docx is required for DOCX extraction: %s", e)
        raise ExtractionError("python-docx library is not installed.") from e

    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text_lines: list[str] = []

        for para in doc.paragraphs:
            content = para.text.strip()
            if not content:
                continue

            style_name = getattr(para.style, "name", "").lower()
            if "heading 1" in style_name:
                text_lines.append(f"\n# {content}\n")
            elif "heading 2" in style_name:
                text_lines.append(f"\n## {content}\n")
            elif "heading 3" in style_name:
                text_lines.append(f"\n### {content}\n")
            else:
                text_lines.append(content)

        # Include tables if any
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_lines.append(row_text)

        full_text = "\n".join(text_lines)
        return [{"page_number": 1, "text": full_text}]
    except Exception as e:
        logger.error("Failed to extract text from DOCX: %s", e, exc_info=True)
        raise ExtractionError(f"Failed to extract text from DOCX: {str(e)}") from e


def extract_text_from_txt(file_bytes: bytes) -> list[dict[str, Any]]:
    """
    Decodes plain text or markdown file bytes.
    """
    try:
        text_content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text_content = file_bytes.decode("latin-1")
        except Exception as e:
            raise ExtractionError(f"Failed to decode text file: {str(e)}") from e

    return [{"page_number": 1, "text": text_content}]


def extract_document_pages(file_bytes: bytes, file_name: str) -> list[dict[str, Any]]:
    """
    Dispatches document bytes to the appropriate extractor based on file extension.
    """
    if not file_bytes:
        raise ExtractionError(f"Uploaded file '{file_name}' is empty (0 bytes).")

    ext = file_name.split(".")[-1].lower() if "." in file_name else ""

    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    elif ext in ("txt", "md", "markdown", "rst"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ExtractionError(f"Unsupported file format '.{ext}'. Supported formats: .pdf, .docx, .txt, .md")


def extract_document_text(
    file_bytes: bytes,
    file_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    """
    Extracts pages, computes document-level SHA-256 hash, and detects file metadata.
    Returns: (pages_data, metadata_dict, document_sha256_hash, file_type)
    """
    pages = extract_document_pages(file_bytes, file_name)
    ext = file_name.split(".")[-1].lower() if "." in file_name else "pdf"

    # Combine text from all pages to compute document-level SHA-256 content hash
    full_text = "\n".join(p.get("text", "") for p in pages)
    normalized = "\n".join(line.strip() for line in full_text.splitlines() if line.strip())
    doc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    metadata = {
        "file_name": file_name,
        "file_type": ext,
        "page_count": len(pages),
        "char_count": len(full_text),
    }

    return pages, metadata, doc_hash, ext


# ============================================================
# 2. HYBRID CHUNKING ENGINE (SECTION, TOPIC, SEMANTIC & OVERLAP)
# ============================================================

@dataclass
class HybridChunk:
    chunk_id: str
    document_id: uuid.UUID
    section: str
    topic: str
    chunk_index: int
    content: str
    content_hash: str
    page_number: int
    metadata: dict[str, Any]


def slugify(text: str, max_words: int = 4) -> str:
    """
    Converts a heading/title into a clean, deterministic, alphanumeric slug.
    Example: 'Leave Policy and Paid Time Off (PTO)' -> 'leave_policy_pto'
    """
    if not text:
        return "general"

    cleaned = re.sub(r"[^\w\s-]", " ", text.lower()).strip()
    words = [w for w in cleaned.split() if w and w not in ("and", "or", "the", "a", "an", "of", "in", "to", "for")]
    if not words:
        words = cleaned.split()[:max_words]
    else:
        words = words[:max_words]

    slug = "_".join(words)
    return slug[:40] if slug else "general"


def normalize_text_for_hash(text: str) -> str:
    """
    Normalizes chunk text for hashing so that trivial whitespace variations
    (trailing spaces, blank lines, carriage returns) do not cause false re-embedding.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(l for l in lines if l)
    return normalized.strip()


def calculate_content_hash(text: str) -> str:
    """
    Computes deterministic SHA-256 hash of the normalized chunk content.
    """
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_heading(line: str) -> tuple[bool, int, str]:
    """
    Detects if a text line is a section or topic heading.
    Returns (is_heading, heading_level, heading_title).
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False, 0, ""

    # 1. Markdown headings (#, ##, ###, ####)
    md_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
    if md_match:
        level = len(md_match.group(1))
        title = md_match.group(2).strip()
        return True, level, title

    # 2. Numbered headings (e.g. '1. Working Hours', 'Section 2: Remote Work', 'Article 3 - Leave')
    numbered_match = re.match(r"^(?:Section\s+|Article\s+)?(\d+(?:\.\d+)*)[:.\-\s]+\s*([A-Za-z].+)$", stripped, re.IGNORECASE)
    if numbered_match:
        num_parts = numbered_match.group(1).split(".")
        level = min(len(num_parts) + 1, 4)
        title = stripped
        return True, level, title

    # 3. Standalone UPPERCASE heading (at least 3 words or 12 chars, not a sentence)
    if stripped.isupper() and len(stripped) >= 8 and not stripped.endswith((".", ":", ";")):
        return True, 1, stripped.title()

    # 4. Heading ending with colon without terminal period and short length
    if stripped.endswith(":") and len(stripped.split()) <= 8 and not any(p in stripped for p in [".", "?", "!"]):
        return True, 2, stripped.rstrip(":")

    return False, 0, ""


def hybrid_chunk_pages(
    pages_data: Sequence[dict[str, Any]],
    document_id: uuid.UUID,
    file_name: str,
    target_chunk_size: int = 800,
    small_overlap: int = 80,
) -> list[HybridChunk]:
    """
    Applies the hybrid chunking pipeline across document pages:
    1. Tracks active section & topic across pages.
    2. Identifies clause-level items (bullet points, sub-clauses).
    3. Breaks oversized sections semantically with small overlap.
    4. Produces stable, deterministic chunk IDs and SHA-256 content hashes.
    """
    short_doc_id = str(document_id).split("-")[0]
    hybrid_chunks: list[HybridChunk] = []

    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chunk_size,
        chunk_overlap=small_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        length_function=len,
    )

    current_section = "Overview"
    current_topic = "Introduction"
    global_chunk_idx = 0

    topic_chunk_counters: dict[str, int] = {}

    for page_info in pages_data:
        page_num = page_info.get("page_number", 1)
        raw_text = page_info.get("text", "")
        if not raw_text.strip():
            continue

        lines = raw_text.split("\n")
        current_block: list[str] = []

        def flush_block(section_name: str, topic_name: str, page: int):
            nonlocal global_chunk_idx
            block_text = "\n".join(current_block).strip()
            current_block.clear()
            if not block_text:
                return

            sec_slug = slugify(section_name, max_words=3)
            top_slug = slugify(topic_name, max_words=3)
            group_key = f"{sec_slug}_{top_slug}"

            if len(block_text) <= target_chunk_size + 150:
                topic_counter = topic_chunk_counters.get(group_key, 0) + 1
                topic_chunk_counters[group_key] = topic_counter

                chunk_id = f"doc_{short_doc_id}_{sec_slug}_{top_slug}_{topic_counter:03d}"
                c_hash = calculate_content_hash(block_text)

                meta = {
                    "document_id": str(document_id),
                    "filename": file_name,
                    "chunk_id": chunk_id,
                    "section": section_name,
                    "topic": topic_name,
                    "chunk_index": global_chunk_idx,
                    "page": page,
                    "content_hash": c_hash,
                }

                hybrid_chunks.append(
                    HybridChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        section=section_name,
                        topic=topic_name,
                        chunk_index=global_chunk_idx,
                        content=block_text,
                        content_hash=c_hash,
                        page_number=page,
                        metadata=meta,
                    )
                )
                global_chunk_idx += 1
            else:
                sub_texts = semantic_splitter.split_text(block_text)
                for sub_t in sub_texts:
                    sub_t_clean = sub_t.strip()
                    if not sub_t_clean:
                        continue

                    topic_counter = topic_chunk_counters.get(group_key, 0) + 1
                    topic_chunk_counters[group_key] = topic_counter

                    chunk_id = f"doc_{short_doc_id}_{sec_slug}_{top_slug}_{topic_counter:03d}"
                    c_hash = calculate_content_hash(sub_t_clean)

                    meta = {
                        "document_id": str(document_id),
                        "filename": file_name,
                        "chunk_id": chunk_id,
                        "section": section_name,
                        "topic": topic_name,
                        "chunk_index": global_chunk_idx,
                        "page": page,
                        "content_hash": c_hash,
                    }

                    hybrid_chunks.append(
                        HybridChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            section=section_name,
                            topic=topic_name,
                            chunk_index=global_chunk_idx,
                            content=sub_t_clean,
                            content_hash=c_hash,
                            page_number=page,
                            metadata=meta,
                        )
                    )
                    global_chunk_idx += 1

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                if current_block:
                    current_block.append("")
                continue

            is_head, level, title = is_heading(trimmed)
            if is_head:
                flush_block(current_section, current_topic, page_num)

                if level == 1:
                    current_section = title
                    current_topic = "General"
                elif level == 2:
                    if current_section == "Overview":
                        current_section = title
                        current_topic = "Details"
                    else:
                        current_topic = title
                else:
                    current_topic = title

                current_block.append(trimmed)
            else:
                bullet_clause = re.match(r"^[-*•\d.]+\s*(?:\*\*(.+?)\*\*|([A-Za-z0-9\s/&]+):)\s*(.+)$", trimmed)
                if bullet_clause and len(current_block) > 4:
                    flush_block(current_section, current_topic, page_num)
                    item_topic = bullet_clause.group(1) or bullet_clause.group(2)
                    if item_topic and len(item_topic.split()) <= 4:
                        current_topic = item_topic.strip()

                current_block.append(trimmed)

        flush_block(current_section, current_topic, page_num)

    return hybrid_chunks


def to_langchain_documents(chunks: Sequence[HybridChunk]) -> list[Document]:
    """Converts HybridChunk objects into LangChain Document instances."""
    return [
        Document(
            page_content=f"## {c.section} - {c.topic}\n{c.content}" if c.section and c.section != "Overview" else c.content,
            metadata=c.metadata,
        )
        for c in chunks
    ]


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
# 2. DOCUMENT SPLITTING (LEGACY COMPATIBILITY)
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
                "filename": POLICY_FILENAME,
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


async def upsert_vector_store(entries: list[dict[str, Any]]) -> None:
    """
    Upserts vector entries with stable chunk IDs into PostgreSQL pgvector
    and local embedded store simultaneously.
    """
    if not entries:
        return

    # 1. Update local store
    local_entries = get_local_vector_store()
    entry_map = {str(e["id"]): e for e in local_entries}
    for e in entries:
        entry_map[str(e["id"])] = e
    save_local_vector_store(list(entry_map.values()))

    # 2. Update PostgreSQL
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO langchain_pg_collection (uuid, name) VALUES (:uuid, :name) ON CONFLICT (uuid) DO NOTHING"),
                {"uuid": COLLECTION_ID, "name": "rag_documents"},
            )
            for entry in entries:
                vec_str = "[" + ",".join(str(f) for f in entry["embedding"]) + "]"
                try:
                    await conn.execute(
                        text("""
                        INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                        VALUES (:id, :col_id, CAST(:vec AS vector), :doc, CAST(:meta AS jsonb))
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = excluded.embedding,
                            document = excluded.document,
                            cmetadata = excluded.cmetadata
                        """),
                        {
                            "id": str(entry["id"]),
                            "col_id": COLLECTION_ID,
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
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = excluded.embedding,
                            document = excluded.document,
                            cmetadata = excluded.cmetadata
                        """),
                        {
                            "id": str(entry["id"]),
                            "col_id": COLLECTION_ID,
                            "vec": entry["embedding"],
                            "doc": entry["document"],
                            "meta": json.dumps(entry["cmetadata"]),
                        },
                    )
        logger.info("Upserted %d vector records to PostgreSQL pgvector.", len(entries))
    except Exception as e:
        logger.debug("PostgreSQL vector upsert skipped (%s). Using local embedded store.", e)


async def delete_from_vector_store(chunk_ids: Sequence[str]) -> None:
    """
    Deletes specified chunk IDs from both PostgreSQL pgvector and local embedded store.
    """
    if not chunk_ids:
        return

    # 1. Local store
    local_entries = get_local_vector_store()
    ids_set = {str(c) for c in chunk_ids}
    remaining = [e for e in local_entries if str(e.get("id")) not in ids_set]
    save_local_vector_store(remaining)

    # 2. PostgreSQL
    try:
        async with engine.begin() as conn:
            for cid in chunk_ids:
                await conn.execute(
                    text("DELETE FROM langchain_pg_embedding WHERE id = :id"),
                    {"id": str(cid)},
                )
        logger.info("Deleted %d vectors from PostgreSQL pgvector.", len(chunk_ids))
    except Exception as e:
        logger.debug("PostgreSQL vector delete skipped (%s).", e)


async def store_documents(documents: Sequence[Document]) -> list[str]:
    """Legacy helper: Stores chunks into local JSON store and syncs to PostgreSQL pgvector."""
    if not documents:
        return []

    texts = [d.page_content for d in documents]
    vectors = embed_texts(texts)

    entries = []
    doc_ids = []
    for doc, vec in zip(documents, vectors):
        chunk_id = doc.metadata.get("chunk_id") or str(uuid.uuid4())
        doc_ids.append(chunk_id)
        entries.append({
            "id": chunk_id,
            "document": doc.page_content,
            "cmetadata": doc.metadata,
            "embedding": vec,
        })

    await upsert_vector_store(entries)
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


# ============================================================
# 4. INCREMENTAL DOCUMENT INDEXING WITH HYBRID CHUNKING
# ============================================================

async def process_document_upload(
    file_bytes: bytes,
    file_name: str,
    db: AsyncSession,
    forced_doc_id: uuid.UUID | None = None,
) -> DocumentUploadResponse:
    """
    Incremental document indexing using hybrid chunking:
    1. Extracts text with page-number and heading structure tracking.
    2. Computes SHA-256 document content hash.
    3. Maintains Document registry.
    4. Applies hybrid chunking (heading detection, topic grouping, semantic sub-chunking, small overlap).
    5. Generates stable chunk IDs (doc_{id}_{sec}_{top}_{idx:03d}) and SHA-256 content hashes.
    6. Incremental comparison:
       - Same chunk ID + same hash -> SKIP embedding
       - Same chunk ID + different hash -> RE-EMBED & UPSERT
       - New chunk ID -> EMBED & INSERT
       - Old chunk ID missing in new document -> DELETE vector & chunk metadata
    7. Updates document status (UPLOADED -> PROCESSING -> INDEXED / UPDATED).
    """
    if not file_bytes:
        raise ExtractionError(f"Uploaded file '{file_name}' is empty (0 bytes).")

    # 1. Text & metadata extraction
    pages, doc_meta, doc_hash, file_type = extract_document_text(file_bytes, file_name)
    if not pages or not any(p.get("text", "").strip() for p in pages):
        raise ExtractionError(f"No readable text could be extracted from '{file_name}'.")

    # 2. Document Registry Lookup
    is_existing = False
    doc_id: uuid.UUID

    if forced_doc_id:
        stmt = select(DocumentModel).where(DocumentModel.document_id == forced_doc_id)
    else:
        stmt = select(DocumentModel).where(DocumentModel.file_name == file_name)

    res = await db.execute(stmt)
    existing_doc = res.scalar_one_or_none()

    if existing_doc:
        doc_id = existing_doc.document_id
        doc_record = existing_doc
        is_existing = True
    else:
        doc_id = forced_doc_id or uuid.uuid4()
        doc_record = DocumentModel(
            document_id=doc_id,
            file_name=file_name,
            file_hash=doc_hash,
            file_type=file_type,
            status="PROCESSING",
            chunk_count=0,
        )
        db.add(doc_record)
        await db.flush()

    # 3. Retrieve existing chunks for this document
    stmt_chunks = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    res_chunks = await db.execute(stmt_chunks)
    old_chunks = {c.chunk_id: c for c in res_chunks.scalars().all()}

    # Scenario 6: Exact same document hash and already indexed
    if is_existing and existing_doc.file_hash == doc_hash and len(old_chunks) > 0:
        logger.info("Document '%s' (id=%s) hash is unchanged; skipping all embeddings.", file_name, doc_id)
        return DocumentUploadResponse(
            document_id=doc_id,
            file_name=file_name,
            file_hash=doc_hash,
            file_type=file_type,
            status=existing_doc.status,
            total_chunks=len(old_chunks),
            chunks_added=0,
            chunks_updated=0,
            chunks_skipped=len(old_chunks),
            chunks_deleted=0,
            message=f"Document '{file_name}' is unchanged. 0 chunks re-embedded, {len(old_chunks)} skipped.",
        )

    doc_record.status = "PROCESSING"
    await db.flush()

    # 4. Hybrid chunking
    hybrid_chunks = hybrid_chunk_pages(
        pages_data=pages,
        document_id=doc_id,
        file_name=file_name,
    )
    new_chunks_map = {c.chunk_id: c for c in hybrid_chunks}

    # 5. Incremental Diff Comparison
    to_skip: list[HybridChunk] = []
    to_update: list[HybridChunk] = []
    to_insert: list[HybridChunk] = []
    to_delete_ids: list[str] = []

    for cid, c in new_chunks_map.items():
        if cid in old_chunks:
            if old_chunks[cid].content_hash == c.content_hash:
                to_skip.append(c)
            else:
                to_update.append(c)
        else:
            to_insert.append(c)

    for old_cid in old_chunks:
        if old_cid not in new_chunks_map:
            to_delete_ids.append(old_cid)

    logger.info(
        "Incremental diff for '%s': %d to insert, %d to update, %d to skip, %d to delete.",
        file_name,
        len(to_insert),
        len(to_update),
        len(to_skip),
        len(to_delete_ids),
    )

    # 6. Execute Deletions
    if to_delete_ids:
        await delete_from_vector_store(to_delete_ids)
        del_stmt = delete(DocumentChunk).where(DocumentChunk.chunk_id.in_(to_delete_ids))
        await db.execute(del_stmt)

    # 7. Embed & Upsert ONLY affected chunks (to_insert + to_update)
    chunks_to_embed = to_insert + to_update
    if chunks_to_embed:
        texts_to_embed = [c.content for c in chunks_to_embed]
        vectors = embed_texts(texts_to_embed)

        vector_entries = []
        for c, vec in zip(chunks_to_embed, vectors):
            vector_entries.append({
                "id": c.chunk_id,
                "document": c.content,
                "cmetadata": {
                    "document_id": str(doc_id),
                    "file_name": file_name,
                    "filename": file_name,
                    "chunk_id": c.chunk_id,
                    "section": c.section,
                    "topic": c.topic,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "page": c.page_number,
                    "content_hash": c.content_hash,
                },
                "embedding": vec,
            })
        await upsert_vector_store(vector_entries)

        now = datetime.now(timezone.utc)
        for c in to_update:
            old_c = old_chunks[c.chunk_id]
            old_c.content = c.content
            old_c.content_hash = c.content_hash
            old_c.section = c.section
            old_c.topic = c.topic
            old_c.page_number = c.page_number
            old_c.chunk_index = c.chunk_index
            old_c.updated_at = now

        for c in to_insert:
            new_chunk_row = DocumentChunk(
                chunk_id=c.chunk_id,
                document_id=doc_id,
                section=c.section,
                topic=c.topic,
                chunk_index=c.chunk_index,
                content=c.content,
                content_hash=c.content_hash,
                page_number=c.page_number,
                created_at=now,
                updated_at=now,
            )
            db.add(new_chunk_row)

    # 8. Update Document record
    doc_record.file_hash = doc_hash
    doc_record.status = "UPDATED" if is_existing else "INDEXED"
    doc_record.chunk_count = len(hybrid_chunks)
    doc_record.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(doc_record)

    return DocumentUploadResponse(
        document_id=doc_record.document_id,
        file_name=doc_record.file_name,
        file_hash=doc_record.file_hash,
        file_type=doc_record.file_type,
        status=doc_record.status,
        total_chunks=len(hybrid_chunks),
        chunks_added=len(to_insert),
        chunks_updated=len(to_update),
        chunks_skipped=len(to_skip),
        chunks_deleted=len(to_delete_ids),
        message=f"Indexed '{file_name}': {len(to_insert)} added, {len(to_update)} updated, {len(to_skip)} skipped, {len(to_delete_ids)} deleted.",
    )


async def get_documents(db: AsyncSession) -> list[DocumentModel]:
    """Returns all registered documents ordered by creation time."""
    stmt = select(DocumentModel).order_by(DocumentModel.created_at.desc())
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def get_document_detail(document_id: uuid.UUID, db: AsyncSession) -> tuple[DocumentModel | None, list[DocumentChunk]]:
    """Returns document record and its indexed chunks."""
    stmt = select(DocumentModel).where(DocumentModel.document_id == document_id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        return None, []
    stmt_chunks = select(DocumentChunk).where(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index)
    res_chunks = await db.execute(stmt_chunks)
    return doc, list(res_chunks.scalars().all())


async def delete_document(document_id: uuid.UUID, db: AsyncSession) -> bool:
    """Deletes document, its chunks, and associated vector representations."""
    stmt = select(DocumentModel).where(DocumentModel.document_id == document_id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        return False

    stmt_chunks = select(DocumentChunk.chunk_id).where(DocumentChunk.document_id == document_id)
    res_chunks = await db.execute(stmt_chunks)
    chunk_ids = list(res_chunks.scalars().all())

    if chunk_ids:
        await delete_from_vector_store(chunk_ids)

    await db.delete(doc)
    await db.commit()
    return True


async def index_company_policy(force_reindex: bool = False) -> int:
    """Indexes company policy on application startup into both vector store and document registry."""
    path = get_policy_file_path()
    if not path.exists():
        logger.error("Policy file not found at: %s", path)
        return 0

    policy_bytes = path.read_bytes()

    try:
        from src.database.connection import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            stmt = select(DocumentModel).where(DocumentModel.document_id == POLICY_DOC_ID)
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()
            if not force_reindex and existing and existing.chunk_count > 0:
                logger.info("Company policy already registered in DB (%d chunks).", existing.chunk_count)
                return existing.chunk_count

            upload_res = await process_document_upload(
                file_bytes=policy_bytes,
                file_name=path.name,
                db=session,
                forced_doc_id=POLICY_DOC_ID,
            )
            return upload_res.total_chunks
    except Exception as e:
        logger.warning("Database unavailable for policy registration (%s). Fallback indexing...", e)
        # Fallback to local vector store
        policy_text = path.read_text(encoding="utf-8")
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
# 5. CONTEXTUAL RETRIEVAL & MULTI-DOCUMENT VECTOR SEARCH
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
    document_id: uuid.UUID | None = None,
) -> list[Document]:
    """
    Retrieves semantically relevant document chunks using dense vector embeddings + cosine similarity.
    Searches across ALL indexed documents by default, or filters by document_id when specified.
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
            if document_id:
                res = await conn.execute(
                    text("SELECT id, cmetadata, document, embedding FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id"),
                    {"doc_id": str(document_id)},
                )
            else:
                res = await conn.execute(
                    text("SELECT id, cmetadata, document, embedding FROM langchain_pg_embedding")
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
            if document_id and cmetadata.get("document_id") != str(document_id):
                continue
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
        filename = doc.metadata.get("filename") or doc.metadata.get("file_name", POLICY_FILENAME)
        page = doc.metadata.get("page") or doc.metadata.get("page_number", 1)
        section = doc.metadata.get("section", "")
        topic = doc.metadata.get("topic", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        score = doc.metadata.get("score", "N/A")

        header_parts = [f"Document: {filename}"]
        if section and section != "Overview":
            header_parts.append(f"Section: {section}")
        if topic and topic not in ("General", "Details", "Introduction"):
            header_parts.append(f"Topic: {topic}")
        header_parts.append(f"Page: {page}")
        if chunk_id:
            header_parts.append(f"Chunk: {chunk_id}")
        header_parts.append(f"Similarity: {score}")

        blocks.append(f"--- [{' | '.join(header_parts)}] ---\n{doc.page_content.strip()}")
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
    document_id: uuid.UUID | None = None,
) -> tuple[str, list[Document], bool]:
    """Executes pure LangChain LCEL RAG chain."""
    chunks = await retrieve_relevant_chunks(
        query=question,
        chat_history=chat_history,
        document_id=document_id,
    )
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

        doc_filter = None
        if request.document_id:
            try:
                doc_filter = uuid.UUID(request.document_id)
            except Exception:
                pass

        # 1. Record user message
        try:
            await self.add_message(role="user", content=request.message, document_id=doc_filter)
        except Exception as e:
            logger.warning("Failed to save incoming user message: %s", e)

        # 2. Generate answer via LangChain RAG
        try:
            answer, matching_docs, show_pdf = await generate_rag_answer(
                question=request.message,
                chat_history=recent_history,
                document_id=doc_filter,
            )
        except Exception as e:
            logger.error("RAG pipeline failed: %s", e, exc_info=True)
            answer = "⚠️ An error occurred while generating the answer. Please try again shortly."
            matching_docs = []
            show_pdf = True

        seen_keys: set[str] = set()
        sources: list[SourceChunk] = []
        for d in matching_docs:
            fname = d.metadata.get("filename") or d.metadata.get("file_name", POLICY_FILENAME)
            pg = int(d.metadata.get("page") or d.metadata.get("page_number", 1))
            sec = d.metadata.get("section")
            top = d.metadata.get("topic")
            cid = d.metadata.get("chunk_id")
            c_idx = int(d.metadata.get("chunk_index", 0))

            dedup_key = cid or f"{fname}_{pg}_{sec}_{top}_{c_idx}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                sources.append(
                    SourceChunk(
                        filename=fname,
                        page=pg,
                        chunk_index=c_idx,
                        section=sec,
                        topic=top,
                        chunk_id=cid,
                    )
                )

        if show_pdf and not sources:
            sources.append(SourceChunk(filename=POLICY_FILENAME, page=1, chunk_index=0))

        # 3. Record assistant answer
        try:
            await self.add_message(role="assistant", content=answer, document_id=doc_filter)
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
