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

from fastapi import HTTPException
from src.core.config import settings
from src.database.connection import engine
from src.database.qdrant import chunk_id_to_qdrant_id, qdrant_service
from qdrant_client.models import PointStruct
from src.rag.model import (
    ChatMessage,
    ChatSession,
    DEFAULT_DOC_ID,
    Document as DocumentModel,
    DocumentChunk,
)
from src.rag.schema import (
    ChatRequest,
    ChatResponse,
    DocumentDetailResponse,
    DocumentInfoResponse,
    DocumentUploadResponse,
    SourceChunk,
)

logger = logging.getLogger("src.rag.service")

POLICY_DOC_ID = uuid.UUID(settings.policy_doc_id)
POLICY_FILENAME = Path(settings.policy_file_path).name

# ============================================================
# 1. DOCUMENT EXTRACTION UTILITIES (PDF, DOCX, TXT)
# ============================================================

class ExtractionError(Exception):
    """Raised when text extraction from a file fails."""
    pass


def normalize_extracted_pdf_text(text: str) -> str:
    """
    Normalizes text extracted from PDF pages.
    PDF text extractors often emit words separated by newlines (\n or \n\n)
    or break sentences unnaturally. This function un-wraps fragmented words
    into clean, flowing sentences and paragraphs while preserving legitimate
    bullet points and headings.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not raw_lines:
        return ""

    output_blocks: list[str] = []
    current_block: list[str] = []
    pending_bullet = False

    for line in raw_lines:
        # Standalone bullet symbol
        if line in ("●", "○", "•", "■", "◆", "►"):
            if current_block:
                output_blocks.append(" ".join(current_block))
                current_block = []
            pending_bullet = True
            continue

        is_bullet = line.startswith(("●", "○", "•", "■", "◆", "►")) or bool(re.match(r"^\d+[\.\)]\s", line))
        is_heading = (
            line.startswith("#")
            or line.startswith("§")
            or (line.endswith(":") and len(line.split()) <= 8)
            or bool(re.match(r"^(?:§\s*|Section\s+|Article\s+|Chapter\s+)?\d+[:.\-\s]+[A-Za-z]", line, re.I))
        )

        if pending_bullet:
            pending_bullet = False
            line = "• " + line.lstrip("●○•■◆►- ")
            is_bullet = True

        if is_bullet:
            if current_block:
                output_blocks.append(" ".join(current_block))
                current_block = []
            current_block.append(line)
        elif is_heading:
            if current_block:
                output_blocks.append(" ".join(current_block))
                current_block = []
            output_blocks.append(line)
        else:
            current_block.append(line)

    if current_block:
        output_blocks.append(" ".join(current_block))

    final_blocks: list[str] = []
    for b in output_blocks:
        b_clean = b.strip()
        if not b_clean:
            continue
        b_clean = re.sub(r"^[●○■◆►]\s*", "• ", b_clean)
        b_clean = re.sub(r"\s+\d+$", "", b_clean)
        final_blocks.append(b_clean)

    return "\n\n".join(final_blocks)


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
            clean_text = normalize_extracted_pdf_text(page_text)
            pages_data.append({
                "page_number": idx + 1,
                "text": clean_text if clean_text else page_text,
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


INVALID_TOPIC_WORDS = {
    "are", "is", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    "ex", "eg", "e.g", "e.g.", "ie", "i.e", "i.e.", "etc", "etc.",
    "case", "example", "examples", "note", "notes",
    "and", "or", "but", "for", "nor", "so", "yet",
    "to", "of", "in", "on", "at", "by", "with", "from", "as",
    "the", "a", "an", "this", "that", "these", "those",
    "such as", "as follows", "including", "given under",
    "details", "general", "overview", "introduction", "section", "part",
    "30-day", "day", "days", "clause",
}


def is_valid_topic_name(title: str | None) -> bool:
    """Checks if a string is a legitimate semantic topic or section name."""
    if not title:
        return False
    clean = title.strip().rstrip(":")
    if len(clean) < 3:
        return False
    # Pure numbers or symbols
    if clean.isdigit() or re.match(r"^[\d.\-_/\\:]+$", clean):
        return False
    clean_lower = clean.lower()
    if clean_lower in INVALID_TOPIC_WORDS:
        return False
    words = clean_lower.split()
    if len(words) == 1 and words[0] in INVALID_TOPIC_WORDS:
        return False
    if words and words[0] in ("are", "is", "was", "were", "and", "or", "to", "for", "in", "with", "by", "from"):
        return False
    return True


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
        if is_valid_topic_name(title):
            return True, level, title

    # 2. Numbered headings (e.g. '1. Working Hours', '§ 5. Health Insurance', 'Section 2: Remote Work', 'Article 3 - Leave')
    numbered_match = re.match(
        r"^(?:§\s*|Section\s+|Article\s+)?(\d+(?:\.\d+)*)[:.\-\s]+\s*([A-Za-z].+)$",
        stripped,
        re.IGNORECASE,
    )
    if numbered_match:
        title = numbered_match.group(2).strip().rstrip(":")
        if is_valid_topic_name(title):
            num_parts = numbered_match.group(1).split(".")
            level = min(len(num_parts), 4)
            return True, level, f"{numbered_match.group(1)}. {title}"

    # 3. Standalone UPPERCASE heading (at least 8 chars, not ending in period)
    if stripped.isupper() and len(stripped) >= 8 and not stripped.endswith((".", ";")):
        title = stripped.rstrip(":").title()
        if is_valid_topic_name(title):
            return True, 1, title

    # 4. Heading ending with colon without terminal period and reasonable length
    if stripped.endswith(":") and len(stripped.split()) <= 8 and not any(p in stripped for p in [".", "?", "!"]):
        raw_title = stripped.rstrip(":").strip()
        if is_valid_topic_name(raw_title) and (raw_title[0].isupper() or raw_title[0].isdigit()):
            # Substantial headings (<= 5 words) get level 1, longer get level 2
            return True, 1 if len(raw_title.split()) <= 5 else 2, raw_title

    # 5. Standalone Title Case line without terminal punctuation (e.g. 'Flight Booking Policy', 'Hotel Reimbursement Limits')
    words = stripped.split()
    if (
        2 <= len(words) <= 6
        and not stripped.endswith((".", ";", ",", "?", "!"))
        and all(w[0].isupper() or w.lower() in ("and", "or", "of", "in", "to", "for", "the", "a", "an", "&") for w in words if w)
        and is_valid_topic_name(stripped)
    ):
        return True, 1, stripped

    return False, 0, ""


def extract_topic_from_content(
    content: str,
    default_topic: str,
    section_name: str,
) -> str:
    """
    Extracts or refines the most specific, accurate topic for a chunk.
    If default_topic is invalid or generic (e.g. 'are', 'ex', 'General', 'Details'),
    scans content lines to extract a genuine topic.
    """
    if is_valid_topic_name(default_topic) and default_topic not in ("General", "Details", "Introduction"):
        return default_topic

    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if not lines:
        return section_name if is_valid_topic_name(section_name) else "General"

    # 1. Check for section symbol § in lines
    for line in lines[:8]:
        sec_sym_search = re.search(r"§\s*(\d+)[:.\-\s]+\s*([A-Za-z][A-Za-z0-9\s/&()\-]+?)(?:\s*\||\s*--|\.\s|$)", line)
        if sec_sym_search:
            candidate = sec_sym_search.group(2).strip().rstrip(":")
            if is_valid_topic_name(candidate) and len(candidate.split()) <= 6:
                return candidate

    # 1. Check for bullet headers: e.g. "• Casual Leave (CL)" or "• Sick Leave (SL)"
    for line in lines[:8]:
        bullet_m = re.match(
            r"^[-*•\d.]+\s*(?:\*\*(.+?)\*\*|([A-Z][A-Za-z0-9\s/&()\-]+?)(?::|$))\s*",
            line,
        )
        if bullet_m:
            candidate = (bullet_m.group(1) or bullet_m.group(2) or "").strip().rstrip(":")
            if is_valid_topic_name(candidate) and len(candidate.split()) <= 5:
                return candidate

        is_h, _, h_title = is_heading(line)
        if is_h and is_valid_topic_name(h_title):
            return h_title

    # 2. Check for sentence subject in first line:
    first_line = lines[0]
    subject_m = re.match(
        r"^([A-Z][A-Za-z0-9\s/&()\-]{3,35})\s+(?:is|are|will|must|should|can|cannot|applies|covers|refers)\b",
        first_line,
    )
    if subject_m:
        cand = subject_m.group(1).strip()
        if is_valid_topic_name(cand) and len(cand.split()) <= 4:
            return cand

    # Fall back to section name if valid, otherwise "General"
    if is_valid_topic_name(section_name) and section_name not in ("Overview", "Document"):
        return section_name

    return "General"


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

            # Refine topic and ensure clean semantic hierarchy
            refined_topic = extract_topic_from_content(block_text, topic_name, section_name)
            refined_section = section_name if is_valid_topic_name(section_name) else (refined_topic or "Overview")

            sec_slug = slugify(refined_section, max_words=3)
            top_slug = slugify(refined_topic, max_words=3)
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
                    "section": refined_section,
                    "topic": refined_topic,
                    "chunk_index": global_chunk_idx,
                    "page": page,
                    "content_hash": c_hash,
                }

                hybrid_chunks.append(
                    HybridChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        section=refined_section,
                        topic=refined_topic,
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

                    # Further refine topic for sub-chunk if it focuses on a specific sub-clause
                    sub_topic = extract_topic_from_content(sub_t_clean, refined_topic, refined_section)
                    sub_top_slug = slugify(sub_topic, max_words=3)
                    sub_group_key = f"{sec_slug}_{sub_top_slug}"

                    topic_counter = topic_chunk_counters.get(sub_group_key, 0) + 1
                    topic_chunk_counters[sub_group_key] = topic_counter

                    chunk_id = f"doc_{short_doc_id}_{sec_slug}_{sub_top_slug}_{topic_counter:03d}"
                    c_hash = calculate_content_hash(sub_t_clean)

                    meta = {
                        "document_id": str(document_id),
                        "filename": file_name,
                        "chunk_id": chunk_id,
                        "section": refined_section,
                        "topic": sub_topic,
                        "chunk_index": global_chunk_idx,
                        "page": page,
                        "content_hash": c_hash,
                    }

                    hybrid_chunks.append(
                        HybridChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            section=refined_section,
                            topic=sub_topic,
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
            if is_head and is_valid_topic_name(title):
                flush_block(current_section, current_topic, page_num)

                # Major heading updates both section & topic; sub-heading updates topic
                if level == 1 or current_section in ("Overview", "Document"):
                    current_section = title
                    current_topic = title
                else:
                    current_topic = title

                current_block.append(trimmed)
            else:
                bullet_clause = re.match(r"^[-*•\d.]+\s*(?:\*\*(.+?)\*\*|([A-Za-z0-9\s/&]+):)\s*(.+)$", trimmed)
                if bullet_clause and len(current_block) > 4:
                    item_topic = (bullet_clause.group(1) or bullet_clause.group(2) or "").strip()
                    if is_valid_topic_name(item_topic) and len(item_topic.split()) <= 4:
                        flush_block(current_section, current_topic, page_num)
                        current_topic = item_topic

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
# 3. VECTOR STORAGE & INDEXING (QDRANT CLOUD)
# ============================================================

async def upsert_vector_store(entries: list[dict[str, Any]]) -> None:
    """
    Upserts vector entries with stable chunk IDs and metadata payloads into Qdrant Cloud.
    """
    if not entries:
        return

    if not qdrant_service.is_configured():
        logger.warning("Qdrant Cloud is not configured; skipping vector upsert.")
        return

    points = []
    for entry in entries:
        cid = str(entry["id"])
        qid = chunk_id_to_qdrant_id(cid)
        meta = entry.get("cmetadata") or {}
        points.append(
            PointStruct(
                id=qid,
                vector=entry["embedding"],
                payload={
                    "document_id": str(meta.get("document_id", "")),
                    "chunk_id": cid,
                    "content": entry.get("document") or entry.get("content", ""),
                    "file_name": meta.get("file_name") or meta.get("filename", ""),
                    "section": meta.get("section"),
                    "topic": meta.get("topic"),
                    "page_number": int(meta.get("page_number") or meta.get("page") or 1),
                    "content_hash": str(meta.get("content_hash", "")),
                },
            )
        )
    try:
        qdrant_service.upsert_points(points)
        logger.info(
            "Upserted %d vector points to Qdrant Cloud collection '%s'.",
            len(points),
            qdrant_service.collection_name,
        )
    except Exception as e:
        logger.error("Failed to upsert points into Qdrant Cloud: %s", e)
        raise


async def delete_from_vector_store(chunk_ids: Sequence[str]) -> None:
    """
    Deletes specified chunk IDs from Qdrant Cloud.
    """
    if not chunk_ids:
        return

    if qdrant_service.is_configured():
        try:
            qdrant_service.delete_points(chunk_ids)
            logger.info("Deleted %d points from Qdrant Cloud.", len(chunk_ids))
        except Exception as e:
            logger.error("Failed to delete points from Qdrant Cloud: %s", e)


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
    if forced_doc_id == POLICY_DOC_ID or file_name == POLICY_FILENAME:
        # Use pristine structured POLICY_PAGES directly to ensure exact section and topic metadata
        pages = []
        for p in POLICY_PAGES:
            pg_text_lines = []
            for s in p.get("sections", []):
                pg_text_lines.append(f"§ {s['num']}. {s['title']}")
                if s.get("intro"):
                    pg_text_lines.append(s["intro"])
                for b in s.get("bullets", []):
                    pg_text_lines.append(f"• {b}")
                pg_text_lines.append("")
            pages.append({
                "page_number": p["page"],
                "text": "\n".join(pg_text_lines),
            })

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
        try:
            await upsert_vector_store(vector_entries)
        except Exception as e:
            logger.error("Vector database upsert failed for '%s': %s", file_name, e)
            doc_record.status = "FAILED"
            await db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Vector indexing failed for '{file_name}'. Document marked as FAILED and can be retried.",
            ) from e

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
    if document_id == POLICY_DOC_ID:
        raise HTTPException(
            status_code=400,
            detail="The default system policy document is protected and cannot be deleted.",
        )

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

    if qdrant_service.is_configured():
        try:
            qdrant_service.delete_by_document(document_id)
            if doc.file_name:
                qdrant_service.delete_by_filename(doc.file_name)
        except Exception as e:
            logger.error("Failed to delete document from Qdrant: %s", e)

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
                if qdrant_service.is_configured():
                    q_health = qdrant_service.health_check()
                    if q_health.get("points_count", 0) == 0:
                        logger.info("Synchronizing existing PostgreSQL chunks to Qdrant Cloud...")
                        stmt_chunks = select(DocumentChunk).where(DocumentChunk.document_id == POLICY_DOC_ID)
                        res_chunks = await session.execute(stmt_chunks)
                        db_chunks = list(res_chunks.scalars().all())
                        if db_chunks:
                            texts = [c.content for c in db_chunks]
                            vecs = embed_texts(texts)
                            entries = []
                            for c, v in zip(db_chunks, vecs):
                                entries.append({
                                    "id": c.chunk_id,
                                    "embedding": v,
                                    "document": c.content,
                                    "cmetadata": {
                                        "document_id": str(c.document_id),
                                        "chunk_id": c.chunk_id,
                                        "section": c.section,
                                        "topic": c.topic,
                                        "page_number": c.page_number,
                                        "content_hash": c.content_hash,
                                    },
                                })
                            await upsert_vector_store(entries)
                            logger.info("Successfully synced %d chunks to Qdrant Cloud.", len(entries))
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

GREETING_WORDS = {
    "hi", "hello", "hey", "hola", "namaste", "greetings", "good morning", "good evening",
    "good afternoon", "howdy", "sup", "what's up", "whats up", "how are you", "who are you",
    "thanks", "thank you", "thx", "bye", "goodbye", "ok", "okay", "cool", "nice", "great",
}


def is_greeting(query: str) -> bool:
    """Checks if the query is a simple greeting, pleasantry, or conversational acknowledgment."""
    if not query:
        return False
    cleaned = re.sub(r"[^\w\s]", "", query).strip().lower()
    if cleaned in GREETING_WORDS:
        return True
    return any(
        cleaned.startswith(g + " ")
        for g in ("hi", "hello", "hey", "good morning", "good evening", "good afternoon")
    )


def is_dependent_followup(query: str) -> bool:
    """Checks if a query is an anaphoric follow-up dependent on prior conversation context."""
    cleaned = query.strip().lower()
    if not cleaned:
        return False
    # Connectors indicating continuation of prior topic
    connectors = (
        "and ", "also ", "what about", "how about", "what if", "can i also",
        "does it", "is it", "who is", "why is", "why does", "how long", "how much",
        "tell me more", "explain more", "give more", "which one",
    )
    if any(cleaned.startswith(c) for c in connectors):
        return True
    # Pronouns that reference earlier entities
    tokens = set(re.findall(r"\b\w+\b", cleaned))
    pronouns = {"it", "its", "this", "that", "these", "those", "them", "they", "same"}
    if tokens.intersection(pronouns) and len(tokens) <= 7:
        return True
    return False


def contextualize_query(query: str, chat_history: Sequence[Any] | None = None) -> str:
    """Contextualizes brief follow-up queries with recent conversation context."""
    if not chat_history or is_greeting(query):
        return query

    cleaned = query.strip()
    if not is_dependent_followup(cleaned):
        return query

    last_user_query = None
    for msg in reversed(chat_history):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "") or "user"
        content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
        if role in ("user", "human") and content.strip():
            # Never contextualize using previous greetings
            if not is_greeting(content.strip()):
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
    Retrieves semantically relevant document chunks using Qdrant Cloud vector search as
    the dedicated vector database, fetching authoritative chunk content from PostgreSQL
    (the source-of-truth database), with seamless fallback to pgvector/local stores.
    """
    if is_greeting(query):
        return []

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

    # 1. Primary Vector Search: Qdrant Cloud
    if qdrant_service.is_configured():
        try:
            filters = {}
            if document_id:
                filters["document_id"] = str(document_id)

            # When querying a specific targeted document, relax threshold so we retrieve its content
            search_threshold = 0.05 if document_id else threshold

            qdrant_results = qdrant_service.search(
                query_vector=q_vec,
                limit=k * 3,
                filters=filters if filters else None,
                score_threshold=search_threshold,
            )

            if qdrant_results:
                chunk_ids = [
                    pt.payload.get("chunk_id")
                    for pt in qdrant_results
                    if pt.payload and pt.payload.get("chunk_id")
                ]

                # Fetch authoritative content and metadata from PostgreSQL (source of truth)
                db_fetch_attempted = False
                db_chunks: dict[str, DocumentChunk] = {}
                doc_names: dict[str, str] = {}
                orphan_chunk_ids: list[str] = []
                try:
                    from src.database.connection import AsyncSessionLocal
                    async with AsyncSessionLocal() as session:
                        if chunk_ids:
                            res_c = await session.execute(
                                select(DocumentChunk).where(DocumentChunk.chunk_id.in_(chunk_ids))
                            )
                            for chk in res_c.scalars().all():
                                db_chunks[chk.chunk_id] = chk

                            d_ids = {chk.document_id for chk in db_chunks.values()}
                            if d_ids:
                                res_d = await session.execute(
                                    select(DocumentModel.document_id, DocumentModel.file_name).where(
                                        DocumentModel.document_id.in_(list(d_ids))
                                    )
                                )
                                for d_row in res_d.all():
                                    doc_names[str(d_row[0])] = d_row[1]
                        db_fetch_attempted = True
                except Exception as db_err:
                    logger.warning("PostgreSQL fetch for chunk content skipped (%s). Using payload fallback.", db_err)

                for pt in qdrant_results:
                    payload = pt.payload or {}
                    cid = payload.get("chunk_id")
                    if not cid:
                        continue
                    score = float(pt.score)

                    # Authoritative PostgreSQL content
                    if db_fetch_attempted:
                        if cid not in db_chunks:
                            # Not in primary database (deleted or orphan) -> ignore & queue for Qdrant prune
                            orphan_chunk_ids.append(cid)
                            continue
                        c_rec = db_chunks[cid]
                        content = c_rec.content
                        doc_id_str = str(c_rec.document_id)
                        fname = doc_names.get(doc_id_str, payload.get("file_name", POLICY_FILENAME))
                        sec = c_rec.section or payload.get("section")
                        top = c_rec.topic or payload.get("topic")
                        pg = c_rec.page_number or payload.get("page_number", 1)
                        c_idx = c_rec.chunk_index
                        c_hash = c_rec.content_hash or payload.get("content_hash", "")
                    else:
                        content = payload.get("content") or payload.get("document", "")
                        fname = payload.get("file_name", POLICY_FILENAME)
                        doc_id_str = str(payload.get("document_id") or "")
                        sec = payload.get("section")
                        top = payload.get("topic")
                        pg = int(payload.get("page_number", 1))
                        c_idx = int(payload.get("chunk_index", 0))
                        c_hash = payload.get("content_hash", "")

                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={
                                "document_id": doc_id_str or payload.get("document_id"),
                                "filename": fname,
                                "file_name": fname,
                                "chunk_id": cid,
                                "section": sec,
                                "topic": top,
                                "page": pg,
                                "page_number": pg,
                                "chunk_index": c_idx,
                                "content_hash": c_hash,
                                "score": round(score, 4),
                            },
                        )
                        scored_chunks.append((score, doc))

                if orphan_chunk_ids and qdrant_service.is_configured():
                    try:
                        qdrant_service.delete_points(orphan_chunk_ids)
                        logger.info("Auto-pruned %d orphan points from Qdrant Cloud.", len(orphan_chunk_ids))
                    except Exception as prune_err:
                        logger.warning("Failed to auto-prune orphan points from Qdrant: %s", prune_err)
        except Exception as e:
            logger.error("Qdrant similarity search encountered an error: %s. Falling back to secondary stores.", e)

    if not scored_chunks and document_id:
        try:
            from src.database.connection import AsyncSessionLocal
            async with AsyncSessionLocal() as session:
                res = await session.execute(
                    select(DocumentChunk, DocumentModel.file_name)
                    .join(DocumentModel, DocumentModel.document_id == DocumentChunk.document_id)
                    .where(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                    .limit(k)
                )
                for chk_row in res.all():
                    c_rec, fname = chk_row[0], chk_row[1]
                    doc = Document(
                        page_content=c_rec.content,
                        metadata={
                            "document_id": str(c_rec.document_id),
                            "filename": fname,
                            "file_name": fname,
                            "chunk_id": c_rec.chunk_id,
                            "section": c_rec.section,
                            "topic": c_rec.topic,
                            "page": c_rec.page_number,
                            "page_number": c_rec.page_number,
                            "chunk_index": c_rec.chunk_index,
                            "content_hash": c_rec.content_hash,
                            "score": 0.5,
                        },
                    )
                    scored_chunks.append((0.5, doc))
        except Exception as db_fallback_err:
            logger.warning("Target document fallback fetch error: %s", db_fallback_err)

    if not scored_chunks:
        return []

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    if document_id:
        # When specifically querying a targeted document, return its top chunks
        return [doc for score, doc in scored_chunks[:k]]

    if not scored_chunks or scored_chunks[0][0] < threshold:
        return []

    top_score = scored_chunks[0][0]
    effective_threshold = max(threshold, top_score - 0.12)
    filtered = [doc for score, doc in scored_chunks[:k] if score >= effective_threshold]

    # Filter out spurious chunks that have zero lexical relevance when similarity is mediocre (< 0.72)
    stop_words = {
        "what", "is", "are", "the", "a", "an", "in", "of", "to", "for", "on", "with",
        "about", "how", "why", "when", "where", "can", "could", "should", "would",
        "do", "does", "did", "please", "tell", "me", "give", "show", "i", "you", "we",
    }
    q_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2 and w not in stop_words}

    meaningful: list[Document] = []
    for doc in filtered:
        d_score = float(doc.metadata.get("score") or 0.0)
        if d_score >= 0.72:
            meaningful.append(doc)
            continue
        chunk_text = (
            doc.page_content + " " + (doc.metadata.get("section") or "") + " " + (doc.metadata.get("topic") or "")
        ).lower()
        if any(w in chunk_text for w in q_words):
            meaningful.append(doc)
        else:
            logger.info(
                "Discarded spurious chunk %s (score %.4f) with 0 keyword overlap for query '%s'",
                doc.metadata.get("chunk_id"),
                d_score,
                query,
            )

    return meaningful



# ============================================================
# 5. PROMPT TEMPLATES & LANGCHAIN LCEL CHAIN
# ============================================================

RAG_SYSTEM_PROMPT = """
You are WorkPilot's intelligent AI Assistant with access to the company's knowledge base and uploaded documents.
Your job is to understand the user's inquiry and provide an accurate, grounded, helpful, and concise answer based strictly on the retrieved knowledge base.

Guidelines:
- The retrieved knowledge base documents determine what is actually true.
- Answer the user's question directly using the facts, data, procedures, limits, and guidelines contained in the retrieved context.
- Bold key numbers, limits, deadlines, codes, and critical terms.
- When referencing information from specific documents, cite the document name or section naturally if helpful.
- If the requested information is not mentioned in the retrieved context, state that it is not specified in the available documents.

============================================================
RETRIEVED KNOWLEDGE BASE
============================================================
{context}

Answer the user's question directly adhering to all guidelines.
"""

GENERAL_SYSTEM_PROMPT = """
You are WorkPilot's intelligent AI Assistant.
For greetings or conversational pleasantries (such as "hi", "hello", "hey", "good morning"), respond warmly, politely, and naturally as an AI assistant ready to help with projects, tasks, and documents, without mentioning missing files.
For specific inquiries whose answers are not found in the available knowledge base or uploaded documents, provide a helpful general response or state that the specific detail is not documented in the current knowledge base and suggest checking with the relevant team or People Operations.
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
            is_policy_doc = any(
                (d.metadata.get("filename") == POLICY_FILENAME or str(d.metadata.get("document_id", "")) == str(POLICY_DOC_ID))
                for d in chunks
            )
            show_pdf = is_policy_doc and any(
                p in clean_answer.lower()
                for p in ["not specified", "not documented", "people operations", "policy handbook"]
            )
            return clean_answer, chunks, show_pdf
        except Exception as e:
            logger.error("LLM RAG invocation error: %s", e)
            return f"⚠️ Error generating answer from LLM: {str(e)}", chunks, False

    # Fallback when out of scope
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    fallback_chain = fallback_prompt | llm | output_parser
    try:
        answer = await fallback_chain.ainvoke({"question": question})
        return str(answer).strip(), [], False
    except Exception as e:
        logger.error("Fallback LLM error: %s", e)
        return (
            "I couldn't find information regarding that in the available knowledge base documents. Please consult People Operations or upload the relevant document.",
            [],
            False,
        )


# ============================================================
# 6. MAIN CHAT SERVICE & DATABASE PERSISTENCE
# ============================================================

async def detect_target_document_from_query(
    query: str,
    db: AsyncSession,
) -> uuid.UUID | None:
    """
    Detects if the user query specifically targets a particular document by name or alias.
    Handles numeric prefixes (e.g. '11.Bubble sort (ascending order).txt'), parentheses,
    and partial phrases (e.g. 'what bubble sort?').
    If so, returns that document's UUID to restrict retrieval and citations strictly to that document.
    """
    if not query:
        return None

    q_lower = query.lower()

    try:
        docs = await get_documents(db)
        if not docs:
            return None

        doc_scores: dict[uuid.UUID, int] = {}

        for doc in docs:
            fname = (doc.file_name or "").lower()
            is_default = (doc.document_id == POLICY_DOC_ID) or ("workpilot" in fname)

            if is_default:
                default_aliases = [
                    "workpilot",
                    "work pilot",
                    "company policy",
                    "company handbook",
                    "default policy",
                    "employee handbook",
                    "workpilot_company_policy",
                ]
                for alias in default_aliases:
                    if alias in q_lower:
                        doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 80 + len(alias))
            else:
                stem = fname.rsplit(".", 1)[0]
                # Strip leading numbering like '11.', '01_', '1 - '
                clean_stem = re.sub(r"^[\d\.\-_\s]+", "", stem).strip()
                # Normalize punctuation and parens to space
                normalized_stem = re.sub(r"[\(\)\[\]\{\}\-_,\.:;]+", " ", clean_stem).strip()
                words = [w for w in normalized_stem.split() if len(w) > 1]

                # Generate n-grams (bigrams and trigrams)
                bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
                trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]

                # Check exact and clean file stems
                if fname in q_lower or (len(stem) >= 3 and stem in q_lower):
                    doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 120)
                elif len(clean_stem) >= 3 and clean_stem in q_lower:
                    doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 110)
                elif len(normalized_stem) >= 3 and normalized_stem in q_lower:
                    doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 100)

                # Check trigrams
                for tri in trigrams:
                    if tri in q_lower and len(tri) >= 4:
                        doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 90 + len(tri))

                # Check bigrams (e.g. "bubble sort", "ms leave")
                for bi in bigrams:
                    if bi in q_lower and len(bi) >= 4:
                        doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 80 + len(bi))

                # Check first unique keyword combined with doc/pdf/file/policy/handbook
                if words and words[0] not in ("company", "policy", "the", "doc", "file"):
                    first = words[0]
                    if len(first) >= 3 and first in q_lower:
                        if any(kw in q_lower for kw in ("policy", "leave", "doc", "pdf", "handbook", "file")):
                            doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 75)
                        elif len(words) == 1 or len(first) >= 6:
                            doc_scores[doc.document_id] = max(doc_scores.get(doc.document_id, 0), 70)

        if not doc_scores:
            return None

        # Sort by score descending
        sorted_matches = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_id, top_score = sorted_matches[0]

        # Require a solid score (>= 75) to count as an explicit targeted document
        if top_score >= 75:
            if len(sorted_matches) > 1 and sorted_matches[1][1] == top_score:
                logger.info("Ambiguous document target query '%s'; multiple docs matched with equal score %d", query, top_score)
                return None
            return top_id

        return None
    except Exception as e:
        logger.warning("Error detecting target document from query: %s", e)
        return None


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
        session_id: str | None = None,
    ) -> ChatMessage:
        """Inserts and commits a new chat message into the database and tracks conversation sessions."""
        try:
            if session_id:
                stmt_s = select(ChatSession).where(ChatSession.id == session_id)
                res_s = await self.session.execute(stmt_s)
                sess = res_s.scalar_one_or_none()
                now = datetime.now(timezone.utc)
                if not sess:
                    title = content[:80].strip() if role == "user" else "Chat Session"
                    sess = ChatSession(
                        id=session_id,
                        title=title,
                        created_at=now,
                        updated_at=now,
                    )
                    self.session.add(sess)
                else:
                    sess.updated_at = now
                    if (not sess.title or sess.title == "Chat Session") and role == "user":
                        sess.title = content[:80].strip()

            msg = ChatMessage(
                document_id=document_id or DEFAULT_DOC_ID,
                session_id=session_id,
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
        session_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[ChatMessage]:
        """Fetches recent message history for a given session or document in chronological order."""
        try:
            stmt = select(ChatMessage)
            if session_id:
                stmt = stmt.where(ChatMessage.session_id == session_id)
            elif document_id:
                stmt = stmt.where(ChatMessage.document_id == document_id)
            else:
                stmt = stmt.where(ChatMessage.document_id == DEFAULT_DOC_ID)

            stmt = stmt.order_by(ChatMessage.created_at.desc()).limit(limit)
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

        # If no explicit document_id passed, auto-detect if the query specifically targets a single document
        if not doc_filter:
            target_id = await detect_target_document_from_query(request.message, self.session)
            if target_id:
                doc_filter = target_id
                logger.info("Targeted query exclusively to document %s for prompt: '%s'", target_id, request.message)

        # 1. Record user message
        try:
            await self.add_message(
                role="user",
                content=request.message,
                document_id=doc_filter,
                session_id=request.session_id,
            )
        except Exception as e:
            logger.warning("Failed to save incoming user message: %s", e)

        # Conversational greeting check: Dynamically generate greeting with zero document citations
        if is_greeting(request.message):
            try:
                answer, _, _ = await generate_rag_answer(
                    question=request.message,
                    chat_history=recent_history,
                    document_id=None,
                )
            except Exception as e:
                logger.error("Dynamic greeting generation failed: %s", e)
                answer = "Hello! How can I assist you today?"
            try:
                await self.add_message(
                    role="assistant",
                    content=answer,
                    document_id=None,
                    session_id=request.session_id,
                )
            except Exception as e:
                logger.warning("Failed to save greeting message: %s", e)
            return ChatResponse(
                answer=answer,
                sources=[],
                show_pdf=False,
                session_id=request.session_id,
            )

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

        if doc_filter and matching_docs:
            matching_docs = [
                d for d in matching_docs
                if str(d.metadata.get("document_id", "")) == str(doc_filter)
            ]

        # Detect if LLM stated the requested topic is not found / not documented
        is_negative_answer = any(
            phrase in answer.lower()
            for phrase in [
                "not specified in the retrieved knowledge base",
                "not documented in the current knowledge base",
                "not found in the available knowledge base",
                "not mentioned in the retrieved context",
                "couldn't find information regarding",
                "not specified in the available documents",
            ]
        )

        seen_keys: set[str] = set()
        sources: list[SourceChunk] = []

        # If it's a negative answer and no specific document was targeted,
        # do not attach spurious sources from unrelated knowledge base documents
        if not (is_negative_answer and not doc_filter):
            # Check which documents are explicitly referenced in the LLM answer
            referenced_filenames = set()
            answer_lower = answer.lower()
            unique_doc_names = {d.metadata.get("filename") or d.metadata.get("file_name", POLICY_FILENAME) for d in matching_docs}
            if len(unique_doc_names) > 1:
                for fn in unique_doc_names:
                    base_fn = re.sub(r"\.[^.]+$", "", fn).lower()
                    clean_fn = re.sub(r"^[\d.\-_ ]+", "", base_fn).strip()
                    if (
                        fn.lower() in answer_lower
                        or base_fn in answer_lower
                        or (len(clean_fn) >= 4 and clean_fn in answer_lower)
                        or ("company policy" in answer_lower and ("workpilot" in fn.lower() or fn == POLICY_FILENAME))
                        or ("handbook" in answer_lower and ("workpilot" in fn.lower() or fn == POLICY_FILENAME))
                    ):
                        referenced_filenames.add(fn)

            for d in matching_docs:
                fname = d.metadata.get("filename") or d.metadata.get("file_name", POLICY_FILENAME)
                # If the answer specifically cited certain document(s) by name, discard unrelated documents
                if referenced_filenames and fname not in referenced_filenames:
                    continue

                pg = int(d.metadata.get("page") or d.metadata.get("page_number", 1))
                sec = d.metadata.get("section")
                top = d.metadata.get("topic")
                cid = d.metadata.get("chunk_id")
                c_idx = int(d.metadata.get("chunk_index", 0))
                doc_id = str(d.metadata.get("document_id") or "")

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
                            document_id=doc_id if doc_id else None,
                        )
                    )

        if doc_filter and not sources:
            target_fname = "Document"
            try:
                res_doc = await self.session.execute(
                    select(DocumentModel.file_name).where(DocumentModel.document_id == doc_filter)
                )
                fn_val = res_doc.scalar_one_or_none()
                if fn_val:
                    target_fname = fn_val
            except Exception:
                pass
            sources.append(
                SourceChunk(
                    filename=target_fname,
                    page=1,
                    chunk_index=0,
                    document_id=str(doc_filter),
                )
            )
        elif show_pdf and not sources and not is_negative_answer:
            sources.append(
                SourceChunk(
                    filename=POLICY_FILENAME,
                    page=1,
                    chunk_index=0,
                    document_id=str(POLICY_DOC_ID),
                )
            )

        # 3. Record assistant answer
        try:
            await self.add_message(
                role="assistant",
                content=answer,
                document_id=doc_filter,
                session_id=request.session_id,
            )
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
        target_path = get_policy_file_path()
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

    doc_header_style = ParagraphStyle(
        "DocHeader",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    )
    doc_sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4F46E5"),
        spaceAfter=6,
    )
    sec_title_style = ParagraphStyle(
        "SecTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=4,
    )
    intro_style = ParagraphStyle(
        "IntroStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=14,
        spaceAfter=4,
    )
    page_footer_style = ParagraphStyle(
        "PageFooter",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
    )

    story = []
    for page_idx, page_data in enumerate(POLICY_PAGES):
        page_num = page_data["page"]
        story.append(Paragraph("WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK", doc_header_style))
        story.append(Paragraph("Official Human Resources & Operations Guidelines | Confidential & Proprietary", doc_sub_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4F46E5"), spaceAfter=10))

        for sec in page_data["sections"]:
            story.append(Paragraph(f"§ {sec['num']}. {sec['title']}", sec_title_style))
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
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
        story.append(Paragraph(f"Page {page_num} of 5 — WorkPilot Official Policy Handbook", page_footer_style))

        if page_idx < len(POLICY_PAGES) - 1:
            story.append(PageBreak())

    doc.build(story)
    return target_path
