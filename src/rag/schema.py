"""
Pydantic schemas for chat requests/responses and document management.
"""
import uuid
from typing import Any
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    filename: str
    page: int
    chunk_index: int
    section: str | None = None
    topic: str | None = None
    chunk_id: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Question about company policy to ask")
    session_id: str | None = Field(default=None, description="Optional session identifier")
    document_id: str | None = Field(default=None, description="Optional document filter")
    history: list[dict[str, Any]] = Field(default=[], description="Optional conversational history turns")
    chat_history: list[dict[str, Any]] | None = Field(default=None, description="Alias for conversational history turns")


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []
    session_id: str | None = None
    show_pdf: bool = False


class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: Any = None
    session_id: str | None = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    file_name: str
    file_hash: str
    file_type: str
    status: str  # "UPLOADED", "PROCESSING", "INDEXED", "UPDATED", "FAILED"
    total_chunks: int
    chunks_added: int
    chunks_updated: int
    chunks_skipped: int
    chunks_deleted: int
    message: str


class DocumentInfoResponse(BaseModel):
    document_id: uuid.UUID
    file_name: str
    file_hash: str
    file_type: str
    status: str
    chunk_count: int
    is_default: bool = False
    created_at: Any = None
    updated_at: Any = None

    model_config = {"from_attributes": True}


class DocumentDetailResponse(BaseModel):
    document: DocumentInfoResponse
    chunks: list[dict[str, Any]] = []
