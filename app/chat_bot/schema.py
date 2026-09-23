import uuid
from typing import Any
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    filename: str
    page: int
    chunk_index: int


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Question about company policy to ask")
    session_id: str | None = Field(default=None, description="Optional session identifier")
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
    created_at: str | None = None
