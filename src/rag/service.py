"""
Service layer orchestrating chat history persistence, RAG pipeline execution,
and response structuring.
"""
import logging
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from src.langchain.chain import generate_rag_answer
from src.langchain.indexing import POLICY_FILENAME
from src.rag.model import ChatMessage
from src.rag.repository import ChatMessageRepository
from src.rag.schema import ChatRequest, ChatResponse, SourceChunk

logger = logging.getLogger("src.rag.service")


class ChatService:
    """
    Coordinates chat message persistence, RAG context retrieval,
    and LLM answer generation.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.chat_repo = ChatMessageRepository(session)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Processes a chat inquiry:
        1. Records the user message to conversation history.
        2. Executes the grounded RAG chain using the LLM.
        3. Persists the assistant's answer to the database.
        4. Returns the response with citations and handbook suggestions.
        """
        raw_history = request.chat_history if request.chat_history is not None else request.history
        recent_history = raw_history[-6:] if raw_history else []

        # 1. Record user message in history
        try:
            await self.chat_repo.add_message(
                role="user",
                content=request.message,
            )
        except Exception as e:
            logger.warning("Failed to record incoming user message to database: %s", e)

        # 2. Generate grounded answer via LangChain RAG Chain
        try:
            res = await generate_rag_answer(
                question=request.message,
                chat_history=recent_history,
            )
            answer = res[0]
            matching_docs = res[1]
            show_pdf = getattr(res, "show_pdf", False)
        except Exception as e:
            logger.error("RAG pipeline failed to execute: %s", e, exc_info=True)
            answer = (
                "⚠️ An error occurred while generating the answer from the AI assistant. "
                "Please check the server logs or try again shortly."
            )
            matching_docs = []
            show_pdf = True

        # Fallback check: if answer indicates information was not found or out of scope, show PDF
        if not show_pdf:
            ans_lower = answer.lower()
            if any(p in ans_lower for p in [
                "couldn't find",
                "could not find",
                "not documented in",
                "outside our documented",
                "outside the handbook",
                "consult people operations",
                "check with people operations",
                "reach out to people operations",
                "policy handbook",
            ]):
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
            sources.append(
                SourceChunk(
                    filename=POLICY_FILENAME,
                    page=1,
                    chunk_index=0,
                )
            )

        # 3. Record assistant response in history
        try:
            await self.chat_repo.add_message(
                role="assistant",
                content=answer,
            )
        except Exception as e:
            logger.warning("Failed to persist assistant message to database: %s", e)

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id,
            show_pdf=show_pdf,
        )

    async def get_history(self, limit: int = 50) -> Sequence[ChatMessage]:
        """
        Retrieves recent conversation history from the repository.
        """
        try:
            return await self.chat_repo.get_history(limit=limit)
        except Exception as e:
            logger.error("Error retrieving chat history: %s", e, exc_info=True)
            return []
