from typing import Sequence
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat_bot.model import ChatMessage
from app.chat_bot.repository import ChatMessageRepository
from app.chat_bot.schema import ChatRequest, ChatResponse, SourceChunk
from app.rag.chain import generate_rag_answer
from app.rag.indexing import POLICY_FILENAME


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.chat_repo = ChatMessageRepository(session)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # 1. Fetch recent history for multi-turn conversational context
        raw_history = request.chat_history if request.chat_history is not None else request.history
        recent_history = raw_history[-6:] if raw_history else []

        # 2. Record user message in history
        await self.chat_repo.add_message(
            role="user",
            content=request.message,
        )

        # 3. Generate grounded answer via LangChain RAG Chain with context awareness
        res = await generate_rag_answer(
            question=request.message,
            chat_history=recent_history,
        )
        answer = res[0]
        matching_docs = res[1]
        show_pdf = getattr(res, "show_pdf", False)

        # Fallback check: if answer indicates information was not found in handbook or out of scope, show PDF
        if not show_pdf:
            ans_lower = answer.lower()
            if any(p in ans_lower for p in [
                "couldn't find",
                "could not find",
                "wasn't able to find",
                "not documented in",
                "outside our documented",
                "outside the handbook",
                "consult workplace operations",
                "consult people operations",
                "check with people operations",
                "reach out to people operations",
                "policy handbook (pdf)",
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

        # 4. Record assistant response in history
        await self.chat_repo.add_message(
            role="assistant",
            content=answer,
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id,
            show_pdf=show_pdf,
        )

    async def get_history(self, limit: int = 50) -> Sequence[ChatMessage]:
        return await self.chat_repo.get_history(limit=limit)
