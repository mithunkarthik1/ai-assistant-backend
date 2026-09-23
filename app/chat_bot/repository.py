import uuid
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat_bot.model import ChatMessage
from app.rag.indexing import POLICY_DOC_ID


class ChatMessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_message(
        self,
        role: str,
        content: str,
        document_id: uuid.UUID | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            document_id=document_id or POLICY_DOC_ID,
            role=role,
            content=content,
        )
        self.session.add(msg)
        await self.session.commit()
        await self.session.refresh(msg)
        return msg

    async def get_history(
        self,
        document_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[ChatMessage]:
        doc_id = document_id or POLICY_DOC_ID
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.document_id == doc_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages = list(result.scalars().all())
        # Return in chronological order (oldest to newest)
        messages.reverse()
        return messages
