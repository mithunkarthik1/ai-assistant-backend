"""
Repository for persisting and querying ChatMessage entities in PostgreSQL.
"""
import logging
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat_bot.model import ChatMessage
from app.rag.indexing import POLICY_DOC_ID

logger = logging.getLogger("app.chat_bot.repository")


class ChatMessageRepository:
    """
    Handles database operations for conversation messages.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_message(
        self,
        role: str,
        content: str,
        document_id: uuid.UUID | None = None,
    ) -> ChatMessage:
        """
        Inserts a new chat message into the database.
        """
        try:
            msg = ChatMessage(
                document_id=document_id or POLICY_DOC_ID,
                role=role,
                content=content,
            )
            self.session.add(msg)
            await self.session.commit()
            await self.session.refresh(msg)
            return msg
        except Exception as e:
            await self.session.rollback()
            logger.error("Failed to commit chat message to database: %s", e, exc_info=True)
            raise

    async def get_history(
        self,
        document_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[ChatMessage]:
        """
        Fetches the recent message history for a given document in chronological order.
        """
        try:
            doc_id = document_id or POLICY_DOC_ID
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.document_id == doc_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            messages = list(result.scalars().all())
            # Return in chronological order (oldest first)
            messages.reverse()
            return messages
        except Exception as e:
            logger.error("Failed to fetch chat history from database: %s", e, exc_info=True)
            raise
