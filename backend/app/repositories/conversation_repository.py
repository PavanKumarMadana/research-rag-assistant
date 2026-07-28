"""
Conversation Repository Module.

Provides data access layer for conversation and message operations.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from backend.app.models.conversation import Conversation, Message


class ConversationRepository:
    """Repository for conversation database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self.session = session

    async def create_conversation(
        self,
        session_id: str,
        title: str = "New Conversation",
    ) -> Conversation:
        """Create a new conversation.

        Args:
            session_id: Session identifier.
            title: Conversation title.

        Returns:
            Conversation: Created conversation.
        """
        conversation = Conversation(
            session_id=session_id,
            title=title,
        )
        self.session.add(conversation)
        await self.session.flush()
        logger.info(f"Conversation created: {conversation.id}")
        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID.

        Args:
            conversation_id: Conversation UUID.

        Returns:
            Optional[Conversation]: Conversation if found.
        """
        result = await self.session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def get_conversation_by_session(
        self,
        session_id: str,
    ) -> Optional[Conversation]:
        """Get active conversation by session ID.

        Args:
            session_id: Session identifier.

        Returns:
            Optional[Conversation]: Active conversation if found.
        """
        result = await self.session.execute(
            select(Conversation)
            .where(
                Conversation.session_id == session_id,
                Conversation.is_active == True,
            )
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_or_create_conversation(
        self,
        session_id: str,
    ) -> Conversation:
        """Get existing active conversation or create a new one.

        Args:
            session_id: Session identifier.

        Returns:
            Conversation: Existing or new conversation.
        """
        conversation = await self.get_conversation_by_session(session_id)
        if not conversation:
            conversation = await self.create_conversation(session_id)
        return conversation

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        metadata_json: Optional[str] = None,
    ) -> Message:
        """Add a message to a conversation.

        Args:
            conversation_id: Conversation UUID.
            role: Message role (user/assistant).
            content: Message content.
            message_type: Type of message.
            metadata_json: Optional metadata JSON.

        Returns:
            Message: Created message.
        """
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_type=message_type,
            metadata_json=metadata_json,
        )
        self.session.add(message)

        # Update conversation timestamp
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.updated_at = datetime.now(timezone.utc)

        await self.session.flush()
        return message

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> list[Message]:
        """Get message history for a conversation.

        Args:
            conversation_id: Conversation UUID.
            limit: Maximum number of messages.

        Returns:
            list[Message]: List of messages in chronological order.
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[Message]:
        """Get message history for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of messages.

        Returns:
            list[Message]: List of messages.
        """
        conversation = await self.get_conversation_by_session(session_id)
        if not conversation:
            return []
        return await self.get_conversation_history(conversation.id, limit)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages.

        Args:
            conversation_id: Conversation UUID.

        Returns:
            bool: True if deleted.
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False

        await self.session.delete(conversation)
        await self.session.flush()
        return True

    async def get_conversation_count(self) -> int:
        """Get total number of conversations.

        Returns:
            int: Total conversation count.
        """
        result = await self.session.execute(
            select(func.count()).select_from(Conversation)
        )
        return result.scalar() or 0

    async def get_all_conversations(
        self,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[Conversation], int]:
        """Get all conversations.

        Args:
            skip: Number to skip.
            limit: Max to return.

        Returns:
            tuple: List of conversations and total count.
        """
        query = select(Conversation).order_by(Conversation.updated_at.desc())
        count_result = await self.session.execute(
            select(func.count()).select_from(Conversation)
        )
        total = count_result.scalar() or 0

        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        conversations = list(result.scalars().all())
        return conversations, total