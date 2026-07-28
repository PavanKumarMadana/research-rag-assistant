"""
Analytics Repository Module.

Provides data access layer for analytics event operations.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from backend.app.models.analytics import AnalyticsEvent


class AnalyticsRepository:
    """Repository for analytics database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self.session = session

    async def record_event(
        self,
        event_type: str,
        event_data: Optional[dict] = None,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None,
        query_text: Optional[str] = None,
        response_time_ms: float = 0.0,
        tokens_used: int = 0,
    ) -> AnalyticsEvent:
        """Record an analytics event.

        Args:
            event_type: Type of event.
            event_data: Additional event data.
            document_id: Related document ID.
            session_id: Session identifier.
            query_text: Query text.
            response_time_ms: Response time in milliseconds.
            tokens_used: Number of tokens used.

        Returns:
            AnalyticsEvent: Created event.
        """
        import json
        event = AnalyticsEvent(
            event_type=event_type,
            event_data=json.dumps(event_data or {}),
            document_id=document_id,
            session_id=session_id,
            query_text=query_text,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def get_event_count_by_type(self, event_type: str) -> int:
        """Get count of events by type.

        Args:
            event_type: Event type to count.

        Returns:
            int: Event count.
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(AnalyticsEvent)
            .where(AnalyticsEvent.event_type == event_type)
        )
        return result.scalar() or 0

    async def get_total_queries(self) -> int:
        """Get total number of queries.

        Returns:
            int: Total query count.
        """
        return await self.get_event_count_by_type("question")

    async def get_total_questions_answered(self) -> int:
        """Get total number of questions answered.

        Returns:
            int: Total answered count.
        """
        return await self.get_event_count_by_type("answer")

    async def get_most_queried_documents(
        self,
        limit: int = 5,
    ) -> list[dict]:
        """Get most frequently queried documents.

        Args:
            limit: Maximum number of results.

        Returns:
            list[dict]: List of document query counts.
        """
        result = await self.session.execute(
            select(
                AnalyticsEvent.document_id,
                func.count().label("query_count"),
            )
            .where(AnalyticsEvent.document_id.isnot(None))
            .group_by(AnalyticsEvent.document_id)
            .order_by(func.count().desc())
            .limit(limit)
        )
        rows = result.all()
        return [
            {"document_id": row[0], "query_count": row[1]}
            for row in rows
        ]

    async def get_recent_activity(
        self,
        limit: int = 20,
    ) -> list[AnalyticsEvent]:
        """Get recent activity events.

        Args:
            limit: Maximum number of events.

        Returns:
            list[AnalyticsEvent]: Recent events.
        """
        result = await self.session.execute(
            select(AnalyticsEvent)
            .order_by(AnalyticsEvent.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_average_response_time(self) -> float:
        """Get average response time across all queries.

        Returns:
            float: Average response time in ms.
        """
        result = await self.session.execute(
            select(func.avg(AnalyticsEvent.response_time_ms))
            .where(AnalyticsEvent.event_type == "answer")
        )
        return result.scalar() or 0.0

    async def get_total_tokens_used(self) -> int:
        """Get total tokens used.

        Returns:
            int: Total tokens.
        """
        result = await self.session.execute(
            select(func.sum(AnalyticsEvent.tokens_used))
        )
        return result.scalar() or 0