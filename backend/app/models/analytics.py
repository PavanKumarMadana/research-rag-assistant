"""
Analytics Model Module.

Defines the SQLAlchemy model for tracking analytics events.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Integer, DateTime, Text, Float
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class AnalyticsEvent(Base):
    """Analytics event model for tracking system usage."""

    __tablename__ = "analytics_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    event_data: Mapped[str] = mapped_column(Text, default="{}")
    document_id: Mapped[str] = mapped_column(String(36), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=True, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=True)
    response_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<AnalyticsEvent(id={self.id}, type={self.event_type})>"

    def to_dict(self) -> dict:
        """Convert event to dictionary.

        Returns:
            dict: Event data as dictionary.
        """
        return {
            "id": self.id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "query_text": self.query_text,
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }