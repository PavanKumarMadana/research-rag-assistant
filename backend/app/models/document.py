"""
Document Model Module.

Defines the SQLAlchemy model for uploaded documents.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Integer, DateTime, Text, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
import enum

from backend.app.core.database import Base


class ProcessingStatus(str, enum.Enum):
    """Enum for document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPROCESSING = "reprocessing"


class DocumentCategory(str, enum.Enum):
    """Enum for document categories."""
    ARTIFICIAL_INTELLIGENCE = "Artificial Intelligence"
    MACHINE_LEARNING = "Machine Learning"
    COMPUTER_VISION = "Computer Vision"
    NATURAL_LANGUAGE_PROCESSING = "Natural Language Processing"
    ROBOTICS = "Robotics"
    CYBER_SECURITY = "Cyber Security"
    CLOUD_COMPUTING = "Cloud Computing"
    UNCATEGORIZED = "Uncategorized"


class Document(Base):
    """Document model representing an uploaded research paper or technical document."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[str] = mapped_column(String(100), default="application/pdf")
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    processing_status: Mapped[str] = mapped_column(
        String(20),
        default=ProcessingStatus.PENDING.value,
    )
    category: Mapped[str] = mapped_column(
        String(50),
        default=DocumentCategory.UNCATEGORIZED.value,
    )
    classification_confidence: Mapped[Optional[float]] = mapped_column(
        default=None
    )
    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    processed_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        default=None,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), default=None)

    def __repr__(self) -> str:
        """String representation of the document."""
        return f"<Document(id={self.id}, filename={self.filename}, status={self.processing_status})>"

    def to_dict(self) -> dict:
        """Convert document to dictionary.

        Returns:
            dict: Document data as dictionary.
        """
        return {
            "document_id": self.id,
            "document_name": self.original_filename,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "processing_status": self.processing_status,
            "category": self.category,
            "classification_confidence": self.classification_confidence,
            "file_size": self.file_size,
            "error_message": self.error_message,
        }