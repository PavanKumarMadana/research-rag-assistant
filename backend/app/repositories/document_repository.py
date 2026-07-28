"""
Document Repository Module.

Provides data access layer for document operations using SQLAlchemy.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, delete, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from backend.app.models.document import Document, ProcessingStatus, DocumentCategory


class DocumentRepository:
    """Repository for document database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self.session = session

    async def create(self, document: Document) -> Document:
        """Create a new document record.

        Args:
            document: Document model instance.

        Returns:
            Document: Created document.
        """
        self.session.add(document)
        await self.session.flush()
        logger.info(f"Document created: {document.id}")
        return document

    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get a document by its ID.

        Args:
            document_id: Document UUID.

        Returns:
            Optional[Document]: Document if found, None otherwise.
        """
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> tuple[list[Document], int]:
        """Get all documents with optional filtering.

        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            status: Filter by processing status.
            category: Filter by category.

        Returns:
            tuple: List of documents and total count.
        """
        query = select(Document)

        if status:
            query = query.where(Document.processing_status == status)
        if category:
            query = query.where(Document.category == category)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated results
        query = query.order_by(Document.upload_timestamp.desc())
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        documents = list(result.scalars().all())

        return documents, total

    async def update_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None,
    ) -> Optional[Document]:
        """Update document processing status.

        Args:
            document_id: Document UUID.
            status: New processing status.
            error_message: Optional error message.

        Returns:
            Optional[Document]: Updated document if found.
        """
        document = await self.get_by_id(document_id)
        if not document:
            logger.warning(f"Document not found for status update: {document_id}")
            return None

        document.processing_status = status.value
        if error_message:
            document.error_message = error_message

        if status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            document.processed_timestamp = datetime.now(timezone.utc)

        await self.session.flush()
        logger.info(f"Document {document_id} status updated to {status.value}")
        return document

    async def update_processing_metadata(
        self,
        document_id: str,
        total_pages: int,
        total_chunks: int,
        category: Optional[str] = None,
        classification_confidence: Optional[float] = None,
        content_hash: Optional[str] = None,
    ) -> Optional[Document]:
        """Update document processing metadata.

        Args:
            document_id: Document UUID.
            total_pages: Number of pages.
            total_chunks: Number of chunks.
            category: Document category.
            classification_confidence: Classification confidence.
            content_hash: Content hash.

        Returns:
            Optional[Document]: Updated document if found.
        """
        document = await self.get_by_id(document_id)
        if not document:
            return None

        document.total_pages = total_pages
        document.total_chunks = total_chunks
        if category:
            document.category = category
        if classification_confidence is not None:
            document.classification_confidence = classification_confidence
        if content_hash:
            document.content_hash = content_hash

        await self.session.flush()
        return document

    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID.

        Args:
            document_id: Document UUID.

        Returns:
            bool: True if deleted, False if not found.
        """
        document = await self.get_by_id(document_id)
        if not document:
            return False

        await self.session.delete(document)
        await self.session.flush()
        logger.info(f"Document deleted: {document_id}")
        return True

    async def get_count(self) -> int:
        """Get total number of documents.

        Returns:
            int: Total document count.
        """
        result = await self.session.execute(
            select(func.count()).select_from(Document)
        )
        return result.scalar() or 0

    async def get_category_distribution(self) -> list[dict]:
        """Get document count by category.

        Returns:
            list[dict]: List of category counts.
        """
        result = await self.session.execute(
            select(
                Document.category,
                func.count().label("count"),
            ).group_by(Document.category)
        )
        rows = result.all()
        return [{"category": row[0], "count": row[1]} for row in rows]

    async def get_total_chunks(self) -> int:
        """Get total number of chunks across all documents.

        Returns:
            int: Total chunk count.
        """
        result = await self.session.execute(
            select(func.sum(Document.total_chunks))
        )
        return result.scalar() or 0