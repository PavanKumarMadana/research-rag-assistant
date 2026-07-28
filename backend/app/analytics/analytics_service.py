"""
Analytics Service Module.

Provides business logic for analytics and reporting.
"""

from typing import Optional

from loguru import logger

from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.conversation_repository import ConversationRepository
from backend.app.repositories.analytics_repository import AnalyticsRepository
from backend.app.vector_store.vector_store_service import VectorStoreService


class AnalyticsService:
    """Service for analytics and reporting operations."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        conversation_repo: ConversationRepository,
        analytics_repo: AnalyticsRepository,
        vector_store: VectorStoreService,
    ):
        """Initialize analytics service.

        Args:
            document_repo: Document repository.
            conversation_repo: Conversation repository.
            analytics_repo: Analytics repository.
            vector_store: Vector store service.
        """
        self.document_repo = document_repo
        self.conversation_repo = conversation_repo
        self.analytics_repo = analytics_repo
        self.vector_store = vector_store

    async def get_overview(self) -> dict:
        """Get analytics overview.

        Returns:
            dict: Overview statistics.
        """
        total_documents = await self.document_repo.get_count()
        total_chunks = await self.document_repo.get_total_chunks()
        total_queries = await self.analytics_repo.get_total_queries()
        total_questions = await self.analytics_repo.get_total_questions_answered()
        total_conversations = await self.conversation_repo.get_conversation_count()

        # Get vector store stats
        vector_stats = self.vector_store.get_collection_stats()

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_embeddings": vector_stats.get("total_chunks", total_chunks),
            "total_queries": total_queries,
            "total_questions_answered": total_questions,
            "total_conversations": total_conversations,
        }

    async def get_full_analytics(self) -> dict:
        """Get complete analytics data.

        Returns:
            dict: Full analytics response.
        """
        overview = await self.get_overview()
        top_documents = await self.analytics_repo.get_most_queried_documents()
        category_distribution = await self.document_repo.get_category_distribution()
        recent_events = await self.analytics_repo.get_recent_activity(limit=20)

        # Enrich top documents with names
        enriched_docs = []
        for doc in top_documents:
            document = await self.document_repo.get_by_id(doc["document_id"])
            enriched_docs.append({
                "document_id": doc["document_id"],
                "document_name": document.original_filename if document else "Unknown",
                "query_count": doc["query_count"],
            })

        # Format recent activity
        recent_activity = []
        for event in recent_events:
            recent_activity.append({
                "id": event.id,
                "event_type": event.event_type,
                "query_text": event.query_text[:100] if event.query_text else None,
                "response_time_ms": event.response_time_ms,
                "created_at": event.created_at.isoformat() if event.created_at else None,
            })

        return {
            "overview": overview,
            "top_documents": enriched_docs,
            "category_distribution": category_distribution,
            "recent_activity": recent_activity,
        }