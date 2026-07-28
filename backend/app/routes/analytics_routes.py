"""
Analytics Routes Module.

REST API endpoints for analytics and reporting.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from backend.app.core.database import get_db
from backend.app.schemas.analytics import AnalyticsResponse
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.conversation_repository import ConversationRepository
from backend.app.repositories.analytics_repository import AnalyticsRepository
from backend.app.vector_store.vector_store_service import VectorStoreService
from backend.app.analytics.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


def get_analytics_service(db: AsyncSession = Depends(get_db)) -> AnalyticsService:
    """Dependency to get analytics service.

    Args:
        db: Database session.

    Returns:
        AnalyticsService: Configured analytics service.
    """
    doc_repo = DocumentRepository(db)
    conv_repo = ConversationRepository(db)
    analytics_repo = AnalyticsRepository(db)
    vector_store = VectorStoreService()
    return AnalyticsService(doc_repo, conv_repo, analytics_repo, vector_store)


@router.get(
    "/overview",
    summary="Get analytics overview",
    description="Get an overview of system usage statistics including document counts, queries, and conversations.",
)
async def get_analytics_overview(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """Get analytics overview.

    Returns:
        dict: Overview statistics.
    """
    return await analytics_service.get_overview()


@router.get(
    "/full",
    response_model=AnalyticsResponse,
    summary="Get full analytics",
    description="Get comprehensive analytics including overview, top documents, category distribution, and recent activity.",
)
async def get_full_analytics(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """Get full analytics data.

    Returns:
        AnalyticsResponse: Complete analytics.
    """
    return await analytics_service.get_full_analytics()


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API server is running and healthy.",
)
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Health status.
    """
    return {
        "status": "healthy",
        "service": "AI Research & Knowledge Assistant",
        "version": "1.0.0",
    }