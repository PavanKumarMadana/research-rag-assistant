"""
Analytics Schemas Module.

Defines Pydantic models for analytics-related API responses.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class AnalyticsOverview(BaseModel):
    """Schema for analytics overview."""

    total_documents: int = Field(..., description="Total number of uploaded documents")
    total_chunks: int = Field(..., description="Total number of processed chunks")
    total_embeddings: int = Field(..., description="Total number of embeddings generated")
    total_queries: int = Field(..., description="Total number of queries answered")
    total_questions_answered: int = Field(..., description="Total questions answered")
    total_conversations: int = Field(..., description="Total conversations")


class TopDocument(BaseModel):
    """Schema for a top queried document."""

    document_id: str = Field(..., description="Document ID")
    document_name: str = Field(..., description="Document name")
    query_count: int = Field(..., description="Number of queries")


class CategoryDistribution(BaseModel):
    """Schema for category distribution."""

    category: str = Field(..., description="Category name")
    count: int = Field(..., description="Number of documents in this category")


class AnalyticsResponse(BaseModel):
    """Schema for analytics response."""

    overview: AnalyticsOverview = Field(..., description="Analytics overview")
    top_documents: list[TopDocument] = Field(
        ...,
        description="Most queried documents",
    )
    category_distribution: list[CategoryDistribution] = Field(
        ...,
        description="Document category distribution",
    )
    recent_activity: list[dict[str, Any]] = Field(
        ...,
        description="Recent activity events",
    )


class ClassificationResponse(BaseModel):
    """Schema for classification response."""

    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    predicted_category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Classification confidence")
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability scores for all categories",
    )


class TextClassificationRequest(BaseModel):
    """Schema for direct text classification request."""

    text: str = Field(
        ...,
        description="Text content to classify",
        min_length=1,
        max_length=20000,
        examples=[
            "This paper studies transformer language models for question answering.",
        ],
    )


class ClassificationBatchResponse(BaseModel):
    """Schema for batch classification response."""

    classifications: list[ClassificationResponse] = Field(
        ...,
        description="List of classification results",
    )
    model_info: dict[str, Any] = Field(
        ...,
        description="Model information",
    )
