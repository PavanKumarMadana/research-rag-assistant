"""
RAG Schemas Module.

Defines Pydantic models for RAG-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Schema for a source document in a response."""

    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="Relevant page number")
    chunk_content: str = Field(..., description="Retrieved chunk content")
    similarity_score: float = Field(..., description="Relevance score")


class QuestionRequest(BaseModel):
    """Schema for asking a question."""

    query: str = Field(..., description="User question", min_length=1, max_length=5000)
    session_id: str = Field(..., description="Conversation session ID")
    document_ids: Optional[list[str]] = Field(
        None,
        description="Optional list of document IDs to restrict search to",
    )
    search_mode: str = Field(
        default="semantic",
        description="Search mode: keyword, semantic, or hybrid",
    )
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve",
        ge=1,
        le=20,
    )


class QuestionResponse(BaseModel):
    """Schema for question answering response."""

    answer: str = Field(..., description="Generated answer")
    retrieved_context: list[SourceDocument] = Field(
        ...,
        description="Retrieved chunks used as grounding context",
    )
    sources: list[SourceDocument] = Field(
        ...,
        description="Source documents used for the answer",
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score of the answer",
        ge=0.0,
        le=1.0,
    )
    session_id: str = Field(..., description="Conversation session ID")
    processing_time_ms: float = Field(
        ...,
        description="Time taken to generate the answer",
    )


class SummarizeRequest(BaseModel):
    """Schema for summarization request."""

    document_id: str = Field(..., description="Document ID to summarize")
    summary_type: str = Field(
        default="executive",
        description="Summary type: executive, technical, bullet, or key_takeaways",
    )
    max_length: int = Field(
        default=500,
        description="Maximum length of summary in words",
        ge=100,
        le=5000,
    )


class SummarizeResponse(BaseModel):
    """Schema for summarization response."""

    document_id: str = Field(..., description="Document ID")
    document_name: str = Field(..., description="Document name")
    summary_type: str = Field(..., description="Type of summary generated")
    summary: str = Field(..., description="Generated summary")
    word_count: int = Field(..., description="Word count of summary")
    processing_time_ms: float = Field(
        ...,
        description="Time taken to generate summary",
    )


class CompareRequest(BaseModel):
    """Schema for document comparison request."""

    document_ids: list[str] = Field(
        ...,
        description="List of document IDs to compare",
        min_length=2,
        max_length=5,
    )
    comparison_aspects: Optional[list[str]] = Field(
        None,
        description="Specific aspects to compare (e.g., methodology, conclusion)",
    )
    session_id: str = Field(..., description="Conversation session ID")


class ComparisonAspect(BaseModel):
    """Schema for a single comparison aspect."""

    aspect: str = Field(..., description="Aspect being compared")
    documents: dict[str, str] = Field(
        ...,
        description="Map of document name to content for this aspect",
    )


class CompareResponse(BaseModel):
    """Schema for comparison response."""

    comparison: str = Field(..., description="Generated comparison text")
    aspects: list[ComparisonAspect] = Field(
        ...,
        description="Structured comparison by aspect",
    )
    documents_compared: list[str] = Field(
        ...,
        description="Names of documents compared",
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to generate comparison",
    )


class ChatRequest(BaseModel):
    """Schema for chat request."""

    message: str = Field(..., description="User message", min_length=1, max_length=5000)
    session_id: str = Field(..., description="Conversation session ID")
    document_ids: Optional[list[str]] = Field(
        None,
        description="Optional document IDs for context",
    )


class ChatResponse(BaseModel):
    """Schema for chat response."""

    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Conversation session ID")
    sources: Optional[list[SourceDocument]] = Field(
        None,
        description="Source documents if applicable",
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to generate response",
    )
