"""
Document Schemas Module.

Defines Pydantic models for document-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Schema for document response."""

    document_id: str = Field(..., description="Unique identifier for the document")
    document_name: str = Field(..., description="Original filename of the document")
    upload_timestamp: Optional[str] = Field(None, description="Upload timestamp")
    total_pages: int = Field(0, description="Total number of pages")
    total_chunks: int = Field(0, description="Total number of chunks")
    processing_status: str = Field(..., description="Current processing status")
    category: str = Field("Uncategorized", description="Document category")
    classification_confidence: Optional[float] = Field(None, description="Classification confidence score")
    file_size: int = Field(0, description="File size in bytes")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

    class Config:
        """Pydantic config."""
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for document list response."""

    documents: list[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(10, description="Number of items per page")


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""

    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="ID of the uploaded document")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")


class DocumentUploadBatchResponse(BaseModel):
    """Schema for multi-document upload response."""

    documents: list[DocumentUploadResponse] = Field(
        ...,
        description="Uploaded documents and their processing status",
    )


class DocumentDeleteResponse(BaseModel):
    """Schema for document deletion response."""

    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="ID of the deleted document")


class DocumentReprocessResponse(BaseModel):
    """Schema for document reprocess response."""

    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="ID of the reprocessed document")
    status: str = Field(..., description="New processing status")


class DocumentChunk(BaseModel):
    """Schema for a document chunk."""

    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk text content")
    page_number: int = Field(..., description="Source page number")
    chunk_index: int = Field(..., description="Chunk index in document")
    document_id: str = Field(..., description="Source document ID")


class SearchResult(BaseModel):
    """Schema for a single search result."""

    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk text content")
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="Source page number")
    similarity_score: float = Field(..., description="Similarity score")
    chunk_index: int = Field(..., description="Chunk index")


class SearchResponse(BaseModel):
    """Schema for search response."""

    query: str = Field(..., description="Original search query")
    results: list[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_mode: str = Field(..., description="Search mode used")
    time_taken_ms: float = Field(..., description="Time taken for search")
