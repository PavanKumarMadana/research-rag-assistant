"""
Document Routes Module.

REST API endpoints for document management.
"""

import os
import uuid
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Query,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db, async_session_factory
from backend.app.core.config import settings
from backend.app.models.document import Document, ProcessingStatus
from backend.app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentUploadBatchResponse,
    DocumentDeleteResponse,
    DocumentReprocessResponse,
    SearchResponse,
)
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.vector_store.vector_store_service import VectorStoreService
from backend.app.services.document_processor import DocumentProcessor
from backend.app.ml.classification_service import ClassificationService

router = APIRouter(prefix="/api/documents", tags=["Documents"])


async def process_document_job(document_id: str) -> None:
    """Process a document in an isolated database session."""
    async with async_session_factory() as session:
        document_repo = DocumentRepository(session)
        await document_repo.update_status(document_id, ProcessingStatus.PROCESSING)
        await session.commit()

        vector_store = VectorStoreService()
        classification_service = ClassificationService()
        processor = DocumentProcessor(document_repo, vector_store, classification_service)
        await processor.process_document(document_id)
        await session.commit()


async def reprocess_document_job(document_id: str) -> None:
    """Reprocess a document in an isolated database session."""
    async with async_session_factory() as session:
        document_repo = DocumentRepository(session)
        await document_repo.update_status(document_id, ProcessingStatus.REPROCESSING)
        await session.commit()

        vector_store = VectorStoreService()
        classification_service = ClassificationService()
        processor = DocumentProcessor(document_repo, vector_store, classification_service)
        await processor.reprocess_document(document_id)
        await session.commit()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse | DocumentUploadBatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload one or more PDF documents",
    description="Upload PDF files for processing. Files are automatically processed through the pipeline.",
)
async def upload_documents(
    files: list[UploadFile] = File(..., description="PDF files to upload"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
):
    """Upload PDF documents for processing.

    Args:
        files: List of PDF files to upload.
        db: Database session.

    Returns:
        DocumentUploadResponse: Upload status.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    results = []
    document_repo = DocumentRepository(db)
    for file in files:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' is not a PDF",
            )

        # Read file content
        content = await file.read()

        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file.filename}' exceeds maximum upload size",
            )

        # Save file to disk
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as f:
            f.write(content)

        # Create document record
        document = Document(
            id=file_id,
            filename=safe_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=len(content),
            mime_type="application/pdf",
            processing_status=ProcessingStatus.PENDING.value,
        )

        document = await document_repo.create(document)

        await db.commit()

        if background_tasks is not None:
            background_tasks.add_task(process_document_job, file_id)

        results.append({
            "message": "Document uploaded and processing started",
            "document_id": file_id,
            "filename": file.filename,
            "status": "processing",
        })

    return results[0] if len(results) == 1 else {"documents": results}


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
    description="Get a paginated list of all uploaded documents with metadata.",
)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by processing status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded documents.

    Args:
        skip: Pagination offset.
        limit: Page size.
        status_filter: Optional status filter.
        category: Optional category filter.
        db: Database session.

    Returns:
        DocumentListResponse: List of documents.
    """
    document_repo = DocumentRepository(db)
    documents, total = await document_repo.get_all(
        skip=skip,
        limit=limit,
        status=status_filter,
        category=category,
    )

    return {
        "documents": [doc.to_dict() for doc in documents],
        "total": total,
        "page": (skip // limit) + 1,
        "page_size": limit,
    }


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Get detailed information about a specific document.",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get document by ID.

    Args:
        document_id: Document UUID.
        db: Database session.

    Returns:
        DocumentResponse: Document details.
    """
    document_repo = DocumentRepository(db)
    document = await document_repo.get_by_id(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return document.to_dict()


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Delete a document and its associated chunks from the vector store.",
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document.

    Args:
        document_id: Document UUID.
        db: Database session.

    Returns:
        DocumentDeleteResponse: Deletion status.
    """
    document_repo = DocumentRepository(db)
    vector_store = VectorStoreService()

    document = await document_repo.get_by_id(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    file_path = document.file_path

    # Delete from vector store
    vector_store.delete_document_chunks(document_id)

    # Delete from database
    await document_repo.delete(document_id)

    # Delete file from disk
    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
    }


@router.post(
    "/reprocess/{document_id}",
    response_model=DocumentReprocessResponse,
    summary="Reprocess a document",
    description="Reprocess an existing document through the entire pipeline.",
)
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Reprocess an existing document.

    Args:
        document_id: Document UUID.
        db: Database session.

    Returns:
        DocumentReprocessResponse: Reprocess status.
    """
    document_repo = DocumentRepository(db)
    document = await document_repo.get_by_id(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    await document_repo.update_status(document_id, ProcessingStatus.REPROCESSING)
    background_tasks.add_task(reprocess_document_job, document_id)

    return {
        "message": "Document reprocessing started",
        "document_id": document_id,
        "status": "reprocessing",
    }


@router.get(
    "/search/{query}",
    response_model=SearchResponse,
    summary="Search across documents",
    description="Search across all uploaded documents using keyword, semantic, or hybrid search.",
)
async def search_documents(
    query: str,
    search_mode: str = Query("semantic", description="Search mode: keyword, semantic, or hybrid"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
    document_ids: Optional[str] = Query(None, description="Comma-separated document IDs to filter"),
    db: AsyncSession = Depends(get_db),
):
    """Search across documents.

    Args:
        query: Search query.
        search_mode: Search strategy.
        top_k: Number of results.
        document_ids: Optional document filter.

    Returns:
        SearchResponse: Search results.
    """
    import time
    start_time = time.time()

    vector_store = VectorStoreService()
    document_repo = DocumentRepository(db)

    doc_id_list = None
    if document_ids:
        doc_id_list = [d.strip() for d in document_ids.split(",")]

    if search_mode == "keyword":
        results = vector_store.keyword_search(query, top_k, doc_id_list)
    elif search_mode == "hybrid":
        results = vector_store.hybrid_search(query, top_k, doc_id_list)
    else:
        results = vector_store.semantic_search(query, top_k, doc_id_list)

    names_by_id = {}
    for result in results:
        result_document_id = result.get("document_id")
        if result_document_id and result_document_id not in names_by_id:
            document = await document_repo.get_by_id(result_document_id)
            names_by_id[result_document_id] = (
                document.original_filename if document else result_document_id
            )
        result["document_name"] = names_by_id.get(result_document_id, result_document_id or "")

    time_taken = (time.time() - start_time) * 1000

    return {
        "query": query,
        "results": results,
        "total_results": len(results),
        "search_mode": search_mode,
        "time_taken_ms": time_taken,
    }
