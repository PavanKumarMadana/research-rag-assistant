"""
ML Routes Module.

REST API endpoints for TensorFlow document classification and ML pipeline management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from backend.app.core.database import get_db
from backend.app.schemas.analytics import (
    ClassificationResponse,
    ClassificationBatchResponse,
    TextClassificationRequest,
)
from backend.app.ml.classification_service import ClassificationService, MLPipeline
from backend.app.repositories.document_repository import DocumentRepository

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])


@router.post(
    "/classify/{document_id}",
    response_model=ClassificationResponse,
    summary="Classify a document",
    description="Classify a document into a predefined category using the TensorFlow model.",
)
async def classify_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Classify a document by its ID.

    Args:
        document_id: Document UUID.
        db: Database session.

    Returns:
        ClassificationResponse: Classification result.
    """
    document_repo = DocumentRepository(db)
    document = await document_repo.get_by_id(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Read document text from file
    try:
        import fitz
        with fitz.open(document.file_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read document: {str(e)}",
        )

    # Classify
    classifier = ClassificationService()
    result = classifier.classify_text(text[:5000])
    await document_repo.update_processing_metadata(
        document_id=document.id,
        total_pages=document.total_pages,
        total_chunks=document.total_chunks,
        category=result["predicted_category"],
        classification_confidence=result["confidence"],
    )

    return {
        "document_id": document_id,
        "filename": document.original_filename,
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    }


@router.post(
    "/classify-text",
    response_model=ClassificationResponse,
    summary="Classify text directly",
    description="Classify arbitrary text into a predefined category.",
)
async def classify_text(
    request: TextClassificationRequest,
):
    """Classify arbitrary text.

    Args:
        request: Text classification request.

    Returns:
        ClassificationResponse: Classification result.
    """
    classifier = ClassificationService()
    result = classifier.classify_text(request.text)

    return {
        "document_id": "text-input",
        "filename": "direct-text-input",
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    }


@router.get(
    "/model-info",
    summary="Get model information",
    description="Get information about the classification model including categories and status.",
)
async def get_model_info():
    """Get classification model information.

    Returns:
        dict: Model information.
    """
    classifier = ClassificationService()
    return classifier.get_model_info()


@router.post(
    "/train",
    summary="Train the classification model",
    description="Train the TensorFlow document classification model using keyword-based training data.",
)
async def train_model():
    """Train the TensorFlow classification model.

    Returns:
        dict: Training status.
    """
    pipeline = MLPipeline()
    success = pipeline.train_model()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model training failed. Ensure TensorFlow is installed.",
        )

    return {
        "message": "Model trained successfully",
        "status": "completed",
    }


@router.get(
    "/evaluate",
    summary="Evaluate the model",
    description="Evaluate the trained model's performance metrics.",
)
async def evaluate_model():
    """Evaluate the trained model.

    Returns:
        dict: Evaluation metrics.
    """
    pipeline = MLPipeline()
    metrics = pipeline.evaluate_model()

    if "error" in metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=metrics["error"],
        )

    return metrics


@router.post(
    "/classify-all",
    response_model=ClassificationBatchResponse,
    summary="Classify all unclassified documents",
    description="Batch classify all documents that are currently uncategorized.",
)
async def classify_all_documents(
    db: AsyncSession = Depends(get_db),
):
    """Classify all unclassified documents.

    Args:
        db: Database session.

    Returns:
        ClassificationBatchResponse: Batch classification results.
    """
    document_repo = DocumentRepository(db)
    classifier = ClassificationService()

    documents, _ = await document_repo.get_all(
        limit=100,
        category="Uncategorized",
    )

    classifications = []
    for doc in documents:
        try:
            import fitz
            with fitz.open(doc.file_path) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text()

            result = classifier.classify_text(text[:5000])

            # Update document category
            await document_repo.update_processing_metadata(
                document_id=doc.id,
                total_pages=doc.total_pages,
                total_chunks=doc.total_chunks,
                category=result["predicted_category"],
                classification_confidence=result["confidence"],
            )

            classifications.append({
                "document_id": doc.id,
                "filename": doc.original_filename,
                "predicted_category": result["predicted_category"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            })

        except Exception as e:
            logger.error(f"Failed to classify document {doc.id}: {e}")

    return {
        "classifications": classifications,
        "model_info": classifier.get_model_info(),
    }
