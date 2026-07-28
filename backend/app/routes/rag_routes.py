"""
RAG Routes Module.

REST API endpoints for RAG operations: question answering, summarization, comparison, and chat.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from backend.app.core.database import get_db
from backend.app.schemas.rag import (
    QuestionRequest,
    QuestionResponse,
    SummarizeRequest,
    SummarizeResponse,
    CompareRequest,
    CompareResponse,
    ChatRequest,
    ChatResponse,
)
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.conversation_repository import ConversationRepository
from backend.app.repositories.analytics_repository import AnalyticsRepository
from backend.app.vector_store.vector_store_service import VectorStoreService
from backend.app.services.llm_service import LLMService
from backend.app.rag.rag_service import RAGService

router = APIRouter(prefix="/api/rag", tags=["RAG & AI Assistant"])


def get_rag_service(db: AsyncSession = Depends(get_db)) -> RAGService:
    """Dependency to get RAG service instance.

    Args:
        db: Database session.

    Returns:
        RAGService: Configured RAG service.
    """
    llm = LLMService()
    vector_store = VectorStoreService()
    doc_repo = DocumentRepository(db)
    conv_repo = ConversationRepository(db)
    analytics_repo = AnalyticsRepository(db)
    return RAGService(llm, vector_store, doc_repo, conv_repo, analytics_repo)


@router.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask a question about documents",
    description="Ask a question and get an answer grounded in the uploaded documents with citations.",
)
async def ask_question(
    request: QuestionRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """Ask a question and get a grounded answer.

    Args:
        request: Question request with query and session.

    Returns:
        QuestionResponse: Answer with sources and confidence.
    """
    try:
        result = await rag_service.answer_question(
            query=request.query,
            session_id=request.session_id,
            document_ids=request.document_ids,
            search_mode=request.search_mode,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}",
        )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a document",
    description="Generate different types of summaries for a document (executive, technical, bullet, key takeaways).",
)
async def summarize_document(
    request: SummarizeRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """Generate a summary for a document.

    Args:
        request: Summarization request.

    Returns:
        SummarizeResponse: Generated summary.
    """
    try:
        result = await rag_service.generate_summary(
            document_id=request.document_id,
            summary_type=request.summary_type,
            max_length=request.max_length,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare multiple documents",
    description="Compare two or more documents across various aspects like methodology, findings, and conclusions.",
)
async def compare_documents(
    request: CompareRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """Compare multiple documents.

    Args:
        request: Comparison request.

    Returns:
        CompareResponse: Comparison results.
    """
    try:
        result = await rag_service.compare_documents(
            document_ids=request.document_ids,
            comparison_aspects=request.comparison_aspects,
            session_id=request.session_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Document comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare documents: {str(e)}",
        )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the AI assistant",
    description="Have a conversation with the AI assistant that maintains context across messages.",
)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """Chat with the AI assistant.

    Args:
        request: Chat request.

    Returns:
        ChatResponse: AI response.
    """
    try:
        result = await rag_service.chat(
            message=request.message,
            session_id=request.session_id,
            document_ids=request.document_ids,
        )
        return result
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )


@router.get(
    "/conversations/{session_id}",
    summary="Get conversation history",
    description="Retrieve the conversation history for a given session.",
)
async def get_conversation_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get conversation history.

    Args:
        session_id: Session identifier.
        db: Database session.

    Returns:
        dict: Conversation history.
    """
    conv_repo = ConversationRepository(db)
    messages = await conv_repo.get_session_history(session_id)
    return {
        "session_id": session_id,
        "messages": [msg.to_dict() for msg in messages],
        "total": len(messages),
    }