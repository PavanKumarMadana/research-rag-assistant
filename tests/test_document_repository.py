"""
Unit tests for Document Repository.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.app.models.document import Document, ProcessingStatus
from backend.app.repositories.document_repository import DocumentRepository


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def document_repo(mock_session):
    """Create a document repository with mock session."""
    return DocumentRepository(mock_session)


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="test-doc-123",
        filename="test.pdf",
        original_filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_size=1024,
        processing_status=ProcessingStatus.PENDING.value,
    )


@pytest.mark.asyncio
async def test_create_document(document_repo, mock_session, sample_document):
    """Test creating a document."""
    result = await document_repo.create(sample_document)
    mock_session.add.assert_called_once_with(sample_document)
    mock_session.flush.assert_called_once()
    assert result == sample_document


@pytest.mark.asyncio
async def test_get_by_id_found(document_repo, mock_session, sample_document):
    """Test getting a document by ID when it exists."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = sample_document
    mock_session.execute.return_value = mock_result

    result = await document_repo.get_by_id("test-doc-123")
    assert result == sample_document


@pytest.mark.asyncio
async def test_get_by_id_not_found(document_repo, mock_session):
    """Test getting a document by ID when it doesn't exist."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await document_repo.get_by_id("non-existent")
    assert result is None


@pytest.mark.asyncio
async def test_update_status(document_repo, mock_session, sample_document):
    """Test updating document status."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = sample_document
    mock_session.execute.return_value = mock_result

    result = await document_repo.update_status(
        "test-doc-123", ProcessingStatus.COMPLETED
    )
    assert result is not None
    assert result.processing_status == ProcessingStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_delete_document(document_repo, mock_session, sample_document):
    """Test deleting a document."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = sample_document
    mock_session.execute.return_value = mock_result

    result = await document_repo.delete("test-doc-123")
    assert result is True
    mock_session.delete.assert_called_once_with(sample_document)


@pytest.mark.asyncio
async def test_delete_document_not_found(document_repo, mock_session):
    """Test deleting a non-existent document."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await document_repo.delete("non-existent")
    assert result is False


@pytest.mark.asyncio
async def test_get_count(document_repo, mock_session):
    """Test getting document count."""
    mock_result = MagicMock()
    mock_result.scalar.return_value = 5
    mock_session.execute.return_value = mock_result

    count = await document_repo.get_count()
    assert count == 5
