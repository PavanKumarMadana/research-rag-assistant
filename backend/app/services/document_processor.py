"""
Document Processing Service Module.

Handles PDF text extraction, cleaning, chunking, and processing pipeline.
"""

import os
import hashlib
import time
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger

from backend.app.core.config import settings
from backend.app.models.document import ProcessingStatus
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.vector_store.vector_store_service import VectorStoreService
from backend.app.ml.classification_service import ClassificationService


class DocumentProcessor:
    """Service for processing uploaded documents."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_store: VectorStoreService,
        classification_service: Optional[ClassificationService] = None,
    ):
        """Initialize document processor.

        Args:
            document_repository: Document data repository.
            vector_store: Vector store service.
            classification_service: Optional classification service.
        """
        self.document_repo = document_repository
        self.vector_store = vector_store
        self.classification_service = classification_service
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    async def process_document(self, document_id: str) -> bool:
        """Process an uploaded document through the pipeline.

        Pipeline steps:
        1. Extract text from PDF
        2. Clean and normalize text
        3. Split into chunks
        4. Generate embeddings
        5. Index in vector store
        6. Classify document
        7. Update metadata

        Args:
            document_id: Document UUID to process.

        Returns:
            bool: True if processing succeeded.
        """
        start_time = time.time()
        logger.info(f"Starting document processing: {document_id}")

        try:
            # Update status to processing
            await self.document_repo.update_status(
                document_id, ProcessingStatus.PROCESSING
            )

            # Get document
            document = await self.document_repo.get_by_id(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                await self.document_repo.update_status(
                    document_id, ProcessingStatus.FAILED,
                    error_message="Document not found"
                )
                return False

            file_path = document.file_path
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                await self.document_repo.update_status(
                    document_id, ProcessingStatus.FAILED,
                    error_message="File not found on disk"
                )
                return False

            # Step 1: Extract text from PDF
            logger.info(f"Extracting text from PDF: {file_path}")
            extracted_data = self._extract_text(file_path)
            full_text = extracted_data["text"]
            pages = extracted_data["pages"]
            total_pages = extracted_data["total_pages"]

            if not full_text.strip():
                logger.error(f"No text extracted from document: {document_id}")
                await self.document_repo.update_status(
                    document_id, ProcessingStatus.FAILED,
                    error_message="No text could be extracted from the PDF"
                )
                return False

            # Step 2: Clean text
            logger.info("Cleaning extracted text")
            cleaned_text = self._clean_text(full_text)

            # Step 3: Chunk text
            logger.info("Splitting text into chunks")
            chunks = self._chunk_text(cleaned_text, pages)
            total_chunks = len(chunks)

            if total_chunks == 0:
                logger.error(f"No chunks generated for document: {document_id}")
                await self.document_repo.update_status(
                    document_id, ProcessingStatus.FAILED,
                    error_message="No chunks could be generated"
                )
                return False

            # Step 4 & 5: Generate embeddings and index in vector store
            logger.info(f"Indexing {total_chunks} chunks in vector store")
            self.vector_store.add_chunks(chunks, document_id)

            # Step 6: Classify document
            category = "Uncategorized"
            confidence = None
            if self.classification_service:
                try:
                    classification = self.classification_service.classify_text(
                        cleaned_text[:5000]  # Use first 5000 chars for classification
                    )
                    category = classification["predicted_category"]
                    confidence = classification["confidence"]
                    logger.info(f"Document classified as: {category} (confidence: {confidence:.2f})")
                except Exception as e:
                    logger.warning(f"Classification failed: {e}")

            # Compute content hash
            content_hash = hashlib.sha256(full_text.encode()).hexdigest()

            # Step 7: Update metadata
            await self.document_repo.update_processing_metadata(
                document_id=document_id,
                total_pages=total_pages,
                total_chunks=total_chunks,
                category=category,
                classification_confidence=confidence,
                content_hash=content_hash,
            )

            # Mark as completed
            await self.document_repo.update_status(
                document_id, ProcessingStatus.COMPLETED
            )

            processing_time = time.time() - start_time
            logger.info(
                f"Document {document_id} processed successfully in {processing_time:.2f}s. "
                f"Pages: {total_pages}, Chunks: {total_chunks}, Category: {category}"
            )
            return True

        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            await self.document_repo.update_status(
                document_id, ProcessingStatus.FAILED,
                error_message=str(e)
            )
            return False

    def _extract_text(self, file_path: str) -> dict:
        """Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            dict: Extracted text, per-page texts, and total pages.
        """
        pages = []
        full_text = ""

        with fitz.open(file_path) as pdf_document:
            total_pages = len(pdf_document)

            for page_num in range(total_pages):
                page = pdf_document[page_num]
                page_text = page.get_text()

                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                })
                full_text += page_text + "\n\n"

        return {
            "text": full_text,
            "pages": pages,
            "total_pages": total_pages,
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.

        Args:
            text: Raw extracted text.

        Returns:
            str: Cleaned text.
        """
        import re

        # Remove null bytes
        text = text.replace("\x00", "")

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove page numbers and headers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\'\"]', ' ', text)

        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    def _chunk_text(self, text: str, pages: list[dict]) -> list[dict]:
        """Split text into chunks with page tracking.

        Uses RecursiveCharacterTextSplitter for intelligent chunking.

        Args:
            text: Cleaned text content.
            pages: List of page data with page numbers.

        Returns:
            list[dict]: List of chunks with content, page number, and index.
        """
        # Create page mapping for tracking page numbers
        page_texts = [p["text"] for p in pages]
        page_boundaries = []
        char_count = 0
        for page_text in page_texts:
            page_boundaries.append(char_count)
            char_count += len(page_text) + 2  # +2 for the \n\n separator

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        result = []
        for i, chunk in enumerate(chunks):
            # Find the approximate page number for this chunk
            chunk_start = text.find(chunk[:50]) if chunk else -1
            page_number = self._find_page_number(chunk_start, page_boundaries)

            result.append({
                "content": chunk,
                "page_number": page_number,
                "chunk_index": i,
            })

        return result

    def _find_page_number(self, char_position: int, page_boundaries: list[int]) -> int:
        """Find the page number for a given character position.

        Args:
            char_position: Character position in the full text.
            page_boundaries: List of character positions where pages start.

        Returns:
            int: Page number (1-based).
        """
        if char_position < 0:
            return 1

        for i, boundary in enumerate(page_boundaries):
            if char_position < boundary:
                return i + 1

        return len(page_boundaries)

    async def reprocess_document(self, document_id: str) -> bool:
        """Reprocess an existing document.

        Args:
            document_id: Document UUID.

        Returns:
            bool: True if reprocessing succeeded.
        """
        logger.info(f"Reprocessing document: {document_id}")

        # Delete existing chunks from vector store
        self.vector_store.delete_document_chunks(document_id)

        # Reset document status
        await self.document_repo.update_status(
            document_id, ProcessingStatus.REPROCESSING
        )

        # Process document again
        return await self.process_document(document_id)
