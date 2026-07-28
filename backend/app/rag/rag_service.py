"""
RAG Service Module.

Provides Retrieval-Augmented Generation capabilities including
question answering, summarization, and document comparison.
"""

import time
import json
from typing import Optional

from loguru import logger

from backend.app.core.config import settings
from backend.app.services.llm_service import LLMService
from backend.app.vector_store.vector_store_service import VectorStoreService
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.conversation_repository import ConversationRepository
from backend.app.repositories.analytics_repository import AnalyticsRepository


class RAGService:
    """Service for Retrieval-Augmented Generation operations."""

    def __init__(
        self,
        llm_service: LLMService,
        vector_store: VectorStoreService,
        document_repo: DocumentRepository,
        conversation_repo: ConversationRepository,
        analytics_repo: AnalyticsRepository,
    ):
        """Initialize RAG service.

        Args:
            llm_service: LLM service instance.
            vector_store: Vector store service.
            document_repo: Document repository.
            conversation_repo: Conversation repository.
            analytics_repo: Analytics repository.
        """
        self.llm = llm_service
        self.vector_store = vector_store
        self.document_repo = document_repo
        self.conversation_repo = conversation_repo
        self.analytics_repo = analytics_repo

    async def answer_question(
        self,
        query: str,
        session_id: str,
        document_ids: Optional[list[str]] = None,
        search_mode: str = "semantic",
        top_k: int = 5,
    ) -> dict:
        """Answer a question using RAG.

        Args:
            query: User question.
            session_id: Conversation session ID.
            document_ids: Optional document filter.
            search_mode: Search strategy (keyword, semantic, hybrid).
            top_k: Number of chunks to retrieve.

        Returns:
            dict: Answer with sources and confidence.
        """
        start_time = time.time()
        logger.info(f"RAG question: '{query[:100]}...' mode={search_mode}")

        # Record analytics event
        await self.analytics_repo.record_event(
            event_type="question",
            query_text=query,
            session_id=session_id,
        )

        # Get or create conversation
        conversation = await self.conversation_repo.get_or_create_conversation(session_id)

        # Save user message
        await self.conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=query,
            message_type="question",
        )

        # Get conversation history for context
        history = await self.conversation_repo.get_session_history(session_id, limit=10)
        history_context = self._format_history(history)

        # Retrieve relevant chunks
        retrieved_chunks = self._retrieve_chunks(query, search_mode, top_k, document_ids)
        retrieved_chunks = await self._enrich_chunks(retrieved_chunks)
        retrieved_chunks = self._filter_by_query_intent(query, retrieved_chunks)

        if not retrieved_chunks:
            answer = "I cannot determine the answer from the uploaded documents."
            sources = []
            confidence = 0.0
            retrieved_context = []
        else:
            # Generate answer using LLM
            answer, sources, confidence = self._generate_answer(
                query, retrieved_chunks[:top_k], history_context
            )
            if self._is_llm_failure(answer):
                answer = self._build_grounded_fallback_answer(retrieved_chunks)
            retrieved_context = sources

        # Save assistant message
        metadata = json.dumps({
            "sources": sources,
            "confidence": confidence,
            "search_mode": search_mode,
        })
        await self.conversation_repo.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            message_type="answer",
            metadata_json=metadata,
        )

        processing_time = (time.time() - start_time) * 1000

        # Record analytics
        await self.analytics_repo.record_event(
            event_type="answer",
            document_id=sources[0]["document_id"] if sources else None,
            session_id=session_id,
            query_text=query,
            response_time_ms=processing_time,
            tokens_used=len(answer.split()),
        )

        logger.info(f"RAG answer generated in {processing_time:.0f}ms. Confidence: {confidence:.2f}")

        return {
            "answer": answer,
            "retrieved_context": retrieved_context,
            "sources": sources,
            "confidence_score": confidence,
            "session_id": session_id,
            "processing_time_ms": processing_time,
        }

    def _retrieve_chunks(
        self,
        query: str,
        search_mode: str,
        top_k: int,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Retrieve relevant chunks based on search mode.

        Args:
            query: Search query.
            search_mode: Search strategy.
            top_k: Number of results.
            document_ids: Optional document filter.

        Returns:
            list[dict]: Retrieved chunks.
        """
        if search_mode == "keyword":
            return self.vector_store.keyword_search(query, top_k, document_ids)
        if search_mode == "hybrid":
            return self.vector_store.hybrid_search(query, top_k, document_ids)
        return self.vector_store.semantic_search(query, top_k, document_ids)

    async def _enrich_chunks(self, chunks: list[dict]) -> list[dict]:
        """Attach document names to retrieved chunks for citations."""
        metadata_by_id: dict[str, dict[str, str]] = {}
        for chunk in chunks:
            document_id = chunk.get("document_id")
            if not document_id:
                continue
            if document_id not in metadata_by_id:
                document = await self.document_repo.get_by_id(document_id)
                metadata_by_id[document_id] = {
                    "name": document.original_filename if document else document_id,
                    "category": document.category if document else "Uncategorized",
                }
            chunk["document_name"] = metadata_by_id[document_id]["name"]
            chunk["category"] = metadata_by_id[document_id]["category"]
        return chunks

    def _filter_by_query_intent(self, query: str, chunks: list[dict]) -> list[dict]:
        """Remove category-mismatched chunks for explicit topic queries."""
        query_lower = f" {query.lower()} "
        asks_ai = (
            " artificial intelligence " in query_lower
            or " ai " in query_lower
            or " neural " in query_lower
            or " intelligent " in query_lower
        )
        asks_security = any(
            term in query_lower
            for term in [
                " cyber",
                " cybersecurity",
                " security",
                " malware",
                " encryption",
                " threat",
                " attack",
            ]
        )

        filtered = chunks
        if asks_ai and not asks_security:
            filtered = [
                chunk
                for chunk in filtered
                if chunk.get("category") != "Cyber Security"
            ]

        return sorted(
            filtered,
            key=lambda chunk: chunk.get("similarity_score", 0.0),
            reverse=True,
        )

    def _generate_answer(
        self,
        query: str,
        chunks: list[dict],
        history_context: str,
    ) -> tuple[str, list[dict], float]:
        """Generate an answer using LLM with retrieved context.

        Args:
            query: User question.
            chunks: Retrieved document chunks.
            history_context: Conversation history.

        Returns:
            tuple: (answer, sources, confidence_score)
        """
        # Prepare context from chunks
        context_parts = []
        sources = []

        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Source {i + 1}] Document: {chunk.get('document_name', chunk['document_id'])}\n"
                f"Page: {chunk['page_number']}\n"
                f"Content: {chunk['content']}\n"
            )
            sources.append({
                "document_id": chunk["document_id"],
                "document_name": chunk.get("document_name", chunk["document_id"]),
                "page_number": chunk["page_number"],
                "chunk_content": chunk["content"][:500],
                "similarity_score": chunk["similarity_score"],
            })

        context = "\n---\n".join(context_parts)

        scores = [max(0.0, min(c["similarity_score"], 1.0)) for c in chunks]
        avg_similarity = sum(scores) / len(scores)
        max_similarity = max(scores)
        coverage = min(len(chunks) / 3, 1.0)
        confidence = min(
            1.0,
            (0.55 * avg_similarity) + (0.35 * max_similarity) + (0.10 * coverage),
        )

        if not self.llm.is_available():
            return self._build_grounded_fallback_answer(chunks), sources, confidence

        system_prompt = """You are an AI Research Assistant. Your role is to answer questions based ONLY on the provided document context.

RULES:
1. Answer ONLY using the information from the provided context.
2. If the context does not contain enough information to answer, say: "I cannot determine the answer from the uploaded documents."
3. Always cite your sources using [Source X] notation.
4. Include page numbers when available.
5. Be concise and accurate.
6. Do not make up information or use external knowledge."""

        prompt = f"""Conversation History:
{history_context}

Retrieved Document Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Include source citations."""

        answer = self.llm.generate(prompt, system_prompt=system_prompt)

        return answer, sources, confidence

    def _is_llm_failure(self, text: str) -> bool:
        """Return True when provider output is an infrastructure failure message."""
        lowered = text.lower()
        failure_markers = [
            "llm is not configured",
            "error generating response",
            "failed to initialize",
            "api key",
        ]
        return any(marker in lowered for marker in failure_markers)

    def _build_grounded_fallback_answer(self, chunks: list[dict]) -> str:
        """Build a concise extractive answer when an LLM provider is unavailable."""
        if not chunks:
            return "I cannot determine the answer from the uploaded documents."

        sentences = []
        for index, chunk in enumerate(chunks[:3], start=1):
            content = " ".join(chunk["content"].split())
            snippet = content[:450].rstrip()
            sentences.append(
                f"[Source {index}, page {chunk['page_number']}] {snippet}"
            )

        return (
            "Based on the retrieved document context:\n\n"
            + "\n\n".join(sentences)
        )

    def _format_history(self, messages: list) -> str:
        """Format conversation history for LLM context.

        Args:
            messages: List of message objects.

        Returns:
            str: Formatted history.
        """
        if not messages:
            return "No previous conversation."

        formatted = []
        for msg in messages[-6:]:  # Last 6 messages for context
            role = msg.role.capitalize()
            content = msg.content[:200]  # Truncate long messages
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    async def generate_summary(
        self,
        document_id: str,
        summary_type: str = "executive",
        max_length: int = 500,
    ) -> dict:
        """Generate a summary of a document.

        Args:
            document_id: Document ID.
            summary_type: Type of summary.
            max_length: Maximum words.

        Returns:
            dict: Summary response.
        """
        start_time = time.time()
        logger.info(f"Generating {summary_type} summary for document {document_id}")

        # Get document
        document = await self.document_repo.get_by_id(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")

        # Retrieve all chunks for the document
        chunks = self.vector_store.get_document_chunks(document_id, limit=100)

        if not chunks:
            raise ValueError(f"No content found for document: {document_id}")

        # Combine chunk content
        full_content = " ".join([c["content"] for c in chunks])

        # Truncate if too long
        if len(full_content) > 10000:
            full_content = full_content[:10000]

        summary_prompts = {
            "executive": "Provide an executive summary of the following document. Focus on key findings, conclusions, and main points.",
            "technical": "Provide a technical summary of the following document. Focus on methodology, technical details, and implementation.",
            "bullet": "Provide a bullet-point summary of the following document. List the key points concisely.",
            "key_takeaways": "Extract the key takeaways from the following document. What are the most important lessons or insights?",
        }

        prompt_template = summary_prompts.get(
            summary_type,
            summary_prompts["executive"],
        )

        prompt = f"""{prompt_template}

Document Content:
{full_content}

Please provide a summary in approximately {max_length} words."""

        if self.llm.is_available():
            summary = self.llm.generate(prompt, max_tokens=max_length * 4)
        else:
            summary = self._build_summary_fallback(full_content, summary_type, max_length)
        if self._is_llm_failure(summary):
            summary = self._build_summary_fallback(full_content, summary_type, max_length)

        processing_time = (time.time() - start_time) * 1000

        return {
            "document_id": document_id,
            "document_name": document.original_filename,
            "summary_type": summary_type,
            "summary": summary,
            "word_count": len(summary.split()),
            "processing_time_ms": processing_time,
        }

    def _build_summary_fallback(
        self,
        content: str,
        summary_type: str,
        max_length: int,
    ) -> str:
        """Create a grounded extractive summary from retrieved document content."""
        words = content.split()
        trimmed = " ".join(words[:max_length])
        if summary_type == "bullet":
            sentences = [s.strip() for s in trimmed.split(".") if s.strip()][:6]
            return "\n".join(f"- {sentence}." for sentence in sentences)
        if summary_type == "key_takeaways":
            sentences = [s.strip() for s in trimmed.split(".") if s.strip()][:5]
            return "\n".join(f"Key takeaway {idx}: {sentence}." for idx, sentence in enumerate(sentences, 1))
        return trimmed

    async def compare_documents(
        self,
        document_ids: list[str],
        comparison_aspects: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Compare multiple documents.

        Args:
            document_ids: List of document IDs to compare.
            comparison_aspects: Specific aspects to compare.
            session_id: Optional session ID.

        Returns:
            dict: Comparison results.
        """
        start_time = time.time()
        logger.info(f"Comparing documents: {document_ids}")

        # Get documents
        documents = []
        for doc_id in document_ids:
            doc = await self.document_repo.get_by_id(doc_id)
            if doc:
                documents.append(doc)

        if len(documents) < 2:
            raise ValueError("At least 2 valid documents are required for comparison")

        # Retrieve content for each document
        doc_contents = {}
        for doc in documents:
            chunks = self.vector_store.get_document_chunks(doc.id, limit=100)
            content = " ".join([c["content"] for c in chunks])
            if len(content) > 5000:
                content = content[:5000]
            doc_contents[doc.original_filename] = content

        # Default comparison aspects
        if not comparison_aspects:
            comparison_aspects = [
                "Methodology",
                "Advantages",
                "Disadvantages",
                "Similarities",
                "Differences",
                "Implementation",
                "Conclusion",
            ]

        # Generate comparison
        doc_text = "\n\n".join([
            f"=== {name} ===\n{content}"
            for name, content in doc_contents.items()
        ])

        aspects_text = ", ".join(comparison_aspects)

        prompt = f"""Compare the following documents across these aspects: {aspects_text}

Documents:
{doc_text}

For each aspect, provide a structured comparison. Format the response as a comparison table."""

        if self.llm.is_available():
            comparison = self.llm.generate(prompt, max_tokens=2048)
        else:
            comparison = self._build_comparison_fallback(doc_contents, comparison_aspects)
        if self._is_llm_failure(comparison):
            comparison = self._build_comparison_fallback(doc_contents, comparison_aspects)

        # Generate per-aspect breakdown
        aspects = []
        for aspect in comparison_aspects:
            aspect_data = {}
            for doc_name in doc_contents.keys():
                aspect_prompt = f"""Based on the document "{doc_name}", what is discussed about "{aspect}"? 
Document content: {doc_contents[doc_name][:2000]}"""
                if self.llm.is_available():
                    aspect_content = self.llm.generate(aspect_prompt, max_tokens=300)
                else:
                    aspect_content = self._extract_aspect_snippet(
                        doc_contents[doc_name], aspect
                    )
                if self._is_llm_failure(aspect_content):
                    aspect_content = self._extract_aspect_snippet(
                        doc_contents[doc_name], aspect
                    )
                aspect_data[doc_name] = aspect_content

            aspects.append({
                "aspect": aspect,
                "documents": aspect_data,
            })

        processing_time = (time.time() - start_time) * 1000

        return {
            "comparison": comparison,
            "aspects": aspects,
            "documents_compared": [doc.original_filename for doc in documents],
            "processing_time_ms": processing_time,
        }

    def _build_comparison_fallback(
        self,
        doc_contents: dict[str, str],
        aspects: list[str],
    ) -> str:
        """Create a markdown comparison table from retrieved content."""
        header = "| Aspect | " + " | ".join(doc_contents.keys()) + " |"
        separator = "|---|" + "|".join(["---"] * len(doc_contents)) + "|"
        rows = []
        for aspect in aspects:
            cells = [
                self._extract_aspect_snippet(content, aspect)
                for content in doc_contents.values()
            ]
            rows.append(f"| {aspect} | " + " | ".join(cells) + " |")
        return "\n".join([header, separator, *rows])

    def _extract_aspect_snippet(self, content: str, aspect: str) -> str:
        """Extract a short aspect-relevant snippet from document content."""
        normalized = " ".join(content.split())
        terms = aspect.lower().split()
        sentences = [s.strip() for s in normalized.split(".") if s.strip()]
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in terms):
                return sentence[:280]
        return normalized[:280] if normalized else "No relevant content retrieved."

    async def chat(
        self,
        message: str,
        session_id: str,
        document_ids: Optional[list[str]] = None,
    ) -> dict:
        """Handle a chat message with context awareness.

        Args:
            message: User message.
            session_id: Session ID.
            document_ids: Optional document filter.

        Returns:
            dict: Chat response.
        """
        start_time = time.time()

        # Get conversation
        conversation = await self.conversation_repo.get_or_create_conversation(session_id)

        # Save user message
        await self.conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=message,
            message_type="text",
        )

        # Get history
        history = await self.conversation_repo.get_session_history(session_id, limit=10)
        history_context = self._format_history(history)

        # Check if this is a follow-up question (pronoun resolution)
        is_follow_up = self._is_follow_up(message, history)

        # Retrieve relevant context if needed
        sources = None
        if not is_follow_up or document_ids:
            chunks = self._retrieve_chunks(message, "semantic", 5, document_ids)
            chunks = self._filter_by_query_intent(
                message,
                await self._enrich_chunks(chunks),
            )
            if chunks:
                context = "\n".join([c["content"] for c in chunks])
                sources = [{
                    "document_id": c["document_id"],
                    "document_name": c.get("document_name", c["document_id"]),
                    "page_number": c["page_number"],
                    "chunk_content": c["content"][:500],
                    "similarity_score": c["similarity_score"],
                } for c in chunks]
            else:
                context = ""
        else:
            context = ""

        system_prompt = """You are an AI Research Assistant. You help users understand their documents.
Be conversational, helpful, and accurate. If you don't know something, say so."""

        prompt = f"""Conversation History:
{history_context}

{"Relevant Document Context:" + context if context else ""}

User: {message}

Respond helpfully and conversationally."""

        if self.llm.is_available():
            response = self.llm.generate(prompt, system_prompt=system_prompt)
        elif sources:
            response = self._build_grounded_fallback_answer(chunks)
        else:
            response = "I cannot determine the answer from the uploaded documents."

        # Save assistant message
        await self.conversation_repo.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response,
            message_type="text",
        )

        processing_time = (time.time() - start_time) * 1000

        return {
            "response": response,
            "session_id": session_id,
            "sources": sources,
            "processing_time_ms": processing_time,
        }

    def _is_follow_up(self, message: str, history: list) -> bool:
        """Check if a message is a follow-up question.

        Args:
            message: Current message.
            history: Conversation history.

        Returns:
            bool: True if this is a follow-up.
        """
        if not history:
            return False

        follow_up_indicators = [
            "its", "it", "this", "that", "these", "those",
            "what about", "how about", "explain more",
            "elaborate", "tell me more", "continue",
        ]

        message_lower = message.lower()
        return any(
            indicator in message_lower
            for indicator in follow_up_indicators
        )
