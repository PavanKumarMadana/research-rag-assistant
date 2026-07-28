"""
Vector Store Service Module.

Provides ChromaDB-based vector storage and retrieval operations.
"""

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from backend.app.core.config import settings
from backend.app.vector_store.embedding_service import EmbeddingService


class VectorStoreService:
    """Service for vector storage and semantic search using ChromaDB."""

    _instance: Optional["VectorStoreService"] = None
    _client: Optional[chromadb.Client] = None
    _collection = None

    def __new__(cls) -> "VectorStoreService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the vector store."""
        if self._client is None:
            self._initialize()
        self.embedding_service = EmbeddingService()

    def _initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            logger.info(f"Initializing ChromaDB at: {settings.VECTOR_DB_PATH}")
            self._client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            collection_name = "document_chunks"
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Loaded vector collection: {collection_name}")

            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_chunks(
        self,
        chunks: list[dict],
        document_id: str,
    ) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of chunk dicts with 'content', 'page_number', 'chunk_index'.
            document_id: Source document ID.

        Returns:
            int: Number of chunks added.
        """
        if not chunks:
            return 0

        try:
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedding_service.encode_batch(texts)

            ids = []
            metadatas = []
            documents = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{chunk['chunk_index']}"
                ids.append(chunk_id)
                documents.append(chunk["content"])
                metadatas.append({
                    "document_id": document_id,
                    "page_number": chunk.get("page_number", 0),
                    "chunk_index": chunk.get("chunk_index", i),
                    "chunk_id": chunk_id,
                })

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            raise

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Perform semantic search using embeddings.

        Args:
            query: Search query.
            top_k: Number of results to return.
            document_ids: Optional filter by document IDs.

        Returns:
            list[dict]: Search results with content, metadata, and scores.
        """
        try:
            query_embedding = self.embedding_service.encode(query)

            where_filter = None
            if document_ids:
                where_filter = {"document_id": {"$in": document_ids}}

            collection_count = self._collection.count()
            if collection_count == 0:
                return []

            candidate_count = min(
                collection_count,
                max(top_k, top_k * settings.RETRIEVAL_CANDIDATE_MULTIPLIER),
            )
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_count,
                where=where_filter,
                include=["documents", "metadatas", "distances", "embeddings"],
            )

            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    similarity = 1 - (results["distances"][0][i] if results["distances"] else 0)
                    if similarity < settings.SIMILARITY_THRESHOLD:
                        continue
                    formatted_results.append({
                        "chunk_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "document_id": results["metadatas"][0][i].get("document_id", ""),
                        "page_number": results["metadatas"][0][i].get("page_number", 0),
                        "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                        "similarity_score": similarity,
                        "_embedding": results["embeddings"][0][i] if results.get("embeddings") else None,
                    })

            reranked = self._max_marginal_relevance(
                formatted_results,
                query_embedding,
                top_k,
            )
            return [
                {key: value for key, value in result.items() if key != "_embedding"}
                for result in reranked
            ]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Perform keyword-based search.

        Args:
            query: Search query.
            top_k: Number of results.
            document_ids: Optional document filter.

        Returns:
            list[dict]: Search results.
        """
        try:
            where_filter = None
            if document_ids:
                where_filter = {"document_id": {"$in": document_ids}}

            results = self._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )

            formatted_results = []
            query_terms = [
                term.lower()
                for term in query.split()
                if len(term.strip()) > 1
            ]

            for chunk_id, document, metadata in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                content_lower = document.lower()
                exact_score = content_lower.count(query.lower())
                term_score = sum(content_lower.count(term) for term in query_terms)
                if exact_score == 0 and term_score == 0:
                    continue

                score = min((exact_score * 2 + term_score) / max(len(query_terms), 1), 1.0)
                if score < settings.SIMILARITY_THRESHOLD:
                    continue
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "content": document,
                    "document_id": metadata.get("document_id", ""),
                    "page_number": metadata.get("page_number", 0),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity_score": score,
                })

            return sorted(
                formatted_results,
                key=lambda item: item["similarity_score"],
                reverse=True,
            )[:top_k]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
        semantic_weight: float = 0.7,
    ) -> list[dict]:
        """Perform hybrid search combining semantic and keyword results.

        Args:
            query: Search query.
            top_k: Number of results.
            document_ids: Optional document filter.
            semantic_weight: Weight for semantic search (0-1).

        Returns:
            list[dict]: Combined and ranked results.
        """
        semantic_results = self.semantic_search(
            query,
            top_k * settings.RETRIEVAL_CANDIDATE_MULTIPLIER,
            document_ids,
        )
        keyword_results = self.keyword_search(
            query,
            top_k * settings.RETRIEVAL_CANDIDATE_MULTIPLIER,
            document_ids,
        )

        # Combine results with weighted scoring
        combined = {}
        for result in semantic_results:
            chunk_id = result["chunk_id"]
            combined[chunk_id] = result
            combined[chunk_id]["similarity_score"] *= semantic_weight

        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id in combined:
                combined[chunk_id]["similarity_score"] += (
                    result["similarity_score"] * (1 - semantic_weight)
                )
            else:
                result["similarity_score"] *= (1 - semantic_weight)
                combined[chunk_id] = result

        # Sort by combined score
        sorted_results = sorted(
            [
                result
                for result in combined.values()
                if result["similarity_score"] >= settings.SIMILARITY_THRESHOLD
            ],
            key=lambda x: x["similarity_score"],
            reverse=True,
        )

        return sorted_results[:top_k]

    def get_document_chunks(
        self,
        document_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Return stored chunks for one document without query relevance filtering."""
        try:
            results = self._collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
                limit=limit,
            )
            chunks = []
            for chunk_id, document, metadata in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": document,
                    "document_id": metadata.get("document_id", ""),
                    "page_number": metadata.get("page_number", 0),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity_score": 1.0,
                })
            return sorted(chunks, key=lambda item: item["chunk_index"])
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []

    def _max_marginal_relevance(
        self,
        candidates: list[dict],
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """Rerank candidates using Max Marginal Relevance."""
        if not candidates:
            return []

        remaining = sorted(
            candidates,
            key=lambda item: item["similarity_score"],
            reverse=True,
        )
        selected: list[dict] = []

        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(remaining):
                relevance = candidate["similarity_score"]
                candidate_embedding = candidate.get("_embedding") or query_embedding
                diversity_penalty = max(
                    self.embedding_service.similarity(
                        candidate_embedding,
                        selected_item.get("_embedding") or query_embedding,
                    )
                    for selected_item in selected
                )
                mmr_score = (
                    settings.MMR_LAMBDA * relevance
                    - (1 - settings.MMR_LAMBDA) * diversity_penalty
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index

            selected.append(remaining.pop(best_index))

        return sorted(
            selected,
            key=lambda item: item["similarity_score"],
            reverse=True,
        )

    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID to delete.

        Returns:
            bool: True if successful.
        """
        try:
            self._collection.delete(
                where={"document_id": document_id}
            )
            logger.info(f"Deleted chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False

    def get_collection_stats(self) -> dict:
        """Get collection statistics.

        Returns:
            dict: Collection stats.
        """
        try:
            count = self._collection.count()
            return {
                "total_chunks": count,
                "collection_name": self._collection.name,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_chunks": 0, "collection_name": "unknown"}
