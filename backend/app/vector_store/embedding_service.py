"""
Embedding Service Module.

Provides text embedding generation using sentence-transformers.
"""

import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from backend.app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""

    _instance: Optional["EmbeddingService"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls) -> "EmbeddingService":
        """Singleton pattern to ensure only one model is loaded."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the embedding service."""
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            start_time = time.time()
            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device="cpu",
            )
            load_time = time.time() - start_time
            logger.info(
                f"Embedding model loaded in {load_time:.2f}s. "
                f"Dimension: {self.dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            int: Embedding dimension.
        """
        if self._model:
            if hasattr(self._model, "get_embedding_dimension"):
                return self._model.get_embedding_dimension()
            return self._model.get_sentence_embedding_dimension()
        return settings.EMBEDDING_DIMENSION

    def encode(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text.

        Returns:
            list[float]: Embedding vector.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension

        embedding = self._model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        )
        return [emb.tolist() for emb in embeddings]

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))
