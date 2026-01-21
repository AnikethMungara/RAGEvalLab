"""Base classes for embedding models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of shape (n_texts, dimension) containing embeddings.
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query text to embed.

        Returns:
            Array of shape (dimension,) containing the embedding.
        """
        return self.embed([query])[0]

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed multiple documents.

        Args:
            documents: List of document texts to embed.

        Returns:
            Array of shape (n_documents, dimension) containing embeddings.
        """
        return self.embed(documents)
