"""Base classes for vector indexing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    """Represents a single search result."""

    chunk_id: str
    score: float
    rank: int


class BaseIndex(ABC):
    """Abstract base class for vector indexes."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add embeddings to the index.

        Args:
            embeddings: Array of shape (n, dimension) containing embeddings.
            ids: List of unique identifiers for each embedding.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Search for nearest neighbors.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        pass

    def batch_search(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> list[list[SearchResult]]:
        """Search for nearest neighbors for multiple queries.

        Args:
            query_embeddings: Array of shape (n_queries, dimension).
            top_k: Number of results to return per query.

        Returns:
            List of result lists, one per query.
        """
        results = []
        for query in query_embeddings:
            results.append(self.search(query, top_k))
        return results

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors in the index."""
        pass
