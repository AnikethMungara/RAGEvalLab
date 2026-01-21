"""Base classes for reranking models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Represents a reranked result."""

    chunk_id: str
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int


class BaseReranker(ABC):
    """Abstract base class for reranking models."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[tuple[str, str]],  # List of (chunk_id, chunk_text)
        original_scores: list[float],
    ) -> list[RerankResult]:
        """Rerank a list of chunks based on the query.

        Args:
            query: The search query.
            chunks: List of (chunk_id, chunk_text) tuples.
            original_scores: Original retrieval scores.

        Returns:
            List of RerankResult objects sorted by new relevance.
        """
        pass

    def rerank_top_k(
        self,
        query: str,
        chunks: list[tuple[str, str]],
        original_scores: list[float],
        top_k: int,
    ) -> list[RerankResult]:
        """Rerank and return top-k results."""
        results = self.rerank(query, chunks, original_scores)
        return results[:top_k]
