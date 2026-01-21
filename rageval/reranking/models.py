"""Reranking model implementations."""

from rageval.config import RerankingConfig
from rageval.reranking.base import BaseReranker, RerankResult


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[tuple[str, str]],
        original_scores: list[float],
    ) -> list[RerankResult]:
        """Rerank chunks using cross-encoder scoring."""
        if not chunks:
            return []

        pairs = [(query, chunk_text) for _, chunk_text in chunks]

        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        results = []
        for i, ((chunk_id, _), score) in enumerate(zip(chunks, scores)):
            results.append(
                RerankResult(
                    chunk_id=chunk_id,
                    original_score=original_scores[i],
                    rerank_score=float(score),
                    original_rank=i,
                    new_rank=-1,  # Will be set after sorting
                )
            )

        results.sort(key=lambda x: x.rerank_score, reverse=True)

        for new_rank, result in enumerate(results):
            result.new_rank = new_rank

        return results


class NoOpReranker(BaseReranker):
    """No-op reranker that returns results in original order."""

    def rerank(
        self,
        query: str,
        chunks: list[tuple[str, str]],
        original_scores: list[float],
    ) -> list[RerankResult]:
        """Return results in original order."""
        results = []
        for i, ((chunk_id, _), score) in enumerate(zip(chunks, original_scores)):
            results.append(
                RerankResult(
                    chunk_id=chunk_id,
                    original_score=score,
                    rerank_score=score,
                    original_rank=i,
                    new_rank=i,
                )
            )
        return results


def create_reranker(config: RerankingConfig) -> BaseReranker:
    """Factory function to create a reranker from config."""
    if not config.enabled:
        return NoOpReranker()

    return CrossEncoderReranker(
        model_name=config.model,
        batch_size=config.batch_size,
    )
