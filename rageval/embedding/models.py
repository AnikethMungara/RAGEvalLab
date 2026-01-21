"""Embedding model implementations."""

import numpy as np

from rageval.config import EmbeddingConfig
from rageval.embedding.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding model using sentence-transformers library."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using sentence-transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings


class RandomEmbedder(BaseEmbedder):
    """Random embedding model for testing purposes."""

    def __init__(self, dimension: int = 384, seed: int = 42):
        self._dimension = dimension
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate random embeddings."""
        embeddings = self._rng.random((len(texts), self._dimension)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms


def create_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """Factory function to create an embedder from config."""
    model_name = config.model.lower()

    if model_name == "random":
        return RandomEmbedder()
    else:
        return SentenceTransformerEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
            normalize=config.normalize,
            device=config.device,
        )
