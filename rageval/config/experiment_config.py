"""Experiment configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    strategy: str = "recursive"
    chunk_size: int = 512
    overlap: int = 50
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"


@dataclass
class IndexingConfig:
    """Configuration for vector indexing."""

    type: str = "faiss"
    index_type: str = "Flat"  # Flat, IVFFlat, HNSW
    nlist: int = 100  # For IVF indexes
    nprobe: int = 10  # For IVF indexes
    ef_construction: int = 200  # For HNSW
    ef_search: int = 50  # For HNSW
    m: int = 16  # For HNSW


@dataclass
class RerankingConfig:
    """Configuration for reranking."""

    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 10
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""

    top_k: int = 10


@dataclass
class GenerationConfig:
    """Configuration for answer generation."""

    model: str = "openai/gpt-3.5-turbo"
    max_tokens: int = 256
    temperature: float = 0.0
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""

    retrieval: list[str] = field(default_factory=lambda: ["precision@k", "recall@k", "mrr"])
    answer: list[str] = field(default_factory=lambda: ["exact_match", "f1"])
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""

    name: str = "default_experiment"
    description: str = ""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output_dir: str = "results"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from a dictionary."""
        config = cls()

        if "name" in data:
            config.name = data["name"]
        if "description" in data:
            config.description = data["description"]
        if "output_dir" in data:
            config.output_dir = data["output_dir"]
        if "seed" in data:
            config.seed = data["seed"]

        if "chunking" in data:
            config.chunking = ChunkingConfig(**data["chunking"])
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "indexing" in data:
            config.indexing = IndexingConfig(**data["indexing"])
        if "reranking" in data:
            config.reranking = RerankingConfig(**data["reranking"])
        if "retrieval" in data:
            config.retrieval = RetrievalConfig(**data["retrieval"])
        if "generation" in data:
            config.generation = GenerationConfig(**data["generation"])
        if "metrics" in data:
            config.metrics = MetricsConfig(**data["metrics"])

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "overlap": self.chunking.overlap,
                "separators": self.chunking.separators,
            },
            "embedding": {
                "model": self.embedding.model,
                "batch_size": self.embedding.batch_size,
                "normalize": self.embedding.normalize,
                "device": self.embedding.device,
            },
            "indexing": {
                "type": self.indexing.type,
                "index_type": self.indexing.index_type,
                "nlist": self.indexing.nlist,
                "nprobe": self.indexing.nprobe,
                "ef_construction": self.indexing.ef_construction,
                "ef_search": self.indexing.ef_search,
                "m": self.indexing.m,
            },
            "reranking": {
                "enabled": self.reranking.enabled,
                "model": self.reranking.model,
                "top_k": self.reranking.top_k,
                "batch_size": self.reranking.batch_size,
            },
            "retrieval": {
                "top_k": self.retrieval.top_k,
            },
            "generation": {
                "model": self.generation.model,
                "max_tokens": self.generation.max_tokens,
                "temperature": self.generation.temperature,
                "api_key_env": self.generation.api_key_env,
            },
            "metrics": {
                "retrieval": self.metrics.retrieval,
                "answer": self.metrics.answer,
                "k_values": self.metrics.k_values,
            },
            "output_dir": self.output_dir,
            "seed": self.seed,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
