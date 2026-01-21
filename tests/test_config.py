"""Tests for configuration management."""

import pytest

from rageval.config import (
    ChunkingConfig,
    EmbeddingConfig,
    ExperimentConfig,
    IndexingConfig,
    MetricsConfig,
)


class TestExperimentConfig:
    def test_default_config(self):
        config = ExperimentConfig()

        assert config.name == "default_experiment"
        assert config.chunking.strategy == "recursive"
        assert config.chunking.chunk_size == 512
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.indexing.index_type == "Flat"
        assert config.retrieval.top_k == 10

    def test_from_dict(self):
        data = {
            "name": "test_experiment",
            "chunking": {
                "strategy": "fixed",
                "chunk_size": 256,
                "overlap": 25,
            },
            "embedding": {
                "model": "test-model",
                "batch_size": 64,
            },
            "retrieval": {
                "top_k": 5,
            },
        }

        config = ExperimentConfig.from_dict(data)

        assert config.name == "test_experiment"
        assert config.chunking.strategy == "fixed"
        assert config.chunking.chunk_size == 256
        assert config.embedding.model == "test-model"
        assert config.retrieval.top_k == 5

    def test_to_dict(self):
        config = ExperimentConfig(name="my_experiment")
        config.chunking.chunk_size = 1024

        data = config.to_dict()

        assert data["name"] == "my_experiment"
        assert data["chunking"]["chunk_size"] == 1024

    def test_yaml_round_trip(self, tmp_path):
        config = ExperimentConfig(name="yaml_test")
        config.chunking.strategy = "sentence"
        config.metrics.k_values = [1, 5, 10, 20]

        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)

        assert loaded.name == "yaml_test"
        assert loaded.chunking.strategy == "sentence"
        assert loaded.metrics.k_values == [1, 5, 10, 20]


class TestChunkingConfig:
    def test_default_separators(self):
        config = ChunkingConfig()

        assert "\n\n" in config.separators
        assert "\n" in config.separators


class TestMetricsConfig:
    def test_default_metrics(self):
        config = MetricsConfig()

        assert "precision@k" in config.retrieval
        assert "recall@k" in config.retrieval
        assert "mrr" in config.retrieval
        assert "exact_match" in config.answer
        assert "f1" in config.answer
