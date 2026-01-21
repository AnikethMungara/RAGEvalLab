"""Tests for vector indexing."""

import numpy as np
import pytest

from rageval.indexing import FaissIndex


class TestFaissIndex:
    def test_add_and_search(self):
        index = FaissIndex(dimension=4, index_type="Flat")

        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        ids = ["doc1", "doc2", "doc3"]

        index.add(embeddings, ids)

        assert index.size == 3

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "doc1"
        assert results[0].rank == 0

    def test_empty_index_search(self):
        index = FaissIndex(dimension=4, index_type="Flat")

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, top_k=5)

        assert len(results) == 0

    def test_batch_search(self):
        index = FaissIndex(dimension=4, index_type="Flat")

        embeddings = np.eye(4, dtype=np.float32)
        ids = ["doc1", "doc2", "doc3", "doc4"]
        index.add(embeddings, ids)

        queries = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        results = index.batch_search(queries, top_k=2)

        assert len(results) == 2
        assert results[0][0].chunk_id == "doc1"
        assert results[1][0].chunk_id == "doc2"

    def test_ivf_index(self):
        index = FaissIndex(dimension=4, index_type="IVFFlat", nlist=2, nprobe=2)

        embeddings = np.random.randn(100, 4).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        ids = [f"doc{i}" for i in range(100)]

        index.add(embeddings, ids)

        assert index.size == 100

        query = embeddings[0]
        results = index.search(query, top_k=5)

        assert len(results) == 5
        assert results[0].chunk_id == "doc0"

    def test_hnsw_index(self):
        index = FaissIndex(dimension=4, index_type="HNSW", m=8)

        embeddings = np.random.randn(50, 4).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        ids = [f"doc{i}" for i in range(50)]

        index.add(embeddings, ids)

        assert index.size == 50

    def test_save_and_load(self, tmp_path):
        index = FaissIndex(dimension=4, index_type="Flat")

        embeddings = np.eye(4, dtype=np.float32)
        ids = ["doc1", "doc2", "doc3", "doc4"]
        index.add(embeddings, ids)

        save_path = tmp_path / "test_index"
        index.save(str(save_path))

        new_index = FaissIndex(dimension=4, index_type="Flat")
        new_index.load(str(save_path))

        assert new_index.size == 4

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = new_index.search(query, top_k=1)

        assert results[0].chunk_id == "doc1"
