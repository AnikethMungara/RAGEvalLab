"""FAISS vector index implementations."""

import json
from pathlib import Path

import faiss
import numpy as np

from rageval.config import IndexingConfig
from rageval.indexing.base import BaseIndex, SearchResult


class FaissIndex(BaseIndex):
    """FAISS-based vector index with multiple index type support."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        nlist: int = 100,
        nprobe: int = 10,
        ef_construction: int = 200,
        ef_search: int = 50,
        m: int = 16,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m

        self._index = self._create_index()
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}
        self._current_id = 0

    def _create_index(self) -> faiss.Index:
        """Create the appropriate FAISS index based on index_type."""
        index_type = self.index_type.lower()

        if index_type == "flat":
            return faiss.IndexFlatIP(self.dimension)

        elif index_type == "ivfflat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            return index

        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, self.m)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add embeddings to the index."""
        if len(embeddings) != len(ids):
            raise ValueError("Number of embeddings must match number of IDs")

        embeddings = embeddings.astype(np.float32)

        if not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings)

        if self.index_type.lower() == "ivfflat" and not self._index.is_trained:
            self._index.train(embeddings)
            self._index.nprobe = self.nprobe

        self._index.add(embeddings)

        for i, id_ in enumerate(ids):
            internal_id = self._current_id + i
            self._id_map[internal_id] = id_
            self._reverse_id_map[id_] = internal_id

        self._current_id += len(ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Search for nearest neighbors."""
        query = query_embedding.astype(np.float32).reshape(1, -1)

        if not query.flags["C_CONTIGUOUS"]:
            query = np.ascontiguousarray(query)

        top_k = min(top_k, self.size)
        if top_k == 0:
            return []

        scores, indices = self._index.search(query, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx in self._id_map:
                results.append(
                    SearchResult(
                        chunk_id=self._id_map[idx],
                        score=float(score),
                        rank=rank,
                    )
                )

        return results

    def batch_search(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> list[list[SearchResult]]:
        """Batch search for nearest neighbors."""
        queries = query_embeddings.astype(np.float32)

        if not queries.flags["C_CONTIGUOUS"]:
            queries = np.ascontiguousarray(queries)

        top_k = min(top_k, self.size)
        if top_k == 0:
            return [[] for _ in range(len(queries))]

        scores, indices = self._index.search(queries, top_k)

        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for rank, (score, idx) in enumerate(zip(query_scores, query_indices)):
                if idx >= 0 and idx in self._id_map:
                    results.append(
                        SearchResult(
                            chunk_id=self._id_map[idx],
                            score=float(score),
                            rank=rank,
                        )
                    )
            all_results.append(results)

        return all_results

    def save(self, path: str) -> None:
        """Save the index and ID mappings to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path.with_suffix(".faiss")))

        metadata = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "current_id": self._current_id,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "m": self.m,
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f)

    def load(self, path: str) -> None:
        """Load the index and ID mappings from disk."""
        path = Path(path)

        self._index = faiss.read_index(str(path.with_suffix(".faiss")))

        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)

        self._id_map = {int(k): v for k, v in metadata["id_map"].items()}
        self._reverse_id_map = {v: k for k, v in self._id_map.items()}
        self._current_id = metadata["current_id"]

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal


def create_index(config: IndexingConfig, dimension: int) -> BaseIndex:
    """Factory function to create an index from config."""
    return FaissIndex(
        dimension=dimension,
        index_type=config.index_type,
        nlist=config.nlist,
        nprobe=config.nprobe,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
        m=config.m,
    )
