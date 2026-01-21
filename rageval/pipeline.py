"""RAG Pipeline implementation."""

import time
from dataclasses import dataclass, field

import numpy as np

from rageval.chunking import Chunk, create_chunker
from rageval.config import ExperimentConfig
from rageval.embedding import create_embedder
from rageval.generation import GenerationResult, create_generator
from rageval.indexing import SearchResult, create_index
from rageval.reranking import create_reranker


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    results: list[SearchResult]
    chunks: list[Chunk]
    latency_ms: float


@dataclass
class RAGResult:
    """Complete result of a RAG query."""

    query: str
    retrieval: RetrievalResult
    generation: GenerationResult
    total_latency_ms: float


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._embedder = None
        self._index = None
        self._reranker = None
        self._generator = None
        self._chunks: dict[str, Chunk] = {}
        self._is_indexed = False

    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            self._embedder = create_embedder(self.config.embedding)
        return self._embedder

    @property
    def index(self):
        """Lazy load index."""
        if self._index is None:
            self._index = create_index(self.config.indexing, self.embedder.dimension)
        return self._index

    @property
    def reranker(self):
        """Lazy load reranker."""
        if self._reranker is None:
            self._reranker = create_reranker(self.config.reranking)
        return self._reranker

    @property
    def generator(self):
        """Lazy load generator."""
        if self._generator is None:
            self._generator = create_generator(self.config.generation)
        return self._generator

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index a list of chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

        texts = [chunk.text for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        embeddings = self.embedder.embed_documents(texts)

        self.index.add(embeddings, ids)
        self._is_indexed = True

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results (defaults to config value).

        Returns:
            RetrievalResult with ranked chunks.
        """
        if not self._is_indexed:
            raise RuntimeError("No chunks indexed. Call index_chunks first.")

        top_k = top_k or self.config.retrieval.top_k

        start_time = time.perf_counter()

        query_embedding = self.embedder.embed_query(query)

        search_results = self.index.search(query_embedding, top_k)

        if self.config.reranking.enabled:
            chunk_data = [
                (r.chunk_id, self._chunks[r.chunk_id].text)
                for r in search_results
                if r.chunk_id in self._chunks
            ]
            original_scores = [r.score for r in search_results]

            rerank_results = self.reranker.rerank_top_k(
                query, chunk_data, original_scores, self.config.reranking.top_k
            )

            search_results = [
                SearchResult(
                    chunk_id=r.chunk_id,
                    score=r.rerank_score,
                    rank=r.new_rank,
                )
                for r in rerank_results
            ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        retrieved_chunks = [
            self._chunks[r.chunk_id]
            for r in search_results
            if r.chunk_id in self._chunks
        ]

        return RetrievalResult(
            query=query,
            results=search_results,
            chunks=retrieved_chunks,
            latency_ms=latency_ms,
        )

    def generate(self, query: str, chunks: list[Chunk]) -> GenerationResult:
        """Generate an answer from retrieved chunks.

        Args:
            query: The user query.
            chunks: Retrieved chunks to use as context.

        Returns:
            GenerationResult with the answer.
        """
        context_texts = [chunk.text for chunk in chunks]
        return self.generator.generate(query, context_texts)

    def query(self, query: str, top_k: int | None = None) -> RAGResult:
        """Execute a complete RAG query.

        Args:
            query: The user query.
            top_k: Number of chunks to retrieve (defaults to config).

        Returns:
            RAGResult with retrieval and generation results.
        """
        start_time = time.perf_counter()

        retrieval_result = self.retrieve(query, top_k)

        generation_result = self.generate(query, retrieval_result.chunks)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return RAGResult(
            query=query,
            retrieval=retrieval_result,
            generation=generation_result,
            total_latency_ms=total_latency_ms,
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> list[Chunk]:
        """Get all indexed chunks."""
        return list(self._chunks.values())


@dataclass
class PerformanceMetrics:
    """Performance metrics for a RAG pipeline."""

    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    avg_total_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    total_queries: int


def measure_performance(
    pipeline: RAGPipeline,
    queries: list[str],
    top_k: int | None = None,
) -> tuple[list[RAGResult], PerformanceMetrics]:
    """Measure pipeline performance across queries.

    Args:
        pipeline: The RAG pipeline to test.
        queries: List of queries to execute.
        top_k: Number of chunks to retrieve per query.

    Returns:
        Tuple of (results, performance_metrics).
    """
    results = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    start_time = time.perf_counter()

    for query in queries:
        result = pipeline.query(query, top_k)
        results.append(result)
        retrieval_latencies.append(result.retrieval.latency_ms)
        generation_latencies.append(result.generation.latency_ms or 0)
        total_latencies.append(result.total_latency_ms)

    total_time = time.perf_counter() - start_time

    total_latencies_sorted = sorted(total_latencies)
    n = len(total_latencies_sorted)

    metrics = PerformanceMetrics(
        avg_retrieval_latency_ms=np.mean(retrieval_latencies),
        avg_generation_latency_ms=np.mean(generation_latencies),
        avg_total_latency_ms=np.mean(total_latencies),
        p50_latency_ms=total_latencies_sorted[n // 2] if n > 0 else 0,
        p95_latency_ms=total_latencies_sorted[int(n * 0.95)] if n > 0 else 0,
        p99_latency_ms=total_latencies_sorted[int(n * 0.99)] if n > 0 else 0,
        throughput_qps=len(queries) / total_time if total_time > 0 else 0,
        total_queries=len(queries),
    )

    return results, metrics
