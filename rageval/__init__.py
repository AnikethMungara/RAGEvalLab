"""RAGEvalLab - Evaluation and benchmarking framework for RAG systems."""

from rageval.chunking import Chunk, Document, create_chunker
from rageval.config import ExperimentConfig
from rageval.embedding import create_embedder
from rageval.evaluator import EvalDataset, EvalQuery, EvaluationReport, RAGEvaluator
from rageval.generation import create_generator
from rageval.indexing import create_index
from rageval.pipeline import RAGPipeline, RAGResult
from rageval.reranking import create_reranker

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "RAGEvaluator",
    "RAGPipeline",
    "RAGResult",
    "ExperimentConfig",
    # Data classes
    "Document",
    "Chunk",
    "EvalDataset",
    "EvalQuery",
    "EvaluationReport",
    # Factory functions
    "create_chunker",
    "create_embedder",
    "create_index",
    "create_reranker",
    "create_generator",
]
