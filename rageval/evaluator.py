"""RAG Evaluation orchestration."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rageval.analysis import (
    ErrorAnalysisSummary,
    aggregate_error_analysis,
    analyze_single_query,
    format_error_report,
)
from rageval.chunking import Chunk, Document, create_chunker
from rageval.config import ExperimentConfig
from rageval.metrics import (
    aggregate_answer_metrics,
    aggregate_retrieval_metrics,
    calculate_answer_metrics,
    calculate_retrieval_metrics,
)
from rageval.pipeline import PerformanceMetrics, RAGPipeline, RAGResult, measure_performance


@dataclass
class EvalQuery:
    """A query for evaluation with ground truth."""

    query: str
    ground_truth_answer: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Evaluation dataset with documents and queries."""

    documents: list[Document]
    queries: list[EvalQuery]
    name: str = "unnamed"


@dataclass
class QueryResult:
    """Result for a single evaluated query."""

    query: str
    predicted_answer: str
    ground_truth_answer: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    retrieval_metrics: dict
    answer_metrics: dict


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    experiment_name: str
    config: dict
    retrieval_metrics: dict[str, float]
    answer_metrics: dict[str, float]
    performance_metrics: dict[str, float]
    error_analysis: dict
    query_results: list[QueryResult]
    timestamp: str

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "retrieval_metrics": self.retrieval_metrics,
            "answer_metrics": self.answer_metrics,
            "performance_metrics": self.performance_metrics,
            "error_analysis": self.error_analysis,
            "query_results": [asdict(qr) for qr in self.query_results],
            "timestamp": self.timestamp,
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationReport":
        """Load report from JSON file."""
        with open(path) as f:
            data = json.load(f)

        query_results = [QueryResult(**qr) for qr in data["query_results"]]

        return cls(
            experiment_name=data["experiment_name"],
            config=data["config"],
            retrieval_metrics=data["retrieval_metrics"],
            answer_metrics=data["answer_metrics"],
            performance_metrics=data["performance_metrics"],
            error_analysis=data["error_analysis"],
            query_results=query_results,
            timestamp=data["timestamp"],
        )


class RAGEvaluator:
    """Orchestrates RAG evaluation experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.pipeline = RAGPipeline(config)

    def prepare(self, documents: list[Document]) -> list[Chunk]:
        """Prepare the pipeline with documents.

        Args:
            documents: Source documents to chunk and index.

        Returns:
            List of created chunks.
        """
        chunker = create_chunker(self.config.chunking)
        chunks = chunker.chunk_documents(documents)

        self.pipeline.index_chunks(chunks)

        return chunks

    def evaluate(self, dataset: EvalDataset) -> EvaluationReport:
        """Run full evaluation on a dataset.

        Args:
            dataset: EvalDataset with documents and queries.

        Returns:
            EvaluationReport with all metrics.
        """
        chunks = self.prepare(dataset.documents)

        chunk_id_map = self._build_chunk_mapping(chunks, dataset)

        results, perf_metrics = measure_performance(
            self.pipeline,
            [q.query for q in dataset.queries],
            self.config.retrieval.top_k,
        )

        retrieval_metrics_list = []
        answer_metrics_list = []
        error_results = []
        query_results = []

        for eval_query, rag_result in zip(dataset.queries, results):
            retrieved_ids = [r.chunk_id for r in rag_result.retrieval.results]
            relevant_ids = set(
                chunk_id_map.get(rid, rid) for rid in eval_query.relevant_chunk_ids
            )

            ret_metrics = calculate_retrieval_metrics(
                retrieved_ids,
                relevant_ids,
                self.config.metrics.k_values,
            )
            retrieval_metrics_list.append(ret_metrics)

            ans_metrics = calculate_answer_metrics(
                rag_result.generation.answer,
                eval_query.ground_truth_answer,
            )
            answer_metrics_list.append(ans_metrics)

            max_k = max(self.config.metrics.k_values)
            retrieval_recall = ret_metrics.recall_at_k.get(max_k, 0.0)

            error_result = analyze_single_query(
                query=eval_query.query,
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                predicted_answer=rag_result.generation.answer,
                ground_truth=eval_query.ground_truth_answer,
                answer_f1=ans_metrics.f1,
                retrieval_recall=retrieval_recall,
            )
            error_results.append(error_result)

            query_results.append(
                QueryResult(
                    query=eval_query.query,
                    predicted_answer=rag_result.generation.answer,
                    ground_truth_answer=eval_query.ground_truth_answer,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=list(relevant_ids),
                    retrieval_metrics={
                        f"precision@{k}": ret_metrics.precision_at_k[k]
                        for k in self.config.metrics.k_values
                    }
                    | {
                        f"recall@{k}": ret_metrics.recall_at_k[k]
                        for k in self.config.metrics.k_values
                    }
                    | {"mrr": ret_metrics.mrr},
                    answer_metrics={
                        "exact_match": ans_metrics.exact_match,
                        "f1": ans_metrics.f1,
                        "precision": ans_metrics.precision,
                        "recall": ans_metrics.recall,
                    },
                )
            )

        agg_retrieval = aggregate_retrieval_metrics(retrieval_metrics_list)
        agg_answer = aggregate_answer_metrics(answer_metrics_list)
        error_summary = aggregate_error_analysis(error_results)

        from datetime import datetime

        return EvaluationReport(
            experiment_name=self.config.name,
            config=self.config.to_dict(),
            retrieval_metrics=agg_retrieval,
            answer_metrics=agg_answer,
            performance_metrics={
                "avg_retrieval_latency_ms": perf_metrics.avg_retrieval_latency_ms,
                "avg_generation_latency_ms": perf_metrics.avg_generation_latency_ms,
                "avg_total_latency_ms": perf_metrics.avg_total_latency_ms,
                "p50_latency_ms": perf_metrics.p50_latency_ms,
                "p95_latency_ms": perf_metrics.p95_latency_ms,
                "p99_latency_ms": perf_metrics.p99_latency_ms,
                "throughput_qps": perf_metrics.throughput_qps,
                "total_queries": perf_metrics.total_queries,
            },
            error_analysis={
                "total_queries": error_summary.total_queries,
                "error_counts": {k.value: v for k, v in error_summary.error_counts.items()},
                "error_rates": {k.value: v for k, v in error_summary.error_rates.items()},
            },
            query_results=query_results,
            timestamp=datetime.now().isoformat(),
        )

    def _build_chunk_mapping(
        self, chunks: list[Chunk], dataset: EvalDataset
    ) -> dict[str, str]:
        """Build mapping from original chunk IDs to generated chunk IDs.

        This handles cases where ground truth uses different ID schemes.
        """
        return {}


def compare_experiments(reports: list[EvaluationReport]) -> dict:
    """Compare metrics across multiple experiment reports.

    Args:
        reports: List of EvaluationReport objects to compare.

    Returns:
        Dictionary with comparison data.
    """
    comparison = {
        "experiments": [],
        "retrieval_comparison": {},
        "answer_comparison": {},
        "performance_comparison": {},
    }

    for report in reports:
        comparison["experiments"].append(report.experiment_name)

    metric_keys = list(reports[0].retrieval_metrics.keys()) if reports else []
    for key in metric_keys:
        comparison["retrieval_comparison"][key] = [
            r.retrieval_metrics.get(key, 0) for r in reports
        ]

    answer_keys = list(reports[0].answer_metrics.keys()) if reports else []
    for key in answer_keys:
        comparison["answer_comparison"][key] = [
            r.answer_metrics.get(key, 0) for r in reports
        ]

    perf_keys = list(reports[0].performance_metrics.keys()) if reports else []
    for key in perf_keys:
        comparison["performance_comparison"][key] = [
            r.performance_metrics.get(key, 0) for r in reports
        ]

    return comparison
