"""Error analysis for RAG systems."""

from dataclasses import dataclass, field
from enum import Enum


class ErrorType(Enum):
    """Types of errors in RAG systems."""

    RETRIEVAL_MISS = "retrieval_miss"
    INCORRECT_CHUNK = "incorrect_chunk"
    HALLUCINATION = "hallucination"
    INCOMPLETE_ANSWER = "incomplete_answer"
    CORRECT = "correct"


@dataclass
class ErrorAnalysisResult:
    """Result of error analysis for a single query."""

    query: str
    error_type: ErrorType
    predicted_answer: str
    ground_truth: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    retrieval_recall: float
    answer_f1: float
    details: dict = field(default_factory=dict)


@dataclass
class ErrorAnalysisSummary:
    """Summary of error analysis across multiple queries."""

    total_queries: int
    error_counts: dict[ErrorType, int]
    error_rates: dict[ErrorType, float]
    examples_by_type: dict[ErrorType, list[ErrorAnalysisResult]]


def classify_error(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    predicted_answer: str,
    ground_truth: str,
    answer_f1: float,
    retrieval_recall: float,
    f1_threshold: float = 0.5,
    recall_threshold: float = 0.5,
) -> ErrorType:
    """Classify the type of error for a RAG query.

    Args:
        retrieved_ids: IDs of retrieved chunks.
        relevant_ids: IDs of relevant chunks.
        predicted_answer: Generated answer.
        ground_truth: Ground truth answer.
        answer_f1: F1 score of the answer.
        retrieval_recall: Recall of retrieval.
        f1_threshold: Threshold for acceptable F1 score.
        recall_threshold: Threshold for acceptable retrieval recall.

    Returns:
        ErrorType classification.
    """
    if answer_f1 >= f1_threshold:
        return ErrorType.CORRECT

    if retrieval_recall < recall_threshold:
        return ErrorType.RETRIEVAL_MISS

    retrieved_relevant = [id_ for id_ in retrieved_ids if id_ in relevant_ids]
    if not retrieved_relevant:
        return ErrorType.RETRIEVAL_MISS

    if retrieval_recall >= recall_threshold and answer_f1 < f1_threshold:
        normalized_pred = predicted_answer.lower().strip()
        normalized_truth = ground_truth.lower().strip()

        if len(normalized_pred) > len(normalized_truth) * 1.5:
            return ErrorType.HALLUCINATION

        if len(normalized_pred) < len(normalized_truth) * 0.5:
            return ErrorType.INCOMPLETE_ANSWER

        return ErrorType.INCORRECT_CHUNK

    return ErrorType.INCORRECT_CHUNK


def analyze_single_query(
    query: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    predicted_answer: str,
    ground_truth: str,
    answer_f1: float,
    retrieval_recall: float,
    f1_threshold: float = 0.5,
    recall_threshold: float = 0.5,
) -> ErrorAnalysisResult:
    """Perform error analysis for a single query.

    Args:
        query: The input query.
        retrieved_ids: IDs of retrieved chunks.
        relevant_ids: IDs of relevant chunks.
        predicted_answer: Generated answer.
        ground_truth: Ground truth answer.
        answer_f1: F1 score of the answer.
        retrieval_recall: Recall of retrieval.
        f1_threshold: Threshold for acceptable F1 score.
        recall_threshold: Threshold for acceptable retrieval recall.

    Returns:
        ErrorAnalysisResult with classification and details.
    """
    error_type = classify_error(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
        answer_f1=answer_f1,
        retrieval_recall=retrieval_recall,
        f1_threshold=f1_threshold,
        recall_threshold=recall_threshold,
    )

    details = {
        "retrieved_relevant_count": len([id_ for id_ in retrieved_ids if id_ in relevant_ids]),
        "total_relevant": len(relevant_ids),
        "total_retrieved": len(retrieved_ids),
    }

    return ErrorAnalysisResult(
        query=query,
        error_type=error_type,
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
        retrieved_chunk_ids=retrieved_ids,
        relevant_chunk_ids=list(relevant_ids),
        retrieval_recall=retrieval_recall,
        answer_f1=answer_f1,
        details=details,
    )


def aggregate_error_analysis(
    results: list[ErrorAnalysisResult],
    max_examples_per_type: int = 5,
) -> ErrorAnalysisSummary:
    """Aggregate error analysis results across queries.

    Args:
        results: List of ErrorAnalysisResult objects.
        max_examples_per_type: Maximum examples to keep per error type.

    Returns:
        ErrorAnalysisSummary with counts and examples.
    """
    error_counts: dict[ErrorType, int] = {error_type: 0 for error_type in ErrorType}
    examples_by_type: dict[ErrorType, list[ErrorAnalysisResult]] = {
        error_type: [] for error_type in ErrorType
    }

    for result in results:
        error_counts[result.error_type] += 1
        if len(examples_by_type[result.error_type]) < max_examples_per_type:
            examples_by_type[result.error_type].append(result)

    total = len(results)
    error_rates = {
        error_type: count / total if total > 0 else 0.0
        for error_type, count in error_counts.items()
    }

    return ErrorAnalysisSummary(
        total_queries=total,
        error_counts=error_counts,
        error_rates=error_rates,
        examples_by_type=examples_by_type,
    )


def format_error_report(summary: ErrorAnalysisSummary) -> str:
    """Format error analysis summary as a readable report.

    Args:
        summary: ErrorAnalysisSummary to format.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 60,
        "ERROR ANALYSIS REPORT",
        "=" * 60,
        f"Total Queries Analyzed: {summary.total_queries}",
        "",
        "Error Distribution:",
        "-" * 40,
    ]

    for error_type in ErrorType:
        count = summary.error_counts[error_type]
        rate = summary.error_rates[error_type] * 100
        lines.append(f"  {error_type.value:20s}: {count:5d} ({rate:5.1f}%)")

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
