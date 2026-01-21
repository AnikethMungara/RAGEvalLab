"""Error analysis tools for RAGEvalLab."""

from rageval.analysis.bucketing import (
    AnswerLength,
    BucketAnalysis,
    BucketedMetrics,
    HopType,
    MatchType,
    QueryBucket,
    aggregate_by_bucket,
    classify_answer_length,
    classify_hop_type,
    classify_match_type,
    classify_query,
    format_bucket_report,
)
from rageval.analysis.errors import (
    ErrorAnalysisResult,
    ErrorAnalysisSummary,
    ErrorType,
    aggregate_error_analysis,
    analyze_single_query,
    classify_error,
    format_error_report,
)

__all__ = [
    # Error analysis
    "ErrorType",
    "ErrorAnalysisResult",
    "ErrorAnalysisSummary",
    "classify_error",
    "analyze_single_query",
    "aggregate_error_analysis",
    "format_error_report",
    # Query bucketing
    "HopType",
    "AnswerLength",
    "MatchType",
    "QueryBucket",
    "BucketedMetrics",
    "BucketAnalysis",
    "classify_hop_type",
    "classify_answer_length",
    "classify_match_type",
    "classify_query",
    "aggregate_by_bucket",
    "format_bucket_report",
]
