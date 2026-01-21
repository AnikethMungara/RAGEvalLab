"""Query bucketing for stratified analysis."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class HopType(Enum):
    """Query complexity based on reasoning hops required."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class AnswerLength(Enum):
    """Expected answer length category."""

    SHORT = "short"  # 1-3 words
    MEDIUM = "medium"  # 4-15 words
    LONG = "long"  # 16+ words


class MatchType(Enum):
    """Type of matching required for retrieval."""

    LEXICAL = "lexical"  # Keywords match directly
    SEMANTIC = "semantic"  # Requires understanding meaning


@dataclass
class QueryBucket:
    """Bucket assignment for a query."""

    hop_type: HopType
    answer_length: AnswerLength
    match_type: MatchType
    custom_buckets: dict[str, str] = field(default_factory=dict)

    @property
    def bucket_key(self) -> str:
        """Generate a composite bucket key."""
        return f"{self.hop_type.value}|{self.answer_length.value}|{self.match_type.value}"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "hop_type": self.hop_type.value,
            "answer_length": self.answer_length.value,
            "match_type": self.match_type.value,
            **self.custom_buckets,
        }


# Multi-hop indicator patterns
MULTI_HOP_PATTERNS = [
    r"\b(compare|comparison|difference|between|versus|vs\.?)\b",
    r"\b(both|all|each|every)\b.*\b(and)\b",
    r"\b(how|why).*\b(affect|impact|influence|relate|connect)\b",
    r"\b(before|after|then|first|second|finally)\b.*\b(what|how)\b",
    r"\b(combine|combined|together|along with)\b",
    r"\b(if|when|assuming|given that)\b.*\b(what|how|would)\b",
    r"\b(cause|effect|result|consequence|lead to)\b",
]

# Semantic query patterns (require understanding, not just keyword match)
SEMANTIC_PATTERNS = [
    r"\b(what is the (meaning|purpose|reason|significance)|why)\b",
    r"\b(explain|describe|elaborate|clarify)\b",
    r"\b(how does|how do|how can|how would)\b",
    r"\b(what happens|what would happen)\b",
    r"\b(in what way|to what extent)\b",
    r"\b(implications?|consequences?)\b",
    r"\b(relate|relationship|connection)\b",
    r"\b(similar|different|same)\b",
]

# Lexical query patterns (direct keyword matching likely sufficient)
LEXICAL_PATTERNS = [
    r"\b(what is the (name|date|number|count|amount|price|size))\b",
    r"\b(who is|who was|who are)\b",
    r"\b(when did|when was|when is)\b",
    r"\b(where is|where are|where was)\b",
    r"\b(how (many|much|old|long|far|tall))\b",
    r"\b(list|name|identify)\b.*\b(the|all)\b",
    r"\b(define|definition of)\b",
]


def classify_hop_type(query: str, ground_truth: str = "") -> HopType:
    """Classify query as single-hop or multi-hop.

    Args:
        query: The query text.
        ground_truth: Optional ground truth answer for additional context.

    Returns:
        HopType classification.
    """
    query_lower = query.lower()

    for pattern in MULTI_HOP_PATTERNS:
        if re.search(pattern, query_lower):
            return HopType.MULTI_HOP

    # Check for multiple question marks or conjunctions
    if query.count("?") > 1:
        return HopType.MULTI_HOP

    if re.search(r"\b(and|or)\b.*\b(what|how|why|when|where|who)\b", query_lower):
        return HopType.MULTI_HOP

    # Check ground truth for multi-part answers
    if ground_truth:
        # Multiple sentences or list items suggest multi-hop
        sentences = re.split(r"[.!?]+", ground_truth)
        if len([s for s in sentences if s.strip()]) > 2:
            return HopType.MULTI_HOP

    return HopType.SINGLE_HOP


def classify_answer_length(ground_truth: str) -> AnswerLength:
    """Classify expected answer length.

    Args:
        ground_truth: The ground truth answer.

    Returns:
        AnswerLength classification.
    """
    words = ground_truth.split()
    word_count = len(words)

    if word_count <= 3:
        return AnswerLength.SHORT
    elif word_count <= 15:
        return AnswerLength.MEDIUM
    else:
        return AnswerLength.LONG


def classify_match_type(query: str, ground_truth: str = "") -> MatchType:
    """Classify whether query requires lexical or semantic matching.

    Args:
        query: The query text.
        ground_truth: Optional ground truth answer.

    Returns:
        MatchType classification.
    """
    query_lower = query.lower()

    # Check for lexical patterns first
    for pattern in LEXICAL_PATTERNS:
        if re.search(pattern, query_lower):
            return MatchType.LEXICAL

    # Check for semantic patterns
    for pattern in SEMANTIC_PATTERNS:
        if re.search(pattern, query_lower):
            return MatchType.SEMANTIC

    # Heuristic: short factoid answers are typically lexical
    if ground_truth:
        words = ground_truth.split()
        if len(words) <= 5 and not any(
            word.lower() in ["because", "therefore", "however", "although"]
            for word in words
        ):
            return MatchType.LEXICAL

    # Default to semantic for complex queries
    if len(query.split()) > 10:
        return MatchType.SEMANTIC

    return MatchType.LEXICAL


def classify_query(query: str, ground_truth: str = "") -> QueryBucket:
    """Classify a query into all bucket dimensions.

    Args:
        query: The query text.
        ground_truth: The ground truth answer.

    Returns:
        QueryBucket with all classifications.
    """
    return QueryBucket(
        hop_type=classify_hop_type(query, ground_truth),
        answer_length=classify_answer_length(ground_truth),
        match_type=classify_match_type(query, ground_truth),
    )


@dataclass
class BucketedMetrics:
    """Metrics aggregated by bucket."""

    bucket_name: str
    bucket_value: str
    count: int
    retrieval_metrics: dict[str, float]
    answer_metrics: dict[str, float]


@dataclass
class BucketAnalysis:
    """Complete bucket analysis results."""

    total_queries: int
    buckets: dict[str, list[BucketedMetrics]]  # dimension -> list of bucket metrics

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_queries": self.total_queries,
            "buckets": {
                dim: [
                    {
                        "bucket_name": bm.bucket_name,
                        "bucket_value": bm.bucket_value,
                        "count": bm.count,
                        "retrieval_metrics": bm.retrieval_metrics,
                        "answer_metrics": bm.answer_metrics,
                    }
                    for bm in bucket_list
                ]
                for dim, bucket_list in self.buckets.items()
            },
        }


def aggregate_by_bucket(
    query_results: list[dict],
    bucket_assignments: list[QueryBucket],
) -> BucketAnalysis:
    """Aggregate metrics by bucket dimensions.

    Args:
        query_results: List of query result dicts with retrieval_metrics and answer_metrics.
        bucket_assignments: List of QueryBucket for each query.

    Returns:
        BucketAnalysis with metrics per bucket.
    """
    if len(query_results) != len(bucket_assignments):
        raise ValueError("query_results and bucket_assignments must have same length")

    # Organize results by each bucket dimension
    dimensions = {
        "hop_type": {},
        "answer_length": {},
        "match_type": {},
    }

    for result, bucket in zip(query_results, bucket_assignments):
        # Group by hop type
        hop_key = bucket.hop_type.value
        if hop_key not in dimensions["hop_type"]:
            dimensions["hop_type"][hop_key] = []
        dimensions["hop_type"][hop_key].append(result)

        # Group by answer length
        length_key = bucket.answer_length.value
        if length_key not in dimensions["answer_length"]:
            dimensions["answer_length"][length_key] = []
        dimensions["answer_length"][length_key].append(result)

        # Group by match type
        match_key = bucket.match_type.value
        if match_key not in dimensions["match_type"]:
            dimensions["match_type"][match_key] = []
        dimensions["match_type"][match_key].append(result)

        # Group by custom buckets
        for custom_dim, custom_value in bucket.custom_buckets.items():
            if custom_dim not in dimensions:
                dimensions[custom_dim] = {}
            if custom_value not in dimensions[custom_dim]:
                dimensions[custom_dim][custom_value] = []
            dimensions[custom_dim][custom_value].append(result)

    # Compute aggregated metrics for each bucket
    bucket_analysis = {}

    for dim_name, buckets in dimensions.items():
        bucket_analysis[dim_name] = []

        for bucket_value, results in buckets.items():
            if not results:
                continue

            # Aggregate retrieval metrics
            retrieval_keys = results[0].get("retrieval_metrics", {}).keys()
            agg_retrieval = {}
            for key in retrieval_keys:
                values = [r["retrieval_metrics"].get(key, 0) for r in results]
                agg_retrieval[key] = sum(values) / len(values) if values else 0

            # Aggregate answer metrics
            answer_keys = results[0].get("answer_metrics", {}).keys()
            agg_answer = {}
            for key in answer_keys:
                values = [r["answer_metrics"].get(key, 0) for r in results]
                agg_answer[key] = sum(values) / len(values) if values else 0

            bucket_analysis[dim_name].append(
                BucketedMetrics(
                    bucket_name=dim_name,
                    bucket_value=bucket_value,
                    count=len(results),
                    retrieval_metrics=agg_retrieval,
                    answer_metrics=agg_answer,
                )
            )

    return BucketAnalysis(
        total_queries=len(query_results),
        buckets=bucket_analysis,
    )


def format_bucket_report(analysis: BucketAnalysis) -> str:
    """Format bucket analysis as a readable report.

    Args:
        analysis: BucketAnalysis to format.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 70,
        "STRATIFIED ANALYSIS BY QUERY BUCKET",
        "=" * 70,
        f"Total Queries: {analysis.total_queries}",
        "",
    ]

    for dim_name, buckets in analysis.buckets.items():
        lines.append(f"\n{dim_name.upper().replace('_', ' ')}")
        lines.append("-" * 50)

        for bm in sorted(buckets, key=lambda x: x.bucket_value):
            pct = (bm.count / analysis.total_queries) * 100 if analysis.total_queries > 0 else 0
            lines.append(f"\n  {bm.bucket_value} (n={bm.count}, {pct:.1f}%)")

            # Show key metrics
            if "mrr" in bm.retrieval_metrics:
                lines.append(f"    MRR: {bm.retrieval_metrics['mrr']:.3f}")
            if "recall@5" in bm.retrieval_metrics:
                lines.append(f"    Recall@5: {bm.retrieval_metrics['recall@5']:.3f}")
            if "f1" in bm.answer_metrics:
                lines.append(f"    F1: {bm.answer_metrics['f1']:.3f}")
            if "exact_match" in bm.answer_metrics:
                lines.append(f"    Exact Match: {bm.answer_metrics['exact_match']:.3f}")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)
