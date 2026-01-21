"""Retrieval quality metrics."""

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    total_relevant: int
    retrieved_relevant: dict[int, int]


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate precision@k.

    Precision@k = (# of relevant items in top-k) / k

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of relevant chunk IDs.
        k: Cutoff position.

    Returns:
        Precision@k score.
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for id_ in top_k if id_ in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate recall@k.

    Recall@k = (# of relevant items in top-k) / (total # of relevant items)

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of relevant chunk IDs.
        k: Cutoff position.

    Returns:
        Recall@k score.
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for id_ in top_k if id_ in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / (rank of first relevant item)

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of relevant chunk IDs.

    Returns:
        MRR score (0 if no relevant items found).
    """
    for rank, id_ in enumerate(retrieved_ids, start=1):
        if id_ in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 for relevant, 0 for non-relevant).

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of relevant chunk IDs.
        k: Cutoff position.

    Returns:
        NDCG@k score.
    """
    import math

    if k <= 0 or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, id_ in enumerate(top_k)
        if id_ in relevant_ids
    )

    ideal_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
) -> RetrievalMetrics:
    """Calculate all retrieval metrics.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of relevant chunk IDs.
        k_values: List of k values for precision/recall@k.

    Returns:
        RetrievalMetrics containing all computed metrics.
    """
    precision_scores = {}
    recall_scores = {}
    retrieved_relevant_counts = {}

    for k in k_values:
        precision_scores[k] = precision_at_k(retrieved_ids, relevant_ids, k)
        recall_scores[k] = recall_at_k(retrieved_ids, relevant_ids, k)
        retrieved_relevant_counts[k] = sum(
            1 for id_ in retrieved_ids[:k] if id_ in relevant_ids
        )

    mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

    return RetrievalMetrics(
        precision_at_k=precision_scores,
        recall_at_k=recall_scores,
        mrr=mrr,
        total_relevant=len(relevant_ids),
        retrieved_relevant=retrieved_relevant_counts,
    )


def aggregate_retrieval_metrics(
    metrics_list: list[RetrievalMetrics],
) -> dict[str, float]:
    """Aggregate retrieval metrics across multiple queries.

    Args:
        metrics_list: List of RetrievalMetrics from individual queries.

    Returns:
        Dictionary of aggregated metric scores.
    """
    if not metrics_list:
        return {}

    k_values = list(metrics_list[0].precision_at_k.keys())
    n = len(metrics_list)

    aggregated = {}

    for k in k_values:
        aggregated[f"precision@{k}"] = sum(m.precision_at_k[k] for m in metrics_list) / n
        aggregated[f"recall@{k}"] = sum(m.recall_at_k[k] for m in metrics_list) / n

    aggregated["mrr"] = sum(m.mrr for m in metrics_list) / n

    return aggregated
