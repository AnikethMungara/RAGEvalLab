"""Answer quality metrics."""

import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass
class AnswerMetrics:
    """Container for answer evaluation metrics."""

    exact_match: float
    f1: float
    precision: float
    recall: float


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Applies:
    - Lowercase conversion
    - Punctuation removal
    - Article removal (a, an, the)
    - Whitespace normalization

    Args:
        text: Raw answer text.

    Returns:
        Normalized text.
    """
    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    text = " ".join(text.split())

    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        1.0 if normalized answers match, 0.0 otherwise.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Calculate token-level F1, precision, and recall.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        Tuple of (f1, precision, recall).
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not truth_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    common = sum((pred_counter & truth_counter).values())

    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(truth_tokens) if truth_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def calculate_answer_metrics(
    prediction: str,
    ground_truth: str,
) -> AnswerMetrics:
    """Calculate all answer metrics.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        AnswerMetrics containing all computed metrics.
    """
    em = exact_match(prediction, ground_truth)
    f1, precision, recall = token_f1(prediction, ground_truth)

    return AnswerMetrics(
        exact_match=em,
        f1=f1,
        precision=precision,
        recall=recall,
    )


def calculate_answer_metrics_multi(
    prediction: str,
    ground_truths: list[str],
) -> AnswerMetrics:
    """Calculate answer metrics against multiple ground truths.

    Takes the maximum score across all ground truths.

    Args:
        prediction: Predicted answer.
        ground_truths: List of acceptable ground truth answers.

    Returns:
        AnswerMetrics with best scores across ground truths.
    """
    if not ground_truths:
        return AnswerMetrics(exact_match=0.0, f1=0.0, precision=0.0, recall=0.0)

    best_em = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for gt in ground_truths:
        metrics = calculate_answer_metrics(prediction, gt)
        best_em = max(best_em, metrics.exact_match)
        best_f1 = max(best_f1, metrics.f1)
        best_precision = max(best_precision, metrics.precision)
        best_recall = max(best_recall, metrics.recall)

    return AnswerMetrics(
        exact_match=best_em,
        f1=best_f1,
        precision=best_precision,
        recall=best_recall,
    )


def aggregate_answer_metrics(metrics_list: list[AnswerMetrics]) -> dict[str, float]:
    """Aggregate answer metrics across multiple examples.

    Args:
        metrics_list: List of AnswerMetrics from individual examples.

    Returns:
        Dictionary of aggregated metric scores.
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)

    return {
        "exact_match": sum(m.exact_match for m in metrics_list) / n,
        "f1": sum(m.f1 for m in metrics_list) / n,
        "precision": sum(m.precision for m in metrics_list) / n,
        "recall": sum(m.recall for m in metrics_list) / n,
    }
