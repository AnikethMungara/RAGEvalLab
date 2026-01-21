"""Tests for evaluation metrics."""

import pytest

from rageval.metrics import (
    calculate_answer_metrics,
    calculate_retrieval_metrics,
    exact_match,
    mean_reciprocal_rank,
    normalize_answer,
    precision_at_k,
    recall_at_k,
    token_f1,
)


class TestRetrievalMetrics:
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}

        assert precision_at_k(retrieved, relevant, 1) == 1.0
        assert precision_at_k(retrieved, relevant, 2) == 0.5
        assert precision_at_k(retrieved, relevant, 3) == 2 / 3
        assert precision_at_k(retrieved, relevant, 5) == 3 / 5

    def test_precision_at_k_no_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}

        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}

        assert recall_at_k(retrieved, relevant, 1) == 1 / 3
        assert recall_at_k(retrieved, relevant, 3) == 2 / 3
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_recall_at_k_empty_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = set()

        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_mrr_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}

        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_second_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"b"}

        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_mrr_no_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x"}

        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_calculate_retrieval_metrics(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}

        metrics = calculate_retrieval_metrics(retrieved, relevant, k_values=[1, 3, 5])

        assert metrics.precision_at_k[1] == 1.0
        assert metrics.recall_at_k[1] == 0.5
        assert metrics.mrr == 1.0
        assert metrics.total_relevant == 2


class TestAnswerMetrics:
    def test_normalize_answer(self):
        assert normalize_answer("The Answer!") == "answer"
        assert normalize_answer("  hello   world  ") == "hello world"
        assert normalize_answer("A cat, a dog.") == "cat dog"

    def test_exact_match_identical(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_exact_match_normalized(self):
        assert exact_match("The Answer", "the answer") == 1.0
        assert exact_match("Hello!", "hello") == 1.0

    def test_exact_match_different(self):
        assert exact_match("hello", "world") == 0.0

    def test_token_f1_identical(self):
        f1, prec, rec = token_f1("hello world", "hello world")
        assert f1 == 1.0
        assert prec == 1.0
        assert rec == 1.0

    def test_token_f1_partial(self):
        f1, prec, rec = token_f1("hello world", "hello there")

        assert 0 < f1 < 1
        assert prec == 0.5
        assert rec == 0.5

    def test_token_f1_no_overlap(self):
        f1, prec, rec = token_f1("hello world", "foo bar")
        assert f1 == 0.0
        assert prec == 0.0
        assert rec == 0.0

    def test_token_f1_empty(self):
        f1, prec, rec = token_f1("", "")
        assert f1 == 1.0

    def test_calculate_answer_metrics(self):
        metrics = calculate_answer_metrics("the quick brown fox", "quick brown fox")

        assert metrics.exact_match == 1.0
        assert metrics.f1 == 1.0
