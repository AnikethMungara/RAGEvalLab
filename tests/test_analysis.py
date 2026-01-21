"""Tests for error analysis."""

import pytest

from rageval.analysis import (
    ErrorType,
    aggregate_error_analysis,
    analyze_single_query,
    classify_error,
    format_error_report,
)


class TestClassifyError:
    def test_correct_classification(self):
        error_type = classify_error(
            retrieved_ids=["a", "b"],
            relevant_ids={"a", "b"},
            predicted_answer="correct answer",
            ground_truth="correct answer",
            answer_f1=0.9,
            retrieval_recall=1.0,
        )

        assert error_type == ErrorType.CORRECT

    def test_retrieval_miss(self):
        error_type = classify_error(
            retrieved_ids=["x", "y"],
            relevant_ids={"a", "b"},
            predicted_answer="wrong answer",
            ground_truth="correct answer",
            answer_f1=0.1,
            retrieval_recall=0.0,
        )

        assert error_type == ErrorType.RETRIEVAL_MISS

    def test_hallucination(self):
        error_type = classify_error(
            retrieved_ids=["a", "b"],
            relevant_ids={"a", "b"},
            predicted_answer="a very very very long hallucinated answer that goes on and on",
            ground_truth="short",
            answer_f1=0.1,
            retrieval_recall=1.0,
        )

        assert error_type == ErrorType.HALLUCINATION

    def test_incomplete_answer(self):
        error_type = classify_error(
            retrieved_ids=["a", "b"],
            relevant_ids={"a", "b"},
            predicted_answer="x",
            ground_truth="a much longer expected answer here",
            answer_f1=0.1,
            retrieval_recall=1.0,
        )

        assert error_type == ErrorType.INCOMPLETE_ANSWER


class TestAnalyzeSingleQuery:
    def test_returns_result(self):
        result = analyze_single_query(
            query="test query",
            retrieved_ids=["a", "b", "c"],
            relevant_ids={"a", "b"},
            predicted_answer="test answer",
            ground_truth="correct answer",
            answer_f1=0.5,
            retrieval_recall=0.8,
        )

        assert result.query == "test query"
        assert result.error_type in ErrorType
        assert result.retrieval_recall == 0.8
        assert result.answer_f1 == 0.5


class TestAggregateErrorAnalysis:
    def test_aggregation(self):
        results = [
            analyze_single_query(
                query=f"query {i}",
                retrieved_ids=["a"],
                relevant_ids={"a"},
                predicted_answer="answer",
                ground_truth="answer",
                answer_f1=0.9,
                retrieval_recall=1.0,
            )
            for i in range(5)
        ]

        summary = aggregate_error_analysis(results)

        assert summary.total_queries == 5
        assert summary.error_counts[ErrorType.CORRECT] == 5
        assert summary.error_rates[ErrorType.CORRECT] == 1.0


class TestFormatErrorReport:
    def test_format_output(self):
        results = [
            analyze_single_query(
                query="test",
                retrieved_ids=["a"],
                relevant_ids={"a"},
                predicted_answer="answer",
                ground_truth="answer",
                answer_f1=0.9,
                retrieval_recall=1.0,
            )
        ]

        summary = aggregate_error_analysis(results)
        report = format_error_report(summary)

        assert "ERROR ANALYSIS REPORT" in report
        assert "Total Queries Analyzed: 1" in report
