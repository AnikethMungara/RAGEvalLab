"""Tests for query bucketing."""

import pytest

from rageval.analysis import (
    AnswerLength,
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


class TestHopTypeClassification:
    def test_single_hop_simple_question(self):
        query = "What is the capital of France?"
        assert classify_hop_type(query) == HopType.SINGLE_HOP

    def test_multi_hop_comparison(self):
        query = "What is the difference between Python and JavaScript?"
        assert classify_hop_type(query) == HopType.MULTI_HOP

    def test_multi_hop_causal(self):
        query = "How does climate change affect biodiversity?"
        assert classify_hop_type(query) == HopType.MULTI_HOP

    def test_multi_hop_multiple_questions(self):
        query = "What is Python? What is JavaScript?"
        assert classify_hop_type(query) == HopType.MULTI_HOP

    def test_multi_hop_conditional(self):
        query = "If we use caching, what would happen to latency?"
        assert classify_hop_type(query) == HopType.MULTI_HOP

    def test_single_hop_who_question(self):
        query = "Who invented the telephone?"
        assert classify_hop_type(query) == HopType.SINGLE_HOP


class TestAnswerLengthClassification:
    def test_short_answer(self):
        assert classify_answer_length("Paris") == AnswerLength.SHORT
        assert classify_answer_length("42") == AnswerLength.SHORT
        assert classify_answer_length("Yes") == AnswerLength.SHORT

    def test_medium_answer(self):
        answer = "Python is a high-level programming language."
        assert classify_answer_length(answer) == AnswerLength.MEDIUM

    def test_long_answer(self):
        answer = " ".join(["word"] * 20)
        assert classify_answer_length(answer) == AnswerLength.LONG


class TestMatchTypeClassification:
    def test_lexical_factoid(self):
        query = "What is the name of the CEO?"
        assert classify_match_type(query) == MatchType.LEXICAL

    def test_lexical_date(self):
        query = "When was the company founded?"
        assert classify_match_type(query) == MatchType.LEXICAL

    def test_lexical_count(self):
        query = "How many employees work here?"
        assert classify_match_type(query) == MatchType.LEXICAL

    def test_semantic_explanation(self):
        query = "Explain how the authentication system works"
        assert classify_match_type(query) == MatchType.SEMANTIC

    def test_semantic_why(self):
        query = "What is the purpose of this feature?"
        assert classify_match_type(query) == MatchType.SEMANTIC

    def test_semantic_relationship(self):
        query = "How does the cache relate to the database?"
        assert classify_match_type(query) == MatchType.SEMANTIC


class TestClassifyQuery:
    def test_returns_query_bucket(self):
        bucket = classify_query(
            "What is the name of the author?",
            "John Smith"
        )

        assert isinstance(bucket, QueryBucket)
        assert bucket.hop_type == HopType.SINGLE_HOP
        assert bucket.answer_length == AnswerLength.SHORT
        assert bucket.match_type == MatchType.LEXICAL

    def test_complex_query(self):
        bucket = classify_query(
            "Compare the performance implications of using Redis vs Memcached for caching",
            "Redis offers persistence and data structures while Memcached is simpler and faster for basic key-value caching. Redis supports replication and clustering natively."
        )

        assert bucket.hop_type == HopType.MULTI_HOP
        assert bucket.answer_length == AnswerLength.LONG

    def test_bucket_key(self):
        bucket = QueryBucket(
            hop_type=HopType.SINGLE_HOP,
            answer_length=AnswerLength.SHORT,
            match_type=MatchType.LEXICAL,
        )

        assert bucket.bucket_key == "single_hop|short|lexical"

    def test_to_dict(self):
        bucket = QueryBucket(
            hop_type=HopType.MULTI_HOP,
            answer_length=AnswerLength.MEDIUM,
            match_type=MatchType.SEMANTIC,
        )

        d = bucket.to_dict()
        assert d["hop_type"] == "multi_hop"
        assert d["answer_length"] == "medium"
        assert d["match_type"] == "semantic"


class TestAggregateByBucket:
    def test_basic_aggregation(self):
        query_results = [
            {
                "retrieval_metrics": {"mrr": 1.0, "recall@5": 1.0},
                "answer_metrics": {"f1": 0.9, "exact_match": 1.0},
            },
            {
                "retrieval_metrics": {"mrr": 0.5, "recall@5": 0.8},
                "answer_metrics": {"f1": 0.7, "exact_match": 0.0},
            },
        ]

        buckets = [
            QueryBucket(HopType.SINGLE_HOP, AnswerLength.SHORT, MatchType.LEXICAL),
            QueryBucket(HopType.SINGLE_HOP, AnswerLength.MEDIUM, MatchType.SEMANTIC),
        ]

        analysis = aggregate_by_bucket(query_results, buckets)

        assert analysis.total_queries == 2
        assert "hop_type" in analysis.buckets
        assert "answer_length" in analysis.buckets
        assert "match_type" in analysis.buckets

        # Both queries are single_hop
        hop_buckets = {b.bucket_value: b for b in analysis.buckets["hop_type"]}
        assert "single_hop" in hop_buckets
        assert hop_buckets["single_hop"].count == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            aggregate_by_bucket(
                [{"retrieval_metrics": {}, "answer_metrics": {}}],
                []
            )


class TestFormatBucketReport:
    def test_format_output(self):
        query_results = [
            {
                "retrieval_metrics": {"mrr": 1.0, "recall@5": 1.0},
                "answer_metrics": {"f1": 0.9, "exact_match": 1.0},
            },
        ]
        buckets = [
            QueryBucket(HopType.SINGLE_HOP, AnswerLength.SHORT, MatchType.LEXICAL),
        ]

        analysis = aggregate_by_bucket(query_results, buckets)
        report = format_bucket_report(analysis)

        assert "STRATIFIED ANALYSIS" in report
        assert "HOP TYPE" in report
        assert "ANSWER LENGTH" in report
        assert "MATCH TYPE" in report
        assert "single_hop" in report
