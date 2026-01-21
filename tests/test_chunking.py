"""Tests for chunking strategies."""

import pytest

from rageval.chunking import (
    Document,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    create_chunker,
)
from rageval.config import ChunkingConfig


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        doc = Document(text="a" * 100, doc_id="doc1")

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        assert all(len(c.text) <= 50 for c in chunks)

    def test_small_document(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        doc = Document(text="Hello world", doc_id="doc1")

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_chunk_ids(self):
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        doc = Document(text="a" * 50, doc_id="test_doc")

        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.doc_id == "test_doc"
            assert chunk.chunk_id == i
            assert chunk.id == f"test_doc_{i}"


class TestRecursiveChunker:
    def test_splits_on_paragraphs(self):
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        doc = Document(
            text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            doc_id="doc1",
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1

    def test_respects_chunk_size(self):
        chunker = RecursiveChunker(chunk_size=50, overlap=5)
        doc = Document(text="word " * 100, doc_id="doc1")

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert len(chunk.text) <= 60  # Allow some flexibility for word boundaries


class TestSentenceChunker:
    def test_splits_sentences(self):
        chunker = SentenceChunker(chunk_size=100, overlap_sentences=1)
        doc = Document(
            text="First sentence. Second sentence. Third sentence. Fourth sentence.",
            doc_id="doc1",
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1

    def test_keeps_sentences_intact(self):
        chunker = SentenceChunker(chunk_size=50, overlap_sentences=0)
        doc = Document(text="Short. Another short.", doc_id="doc1")

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert "." in chunk.text or chunk.text in ["Short", "Another short"]


class TestCreateChunker:
    def test_create_fixed_chunker(self):
        config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=10)
        chunker = create_chunker(config)

        assert isinstance(chunker, FixedSizeChunker)

    def test_create_recursive_chunker(self):
        config = ChunkingConfig(strategy="recursive", chunk_size=100, overlap=10)
        chunker = create_chunker(config)

        assert isinstance(chunker, RecursiveChunker)

    def test_create_sentence_chunker(self):
        config = ChunkingConfig(strategy="sentence", chunk_size=100, overlap=1)
        chunker = create_chunker(config)

        assert isinstance(chunker, SentenceChunker)

    def test_unknown_strategy_raises(self):
        config = ChunkingConfig(strategy="unknown")

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker(config)
