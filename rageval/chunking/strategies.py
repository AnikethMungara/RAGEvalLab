"""Document chunking strategies."""

from rageval.chunking.base import BaseChunker, Chunk, Document
from rageval.config import ChunkingConfig


class FixedSizeChunker(BaseChunker):
    """Chunks documents into fixed-size pieces with optional overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks."""
        text = document.text
        chunks = []
        chunk_id = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        doc_id=document.doc_id,
                        chunk_id=chunk_id,
                        start_char=start,
                        end_char=end,
                        metadata=document.metadata,
                    )
                )
                chunk_id += 1

            start = end - self.overlap if end < len(text) else end

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively splits documents using a hierarchy of separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document recursively using separators."""
        splits = self._recursive_split(document.text, self.separators)
        return self._merge_splits(splits, document)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
        if not text:
            return []

        if not separators:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator not in text:
            return self._recursive_split(text, remaining_separators)

        splits = text.split(separator)
        result = []

        for split in splits:
            if len(split) <= self.chunk_size:
                if split.strip():
                    result.append(split)
            else:
                result.extend(self._recursive_split(split, remaining_separators))

        return result

    def _merge_splits(self, splits: list[str], document: Document) -> list[Chunk]:
        """Merge small splits into chunks respecting chunk_size."""
        chunks = []
        current_text = ""
        current_start = 0
        chunk_id = 0

        for split in splits:
            if len(current_text) + len(split) + 1 <= self.chunk_size:
                if current_text:
                    current_text += " " + split
                else:
                    current_text = split
            else:
                if current_text.strip():
                    end_char = current_start + len(current_text)
                    chunks.append(
                        Chunk(
                            text=current_text,
                            doc_id=document.doc_id,
                            chunk_id=chunk_id,
                            start_char=current_start,
                            end_char=end_char,
                            metadata=document.metadata,
                        )
                    )
                    chunk_id += 1
                    current_start = max(0, end_char - self.overlap)

                current_text = split

        if current_text.strip():
            chunks.append(
                Chunk(
                    text=current_text,
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(current_text),
                    metadata=document.metadata,
                )
            )

        return chunks


class SentenceChunker(BaseChunker):
    """Chunks documents by sentences, grouping them to meet size requirements."""

    def __init__(self, chunk_size: int = 512, overlap_sentences: int = 1):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into sentence-based chunks."""
        sentences = self._split_sentences(document.text)
        return self._group_sentences(sentences, document)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences(self, sentences: list[str], document: Document) -> list[Chunk]:
        """Group sentences into chunks."""
        chunks = []
        current_sentences: list[str] = []
        current_length = 0
        chunk_id = 0
        char_position = 0

        for sentence in sentences:
            if current_length + len(sentence) + 1 <= self.chunk_size:
                current_sentences.append(sentence)
                current_length += len(sentence) + 1
            else:
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            doc_id=document.doc_id,
                            chunk_id=chunk_id,
                            start_char=char_position,
                            end_char=char_position + len(chunk_text),
                            metadata=document.metadata,
                        )
                    )
                    chunk_id += 1
                    char_position += len(chunk_text) - sum(
                        len(s) for s in current_sentences[-self.overlap_sentences :]
                    )

                    current_sentences = current_sentences[-self.overlap_sentences :]
                    current_length = sum(len(s) + 1 for s in current_sentences)

                current_sentences.append(sentence)
                current_length += len(sentence) + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                    metadata=document.metadata,
                )
            )

        return chunks


def create_chunker(config: ChunkingConfig) -> BaseChunker:
    """Factory function to create a chunker from config."""
    strategy = config.strategy.lower()

    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=config.chunk_size, overlap=config.overlap)
    elif strategy == "recursive":
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            separators=config.separators,
        )
    elif strategy == "sentence":
        return SentenceChunker(chunk_size=config.chunk_size, overlap_sentences=config.overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
