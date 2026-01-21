"""Base classes for document chunking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a document chunk."""

    text: str
    doc_id: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: dict | None = None

    @property
    def id(self) -> str:
        """Unique identifier for the chunk."""
        return f"{self.doc_id}_{self.chunk_id}"


@dataclass
class Document:
    """Represents a source document."""

    text: str
    doc_id: str
    metadata: dict | None = None


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        pass

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split multiple documents into chunks."""
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
