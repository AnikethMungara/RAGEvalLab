"""Base classes for answer generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Represents a generated answer."""

    answer: str
    prompt: str
    context_used: list[str]
    model: str
    tokens_used: int | None = None
    latency_ms: float | None = None


class BaseGenerator(ABC):
    """Abstract base class for answer generators."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context_chunks: list[str],
    ) -> GenerationResult:
        """Generate an answer based on query and context.

        Args:
            query: The user's question.
            context_chunks: List of relevant text chunks.

        Returns:
            GenerationResult containing the answer and metadata.
        """
        pass

    def _build_prompt(self, query: str, context_chunks: list[str]) -> str:
        """Build the prompt for generation."""
        context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))

        prompt = f"""Answer the question based on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        return prompt
