"""Generation model implementations."""

import os
import time

from rageval.config import GenerationConfig
from rageval.generation.base import BaseGenerator, GenerationResult


class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 256,
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def generate(
        self,
        query: str,
        context_chunks: list[str],
    ) -> GenerationResult:
        """Generate an answer using OpenAI API."""
        prompt = self._build_prompt(query, context_chunks)

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        answer = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else None

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            context_used=context_chunks,
            model=self.model,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )


class MockGenerator(BaseGenerator):
    """Mock generator for testing without API calls."""

    def __init__(self, default_answer: str = "This is a mock answer."):
        self.default_answer = default_answer
        self.model = "mock"

    def generate(
        self,
        query: str,
        context_chunks: list[str],
    ) -> GenerationResult:
        """Return a mock answer."""
        prompt = self._build_prompt(query, context_chunks)

        if context_chunks:
            answer = f"Based on the context: {context_chunks[0][:100]}..."
        else:
            answer = self.default_answer

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            context_used=context_chunks,
            model=self.model,
            tokens_used=None,
            latency_ms=1.0,
        )


class ContextOnlyGenerator(BaseGenerator):
    """Generator that returns concatenated context (for retrieval-only evaluation)."""

    def __init__(self):
        self.model = "context_only"

    def generate(
        self,
        query: str,
        context_chunks: list[str],
    ) -> GenerationResult:
        """Return concatenated context as the answer."""
        answer = " ".join(context_chunks)

        return GenerationResult(
            answer=answer,
            prompt=query,
            context_used=context_chunks,
            model=self.model,
            tokens_used=None,
            latency_ms=0.0,
        )


def create_generator(config: GenerationConfig) -> BaseGenerator:
    """Factory function to create a generator from config."""
    model = config.model.lower()

    if model == "mock":
        return MockGenerator()
    elif model == "context_only":
        return ContextOnlyGenerator()
    elif "openai" in model or "gpt" in model:
        model_name = config.model.replace("openai/", "")
        api_key = os.environ.get(config.api_key_env)
        return OpenAIGenerator(
            model=model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown generation model: {config.model}")
