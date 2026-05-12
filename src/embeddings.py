"""
OpenAI embedding helper used by indexing and retrieval.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm

from src.config import settings


class Embedder:
    """Thin wrapper around the OpenAI embeddings API."""

    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
        batch_size: int = 96,
        max_retries: int = 3,
    ):
        if client is None and not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env before creating embeddings."
            )

        self.model = model or settings.embedding_model
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed(self, texts: Sequence[str], progress: bool = True) -> np.ndarray:
        """Return one float32 embedding vector per input text."""
        inputs = [(text or " ").strip() or " " for text in texts]
        if not inputs:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        starts = range(0, len(inputs), self.batch_size)
        if progress:
            starts = tqdm(starts, desc="Embedding chunks", unit="batch")

        for start in starts:
            batch = inputs[start:start + self.batch_size]
            response = self._create_embeddings(batch)
            vectors.extend(
                item.embedding
                for item in sorted(response.data, key=lambda item: item.index)
            )

        return np.asarray(vectors, dtype=np.float32)

    def _create_embeddings(self, batch: list[str]):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.embeddings.create(model=self.model, input=batch)
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError("Embedding request failed without raising an OpenAI error.")
