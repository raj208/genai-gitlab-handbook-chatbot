"""
Semantic retrieval over the FAISS index built in Step 3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.config import settings
from src.embeddings import Embedder


@dataclass
class RetrievedChunk:
    score: float          # cosine similarity in [-1, 1]
    chunk_id: int
    url: str
    title: str
    source: str           # "handbook" or "direction"
    text: str


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return np.ascontiguousarray(v / n, dtype=np.float32)


class Retriever:
    """
    Loads the FAISS index and chunk metadata once, then serves queries.
    Designed to be instantiated once and reused (e.g. cached in Streamlit).
    """

    def __init__(
        self,
        index_dir: Path | None = None,
        embedder: Embedder | None = None,
    ):
        index_dir = index_dir or settings.index_dir
        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.jsonl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index files missing in {index_dir}. Run `python -m src.build_index` first."
            )

        self.index = faiss.read_index(str(index_path))
        self.chunks: list[dict] = [
            json.loads(line) for line in chunks_path.open(encoding="utf-8") if line.strip()
        ]
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index/chunks mismatch: {self.index.ntotal} vs {len(self.chunks)}. Rebuild the index."
            )

        self.embedder = embedder or Embedder()

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        k = top_k or settings.top_k
        qvec = self.embedder.embed([query], progress=False)
        qvec = _normalize(qvec)

        scores, ids = self.index.search(qvec, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:  # FAISS pads with -1 if fewer than k results
                continue
            ch = self.chunks[idx]
            results.append(RetrievedChunk(
                score=float(score),
                chunk_id=ch["chunk_id"],
                url=ch["url"],
                title=ch["title"],
                source=ch["source"],
                text=ch["text"],
            ))
        return results
