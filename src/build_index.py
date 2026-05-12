"""
Build a FAISS index over chunked GitLab pages.

Usage:
    python -m src.build_index                 # uses data/raw_pages.jsonl
    python -m src.build_index --rebuild       # overwrite existing index
    python -m src.build_index --limit 50      # only first N pages (testing)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import faiss
import numpy as np

from src.chunking import chunk_text
from src.config import settings
from src.embeddings import Embedder


@dataclass
class ChunkRecord:
    chunk_id: int
    url: str
    title: str
    source: str
    text: str
    token_count: int


def _read_pages(path: Path, limit: int | None) -> list[dict]:
    pages = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(pages) >= limit:
                break
    return pages


def _build_chunks(pages: list[dict]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    cid = 0
    for p in pages:
        chunks = chunk_text(
            p.get("content", ""),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for ch in chunks:
            records.append(ChunkRecord(
                chunk_id=cid,
                url=p.get("url", ""),
                title=p.get("title", ""),
                source=p.get("source", ""),
                text=ch.text,
                token_count=ch.token_count,
            ))
            cid += 1
    return records


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build(rebuild: bool, limit: int | None) -> int:
    raw_path = settings.data_dir / "raw_pages.jsonl"
    if not raw_path.exists():
        print(f"  [FAIL]  Missing {raw_path}. Run Step 2 first.")
        return 1

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = settings.index_dir / "faiss.index"
    chunks_path = settings.index_dir / "chunks.jsonl"
    meta_path = settings.index_dir / "meta.json"

    if index_path.exists() and not rebuild:
        print(f"  Index already exists at {index_path}.")
        print("  Use --rebuild to overwrite.")
        return 0

    print("Reading pages...")
    pages = _read_pages(raw_path, limit)
    print(f"  {len(pages)} pages loaded")

    print(f"Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    records = _build_chunks(pages)
    if not records:
        print("  [FAIL]  No chunks produced.")
        return 1
    avg_tok = sum(r.token_count for r in records) // len(records)
    print(f"  {len(records)} chunks (avg {avg_tok} tokens)")

    print(f"Embedding with {settings.embedding_model}...")
    embedder = Embedder()
    vectors = embedder.embed([r.text for r in records])
    if vectors.size == 0:
        print("  [FAIL]  No embeddings returned.")
        return 1
    print(f"  vectors shape: {vectors.shape}, dtype: {vectors.dtype}")

    vectors = _normalize(vectors)

    print("Building FAISS index...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    print(f"  wrote {index_path}  ({index.ntotal} vectors, dim={dim})")

    print("Writing chunk metadata...")
    with chunks_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"  wrote {chunks_path}")

    meta = {
        "num_pages": len(pages),
        "num_chunks": len(records),
        "embedding_model": settings.embedding_model,
        "embedding_dim": dim,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path}")

    print("\nDone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="Overwrite existing index.")
    p.add_argument("--limit", type=int, default=None, help="Only first N pages.")
    args = p.parse_args(argv)
    return build(rebuild=args.rebuild, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())