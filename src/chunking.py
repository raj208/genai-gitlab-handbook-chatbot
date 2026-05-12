"""
Token-aware text chunking with sentence-boundary preference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

# Use cl100k_base — same tokenizer family used by OpenAI embedding + GPT-4 models.
# Avoids needing a model-specific encoder and works for our purposes.
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    token_count: int


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines; keep non-empty paragraphs."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter. Not perfect English grammar, but good
    enough for chunking and avoids pulling in nltk/spacy as dependencies.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last resort: split a too-long string by raw tokens."""
    tokens = _ENCODER.encode(text)
    out: list[str] = []
    for i in range(0, len(tokens), max_tokens):
        out.append(_ENCODER.decode(tokens[i:i + max_tokens]))
    return out


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Split text into overlapping chunks of ~chunk_size tokens.

    Strategy:
      1. Break into paragraphs.
      2. Greedily fill a window up to chunk_size tokens.
      3. If a single paragraph is too long, fall back to sentences.
      4. If a single sentence is too long, hard-split by tokens.
      5. Slide window forward leaving chunk_overlap tokens of overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    # Build a flat list of "units" (sentences) we can pack greedily.
    units: list[str] = []
    for para in _split_into_paragraphs(text):
        if _token_len(para) <= chunk_size:
            units.append(para)
        else:
            for sent in _split_into_sentences(para):
                if _token_len(sent) <= chunk_size:
                    units.append(sent)
                else:
                    units.extend(_hard_split(sent, chunk_size))

    if not units:
        return []

    # Pre-compute token counts once
    unit_tokens = [_token_len(u) for u in units]

    chunks: list[Chunk] = []
    i = 0
    n = len(units)
    while i < n:
        cur: list[str] = []
        cur_tokens = 0
        j = i
        while j < n and cur_tokens + unit_tokens[j] <= chunk_size:
            cur.append(units[j])
            cur_tokens += unit_tokens[j]
            j += 1

        # If the very first unit at position i didn't fit (shouldn't happen
        # given _hard_split, but defensive), force-include it.
        if not cur:
            cur.append(units[i])
            cur_tokens = unit_tokens[i]
            j = i + 1

        chunks.append(Chunk(text="\n\n".join(cur), token_count=cur_tokens))

        if j >= n:
            break

        # Step back to create overlap measured in tokens.
        back_tokens = 0
        k = j
        while k > i + 1 and back_tokens < chunk_overlap:
            back_tokens += unit_tokens[k - 1]
            k -= 1
        i = k

    return chunks