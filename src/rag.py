"""
RAG orchestration: retrieve → build prompt → call OpenAI → format response.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from src.config import settings
from src.retriever import Retriever, RetrievedChunk
from src.guardrails import check_input, classify_confidence, REFUSAL_MESSAGE



SYSTEM_PROMPT = """\
You are an assistant that answers questions about GitLab's company handbook
and product direction, strictly using the SOURCE PASSAGES provided by the user.

Rules you must follow:
1. Base every factual claim on the SOURCE PASSAGES. Do not use outside knowledge.
2. After each claim, cite the source it came from using bracketed numbers like
   [1] or [2, 3]. The numbers refer to the source order shown to you.
3. Citation rules — these are strict:
   - Never repeat the same number within one citation. Write [1], not [1, 1].
   - List numbers in ascending order: [1, 3], not [3, 1].
   - Only cite numbers that actually appear in the SOURCE PASSAGES section.
4. If the passages do not contain enough information to answer, reply exactly:
   "I don't have enough information in the GitLab handbook to answer that."
   Do not guess, speculate, or fall back on general knowledge.
5. Keep answers clear, neutral, and concise. Prefer short paragraphs.
6. Do not invent URLs, names, dates, or numbers that are not in the passages.
"""


@dataclass
class Source:
    n: int           # 1-based citation number shown to the user
    url: str
    title: str
    source: str      # "handbook" or "direction"
    best_score: float


@dataclass
class RagResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0
    confidence: str = "high"   # 'high' | 'low_confidence' | 'refuse'
    refused: bool = False
    refusal_reason: str = ""


def _dedupe_by_url(chunks: list[RetrievedChunk]) -> list[Source]:
    """Collapse chunks to one Source per URL, preserving rank order."""
    seen: dict[str, Source] = {}
    counter = 0
    for ch in chunks:
        if ch.url in seen:
            # Keep the highest score for that URL
            seen[ch.url].best_score = max(seen[ch.url].best_score, ch.score)
            continue
        counter += 1
        seen[ch.url] = Source(
            n=counter,
            url=ch.url,
            title=ch.title or ch.url,
            source=ch.source,
            best_score=ch.score,
        )
    return list(seen.values())


def _build_context_block(chunks: list[RetrievedChunk], sources: list[Source]) -> str:
    """
    Produce the SOURCE PASSAGES section. Each chunk is labelled with the
    citation number assigned to its URL in `sources`.
    """
    url_to_n = {s.url: s.n for s in sources}
    blocks: list[str] = []
    for ch in chunks:
        n = url_to_n[ch.url]
        header = f"[{n}] {ch.title or ch.url}\nURL: {ch.url}"
        blocks.append(f"{header}\n\n{ch.text}")
    return "\n\n---\n\n".join(blocks)


class RagPipeline:
    """End-to-end RAG. Instantiated once; .answer() called per query."""

    def __init__(self, retriever: Retriever | None = None, client: OpenAI | None = None):
        self.retriever = retriever or Retriever()
        self.client = client or OpenAI(api_key=settings.openai_api_key)

    def answer(self, question: str, top_k: int | None = None) -> RagResponse:
        # Input-level guardrails
        block = check_input(question)
        if block is not None:
            return RagResponse(
                answer=block.message,
                confidence="refuse",
                refused=True,
                refusal_reason=block.reason,
            )

        question = question.strip()
        retrieved = self.retriever.search(question, top_k=top_k)
        if not retrieved:
            return RagResponse(
                answer=REFUSAL_MESSAGE,
                confidence="refuse",
                refused=True,
                refusal_reason="no_results",
            )

        top_score = max(ch.score for ch in retrieved)
        confidence = classify_confidence(top_score)

        # Retrieval-level guardrail: refuse before calling the LLM
        if confidence == "refuse":
            return RagResponse(
                answer=REFUSAL_MESSAGE,
                retrieved=retrieved,
                top_score=top_score,
                confidence="refuse",
                refused=True,
                refusal_reason="low_similarity",
            )

        sources = _dedupe_by_url(retrieved)
        context = _build_context_block(retrieved, sources)
        user_message = (
            f"SOURCE PASSAGES:\n\n{context}\n\n"
            f"---\n\nQUESTION: {question}\n\n"
            f"Answer using only the passages above, citing sources with [n]."
        )

        resp = self.client.chat.completions.create(
            model=settings.chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer_text = resp.choices[0].message.content.strip()

        return RagResponse(
            answer=answer_text,
            sources=sources,
            retrieved=retrieved,
            top_score=top_score,
            confidence=confidence,
        )
    
    def answer_stream(self, question: str, top_k: int | None = None):
        """
        Streaming variant of .answer(). Yields:
          - {"type": "refusal", "answer": str, "reason": str}  (short-circuit)
          - {"type": "meta", "sources": [...], "retrieved": [...],
             "top_score": float, "confidence": str}            (before tokens)
          - str token deltas
          - {"type": "done"}
        """
        # Input-level guardrails
        block = check_input(question)
        if block is not None:
            yield {"type": "refusal", "answer": block.message, "reason": block.reason}
            return

        question = question.strip()
        retrieved = self.retriever.search(question, top_k=top_k)
        if not retrieved:
            yield {"type": "refusal", "answer": REFUSAL_MESSAGE, "reason": "no_results"}
            return

        top_score = max(ch.score for ch in retrieved)
        confidence = classify_confidence(top_score)

        if confidence == "refuse":
            yield {"type": "refusal", "answer": REFUSAL_MESSAGE, "reason": "low_similarity"}
            return

        sources = _dedupe_by_url(retrieved)
        context = _build_context_block(retrieved, sources)

        yield {
            "type": "meta",
            "sources": sources,
            "retrieved": retrieved,
            "top_score": top_score,
            "confidence": confidence,
        }

        user_message = (
            f"SOURCE PASSAGES:\n\n{context}\n\n"
            f"---\n\nQUESTION: {question}\n\n"
            f"Answer using only the passages above, citing sources with [n]."
        )

        stream = self.client.chat.completions.create(
            model=settings.chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta

        yield {"type": "done"}