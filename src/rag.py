"""
RAG orchestration: retrieve → build prompt → call OpenAI → format response.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from src.config import settings
from src.retriever import Retriever, RetrievedChunk


SYSTEM_PROMPT = """\
You are an assistant that answers questions about GitLab's company handbook
and product direction, strictly using the SOURCE PASSAGES provided by the user.

Rules you must follow:
1. Base every factual claim on the SOURCE PASSAGES. Do not use outside knowledge.
2. After each claim, cite the source it came from using bracketed numbers like
   [1] or [2, 3]. The numbers refer to the source order shown to you.
3. If the passages do not contain enough information to answer, reply exactly:
   "I don't have enough information in the GitLab handbook to answer that."
   Do not guess, speculate, or fall back on general knowledge.
4. Keep answers clear, neutral, and concise. Prefer short paragraphs.
5. Do not invent URLs, names, dates, or numbers that are not in the passages.
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
    # Flags filled in by Step 6 guardrails layer later
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
        question = (question or "").strip()
        if not question:
            return RagResponse(
                answer="Please enter a question.",
                refused=True,
                refusal_reason="empty_query",
            )

        retrieved = self.retriever.search(question, top_k=top_k)
        if not retrieved:
            return RagResponse(
                answer="I don't have enough information in the GitLab handbook to answer that.",
                refused=True,
                refusal_reason="no_results",
            )

        sources = _dedupe_by_url(retrieved)
        context = _build_context_block(retrieved, sources)
        top_score = max(ch.score for ch in retrieved)

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
        )
    
    
    def answer_stream(self, question: str, top_k: int | None = None):
        """
        Streaming variant of .answer().

        Yields a sequence of items:
          - dict {"type": "meta", "sources": [...], "retrieved": [...], "top_score": float}
            sent once at the start, before any tokens
          - str tokens as they arrive from the LLM
          - dict {"type": "done"} at the end

        For empty queries or no-result cases, yields a single dict with
        {"type": "refusal", "answer": str, "reason": str} and nothing else.
        """
        question = (question or "").strip()
        if not question:
            yield {
                "type": "refusal",
                "answer": "Please enter a question.",
                "reason": "empty_query",
            }
            return

        retrieved = self.retriever.search(question, top_k=top_k)
        if not retrieved:
            yield {
                "type": "refusal",
                "answer": "I don't have enough information in the GitLab handbook to answer that.",
                "reason": "no_results",
            }
            return

        sources = _dedupe_by_url(retrieved)
        context = _build_context_block(retrieved, sources)
        top_score = max(ch.score for ch in retrieved)

        # Emit metadata first so the UI can render the source panel
        # before tokens start arriving.
        yield {
            "type": "meta",
            "sources": sources,
            "retrieved": retrieved,
            "top_score": top_score,
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