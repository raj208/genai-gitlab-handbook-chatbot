"""
Guardrails layer: input validation, prompt-injection screening, and
similarity-based retrieval gating.

Each function returns either None (input passed) or a `Block` dataclass
with a user-facing message and a machine-readable reason code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


MIN_QUERY_LEN = 3
MAX_QUERY_LEN = 1000

# Patterns that strongly suggest an attempt to override the system prompt.
# Not a security boundary — just a polite-deflection layer for the obvious cases.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\bforget\s+(everything|all|previous|prior)\b", re.I),
    re.compile(r"\b(reveal|show|print|leak|repeat)\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"\bact\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+", re.I),
    re.compile(r"<\s*/?\s*(system|s|sys)\s*>", re.I),
]

# Look like email addresses or US-style phone numbers — used only to add a
# friendly nudge, not to block the request.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b")


@dataclass(frozen=True)
class Block:
    message: str
    reason: str


def check_input(query: str) -> Block | None:
    """Run input-level checks. Returns Block if input is rejected, else None."""
    q = (query or "").strip()

    if len(q) < MIN_QUERY_LEN:
        return Block(
            message="Please enter a question with at least a few words.",
            reason="too_short",
        )

    if len(q) > MAX_QUERY_LEN:
        return Block(
            message=f"That question is too long ({len(q)} chars). Please keep it under {MAX_QUERY_LEN} characters.",
            reason="too_long",
        )

    for pat in _INJECTION_PATTERNS:
        if pat.search(q):
            return Block(
                message=(
                    "Your question looks like it's trying to override the assistant's "
                    "instructions. This assistant only answers questions about GitLab's "
                    "public handbook and direction. Please rephrase."
                ),
                reason="injection_pattern",
            )

    return None


def has_pii(query: str) -> bool:
    """True if the query appears to contain PII. Not a block — just a flag."""
    return bool(_EMAIL_RE.search(query) or _PHONE_RE.search(query))


# --- Confidence classification --------------------------------------------

# These thresholds work against cosine similarity in [-1, 1] from our
# normalized-vector + IndexFlatIP setup.
HIGH_CONFIDENCE = 0.45
LOW_CONFIDENCE_MIN = 0.30  # below this → refuse before calling the LLM


def classify_confidence(top_score: float) -> str:
    """Returns one of: 'high', 'low_confidence', 'refuse'."""
    if top_score >= HIGH_CONFIDENCE:
        return "high"
    if top_score >= LOW_CONFIDENCE_MIN:
        return "low_confidence"
    return "refuse"


REFUSAL_MESSAGE = (
    "I don't have enough information in the GitLab handbook to answer that. "
    "Try rephrasing, or ask about a topic covered in the public Handbook or Direction pages."
)