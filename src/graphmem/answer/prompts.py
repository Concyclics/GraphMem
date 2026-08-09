"""The answer contract.

The prompt text is frozen behind ``PROMPT_HASH``.  Every ablation arm compares
answers produced under one prompt; editing this file mid-study silently
invalidates every arm that ran before the edit, so the hash is asserted into
each run manifest.

The contract is benchmark-neutral: it names no dataset, no question type
taxonomy and no gold field.  ``graphmem.answer`` must stay importable without
``graphmem.eval``.
"""
from __future__ import annotations

import hashlib
import re

PROMPT_VERSION = "graphmem-v5.6-answer-v1"

ANSWER_SYSTEM_PROMPT = (
    "Answer the question using only the supplied conversation memories. "
    "Prefer exact statements in the original turns over summaries. Keep speaker, "
    "entity, action, polarity, completion status, units, and dates exact. "
    "For counts, totals, comparisons, updates, or event order, use all relevant "
    "memories, deduplicate repeated mentions, and compute the requested result. "
    "When asked how many more are needed, remain, or must be earned, compute "
    "target minus the latest current amount; do not return the target itself. "
    "For current quantities, apply additions, removals, cancellations, and "
    "replacements to distinct named items before counting active items. "
    "For most/frequency questions, count completed occurrences, expand explicit "
    "multiplicities such as 'each way' or 'per batch', and exclude plans or "
    "recommendations. "
    "For previous-versus-current questions, order states by their evidence dates "
    "and report the older state before the latest state. "
    "Resolve relative dates from each memory's conversation date. Observation "
    "dates anchor relative expressions but are not necessarily event dates. If "
    "the memories state that an item was used in or brought to an event, treat "
    "its possession or arrival as preceding that event even if both were "
    "discussed later. "
    "If the exact requested entity or relation was never mentioned, say so and "
    "name the near-match only when useful. "
    "A computed candidate shown under 'Candidate answer' is a fallible "
    "mechanical proposal, not evidence: use its value only when the cited "
    "memories support every part of it, and otherwise ignore it. "
    "Return only the concise final answer."
)

SOURCE_TIME_PROMPT_VERSION = "graphmem-v5.12-answer-source-time-v1"
SOURCE_TIME_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT.replace(
    "Resolve relative dates from each memory's conversation date. Observation ",
    "Resolve relative dates inside each memory from that memory's conversation "
    "date, never from the question date. The question date anchors only relative "
    "phrases in the question itself. A [source-time ...] annotation is the "
    "deterministic resolution of a memory phrase and must be used as written. Observation ",
)


def _query_operation_contract(question: str) -> str:
    """Name the arithmetic contract when the question form fixes it.

    Kept from V3 because 'how many more' questions are answered with the target
    rather than the delta often enough to be worth stating explicitly.
    """
    normalized = " ".join(question.casefold().split())
    if (re.search(r"\bhow many more\b|\bhow many\b.*\b(?:remain|remaining|left)\b", normalized)
            or re.search(r"\bhow many\b.*\bneed(?:ed)?\b.*\b(?:earn|gain|add|save|accumulate|reach)\b",
                         normalized)):
        return "scalar_delta: answer = target amount - latest current amount"
    return "derive the exact answer form requested by the question"


def build_answer_messages(
    *,
    question: str,
    question_date: str | None,
    evidence_text: str,
    candidate_answer: str | None = None,
    normalize_relative_time: bool = False,
) -> list[dict[str, str]]:
    sections = [
        f"Question date: {question_date or 'unknown'}",
        f"Question: {question}",
        "",
        f"Query operation: {_query_operation_contract(question)}",
    ]
    if candidate_answer:
        sections += ["", f"Candidate answer (unverified proposal): {candidate_answer}"]
    sections += ["", "Conversation memories:", evidence_text]
    return [
        {"role": "system", "content": prompt_contract(
            normalize_relative_time)[1]},
        {"role": "user", "content": "\n".join(sections)},
    ]


PROMPT_HASH = hashlib.sha256(
    (PROMPT_VERSION + ANSWER_SYSTEM_PROMPT).encode("utf-8")
).hexdigest()


def prompt_contract(normalize_relative_time: bool = False) -> tuple[str, str, str]:
    """Return the exact version/text/hash for an answer configuration."""

    version, prompt = (
        (SOURCE_TIME_PROMPT_VERSION, SOURCE_TIME_SYSTEM_PROMPT)
        if normalize_relative_time else (PROMPT_VERSION, ANSWER_SYSTEM_PROMPT))
    return version, prompt, hashlib.sha256((version + prompt).encode("utf-8")).hexdigest()
