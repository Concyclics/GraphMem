from __future__ import annotations

import re
from typing import Any


def _query_operation_contract(question: str) -> str:
    normalized = " ".join(question.casefold().split())
    if (
        re.search(
            r"\bhow many more\b|\bhow many\b.*\b(?:remain|remaining|left)\b",
            normalized,
        )
        or re.search(
            r"\bhow many\b.*\bneed(?:ed)?\b.*\b(?:earn|gain|add|save|accumulate|reach)\b",
            normalized,
        )
    ):
        return "scalar_delta: answer = target amount - latest current amount"
    return "derive the exact answer form requested by the question"


def scalar_delta_proposal(
    question: str,
    evidence_text: str,
) -> dict[str, Any] | None:
    """Propose target-current arithmetic from explicitly bound evidence."""

    if not _query_operation_contract(question).startswith("scalar_delta:"):
        return None
    normalized = " ".join(evidence_text.casefold().split())
    target_patterns = (
        r"\bneed(?:s|ed)?\s+(?:a\s+)?total\s+of\s+(\d[\d,]*)",
        r"\b(?:target|goal)\s+of\s+(\d[\d,]*)",
        r"\breach(?:ing)?\s+(?:a\s+)?(\d[\d,]*)\s+\w+\s+(?:target|goal)\b",
    )
    current_patterns = (
        r"\b(?:currently|already)\s+(?:have|has|at)\s+(\d[\d,]*)",
        r"\bcurrent\s+(?:amount|balance|total)\s+(?:is|of)\s+(\d[\d,]*)",
        r"\bbringing\s+(?:my|the)\s+total\s+to\s+(\d[\d,]*)",
        r"\bwith\s+(\d[\d,]*)\s+(?:points?|credits?|miles?|tokens?)\b",
    )

    def values(patterns: tuple[str, ...]) -> list[int]:
        found: list[int] = []
        for pattern in patterns:
            for value in re.findall(pattern, normalized):
                try:
                    found.append(int(value.replace(",", "")))
                except ValueError:
                    continue
        return found

    targets = values(target_patterns)
    currents = values(current_patterns)
    if not targets or not currents:
        return None
    target = max(targets)
    valid_currents = [value for value in currents if 0 <= value < target]
    if not valid_currents:
        return None
    current = max(valid_currents)
    return {
        "operation": "target_minus_latest_current",
        "target": target,
        "current": current,
        "proposed_answer": target - current,
        "status": "verify_against_lossless_evidence",
    }


def direct_lossless_answer_messages(
    *,
    question: str,
    question_date: str | None,
    evidence_text: str,
) -> list[dict[str, str]]:
    """Minimal benchmark-neutral answer contract for lossless memories."""

    return [
        {
            "role": "system",
            "content": (
                "Answer the memory question using only the supplied conversation memories. "
                "Prefer exact statements in original turns over summaries. Keep speaker, "
                "entity, action, polarity, completion status, units, and dates exact. "
                "For counts, totals, comparisons, updates, or event order, use all relevant "
                "memories, deduplicate repeated mentions, and compute the requested result. "
                "When asked how many more are needed, remain, or must be earned, compute "
                "target minus the latest current amount; do not return the target itself. "
                "For current quantities, apply additions, removals, cancellations, and "
                "replacements to distinct named items before counting active items. "
                "For most/frequency questions, count completed occurrences, expand "
                "explicit multiplicities such as 'each way' or 'per batch', and exclude "
                "plans or recommendations. "
                "For previous-versus-current questions, order states by their evidence "
                "dates and report the older state before the latest state. "
                "Resolve relative dates from each memory's conversation date. If the "
                "memories state that an item was used in or brought to an event, treat its "
                "possession or arrival as preceding that event even if both were discussed "
                "later. "
                "If the exact "
                "requested entity or relation was never mentioned, say so and name the "
                "near-match only when useful. Recommendations must satisfy the concrete "
                "entities and constraints in the question and personal memories, rather "
                "than substitute generic nearby advice. A navigation candidate in the "
                "evidence header is a fallible extraction proposal: preserve it only when "
                "the cited original turns support every requested slot. A mechanically "
                "certified operator proposal is also fallible, but when its complete operand "
                "list and cited source turns agree, use its computed value even if a "
                "navigation candidate was based on fewer cards. Observation dates anchor "
                "relative expressions but are not necessarily event dates. For a "
                "fixed-numerator ratio, a smaller denominator means less of the denominator "
                "quantity per numerator unit. Return only the concise final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question date: {question_date or 'unknown'}\n"
                f"Question: {question}\n\n"
                f"Query operation: {_query_operation_contract(question)}\n\n"
                f"Conversation memories:\n{evidence_text}"
            ),
        },
    ]
