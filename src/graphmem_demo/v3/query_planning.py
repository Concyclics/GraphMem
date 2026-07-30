from __future__ import annotations

import re

from .schema import QueryFrame


_OPERATION_RELATIONS = {
    "count": "participant acquired owned received completed distinct entities quantities",
    "recurrence": "participant repeated event occurrences cadence interval frequency",
    "list": "participant asserted entities collection members",
    "ordering": "participant completed attended joined events dates chronological",
    "latest": "participant current latest valid state update",
    "earliest": "participant completed events dates earliest",
    "date": "participant event occurred completed explicit date",
    "planned_date": "participant planned scheduled event date",
    "duration": "event endpoints dates elapsed duration",
    "state": "participant current valid state update",
    "location": "participant event located held place",
    "recommendation": "participant preference constraint plan need",
    "preference_list": "participant likes dislikes preferences",
    "counterfactual": "condition cause outcome dependency",
    "lookup": "participant asserted event relation",
}


def answer_slot_phrase(question: str) -> str | None:
    """Extract a requested answer type such as kitchen appliance or game."""
    lowered = question.casefold().strip()
    patterns = (
        r"^how\s+many\s+(.+?)\s+(?:am|are|did|do|does|had|has|have|"
        r"is|was|were|will|would)\b",
        r"^what\s+(.+?)\s+(?:did|does|do)\b",
        r"^which\s+(.+?)\s+(?:did|does|do|was|were|is|are)\b",
        r"^what\s+(.+?)\s+(?:was|were|is|are)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = re.sub(r"\b(?:kind|type)\s+of\s+", "", match.group(1)).strip()
        if value and value not in {"did", "does", "do", "is", "was"}:
            return value
    return None


def query_views(frame: QueryFrame) -> list[str]:
    """Create bounded, topic-agnostic IR views without an LLM call."""
    content = " ".join(dict.fromkeys(
        [*frame.content_terms, *frame.participant_terms, *frame.temporal_terms]
    )).strip()
    relation = _OPERATION_RELATIONS.get(
        frame.requested_operation, _OPERATION_RELATIONS["lookup"]
    )
    slot = answer_slot_phrase(frame.raw_question)
    values = [
        frame.raw_question.strip(),
        content,
        f"{relation} {content}".strip(),
        slot or "",
    ]
    return list(dict.fromkeys(value for value in values if value))
