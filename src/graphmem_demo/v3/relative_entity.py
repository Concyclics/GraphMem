from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .catalog_schema import EventFrameV3, OperandRecordV3
from .schema import QueryFrame


def _subject_matches(frame: QueryFrame, subject_key: str) -> bool:
    lowered = subject_key.casefold()
    named = {value.casefold() for value in frame.participant_terms}
    if named:
        return any(value in lowered for value in named)
    if re.search(r"\b(?:i|me|my)\b", frame.raw_question.casefold()):
        return "participant 2" not in lowered and "assistant" not in lowered
    return True


def resolve_relative_entity(
    frame: QueryFrame,
    hint: dict[str, Any] | None,
    nodes: dict[str, Any],
    operands: list[OperandRecordV3],
    *,
    similarity: Callable[[Any], float],
) -> dict[str, Any] | None:
    """Resolve a dated event/frame back to its exact typed object operand."""
    if not hint or not hint.get("within_tolerance"):
        return None
    support_ids = list(hint.get("supporting_node_ids", []))
    if len(support_ids) != 1:
        return None
    support = nodes.get(support_ids[0])
    if support is None:
        return None
    if isinstance(support, OperandRecordV3):
        candidates = [support]
        relation = "direct_operand"
    elif isinstance(support, EventFrameV3):
        candidates = [
            item for item in operands if item.event_frame_id == support.frame_id
        ]
        relation = "event_frame_member"
    else:
        source_id = getattr(support, "node_id", "")
        candidates = [
            item for item in operands if source_id in item.source_turn_ids
        ]
        relation = "source_projection"
    candidates = [
        item for item in candidates
        if item.object_text.strip()
        and item.polarity != "negative"
        and item.modality not in {"planned", "possible", "hypothetical"}
        and _subject_matches(frame, item.subject_key)
    ]
    if not candidates:
        return None
    item = max(candidates, key=lambda value: (
        similarity(value), value.confidence, value.operand_id,
    ))
    return {
        "operation": "relative_time_entity_binding",
        "value": item.object_text,
        "operand_id": item.operand_id,
        "supporting_node_ids": [support_ids[0], item.operand_id],
        "source_turn_ids": list(item.source_turn_ids),
        "relation": relation,
        "complete": True,
    }
