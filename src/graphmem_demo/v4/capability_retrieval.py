from __future__ import annotations

import math
import re

from ..clients import rough_token_count
from ..models import RetrievedContext
from ..v36.schema import V36Index
from .schema import CapabilityViewV4


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / denominator if denominator else 0.0


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", text)
        if len(token) > 2
    }


def supplement_capability_gaps(
    *,
    result: RetrievedContext,
    index: V36Index,
    capability_view: CapabilityViewV4,
    requested: list[str],
    query_vectors: list[list[float]],
    question: str,
    token_budget: int,
) -> list[dict[str, object]]:
    """Fill capability gaps inside routed sessions; never widen the coarse scope."""
    selected_ids = set(result.fact_node_ids)
    missing = [
        name for name in requested
        if name != "fact"
        and not selected_ids.intersection(
            capability_view.frame_ids_by_capability.get(name, [])
        )
    ]
    selected_sessions = set(result.retrieved_session_ids)
    if not missing or not selected_sessions:
        return []

    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    question_terms = _terms(question)
    rows: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for capability in missing:
        for frame_id in capability_view.frame_ids_by_capability.get(capability, []):
            if frame_id in seen or frame_id not in frame_by_id:
                continue
            frame = frame_by_id[frame_id]
            if not selected_sessions.intersection(frame.session_ids):
                continue
            seen.add(frame_id)
            dense = max(
                (_cosine(frame.embedding or [], vector) for vector in query_vectors),
                default=0.0,
            )
            lexical = len(question_terms.intersection(_terms(frame.retrieval_text)))
            rows.append(
                (dense + min(0.25, lexical * 0.04), capability, frame_id)
            )
    rows.sort(key=lambda item: (-item[0], item[2]))

    supplements: list[dict[str, object]] = []
    context = result.context_text
    for score, capability, frame_id in rows:
        if len(supplements) >= 4:
            break
        frame = frame_by_id[frame_id]
        source_lines: list[str] = []
        for source_id in frame.source_turn_ids[:2]:
            turn = turn_by_id.get(source_id)
            if turn is not None:
                source_lines.append(
                    f"source={source_id}; speaker={turn.speaker_key}; "
                    f"text={turn.text[:360]}"
                )
        block = "\n".join([
            f"[V4_CAPABILITY_EVIDENCE {frame_id}; capability={capability}]",
            frame.retrieval_text[:520],
            *source_lines,
        ])
        added_tokens = rough_token_count(block)
        if result.packed_rough_tokens + added_tokens > token_budget:
            continue
        context = f"{context}\n\n{block}" if context else block
        result.packed_rough_tokens += added_tokens
        result.fact_node_ids.append(frame_id)
        for source_id in frame.source_turn_ids:
            if source_id not in result.evidence_leaf_ids:
                result.evidence_leaf_ids.append(source_id)
        ledger_row: dict[str, object] = {
            "kind": "v4_capability_projection",
            "capability": capability,
            "frame_id": frame_id,
            "source_turn_ids": list(frame.source_turn_ids),
            "score": round(score, 6),
            "provenance_complete": bool(frame.source_turn_ids),
        }
        result.evidence_ledger.append(ledger_row)
        supplements.append(ledger_row)
    result.context_text = context
    return supplements


__all__ = ["supplement_capability_gaps"]
