from __future__ import annotations

import re
from typing import Any

from ..clients import rough_token_count
from .catalog_schema import EventFrameV3, OperandRecordV3
from .schema import (
    ClaimNode, EpisodeNode, EventEntityNode, EventNode, QueryFrame, ThemeNode,
    TurnNode,
)


_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


def _token_key(value: str) -> str:
    value = value.casefold().strip("'\"")
    if value.endswith("'s"):
        value = value[:-2]
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _tokens(text: str) -> set[str]:
    normalized = text.replace("_", " ").replace("-", " ")
    return {_token_key(value) for value in _WORD_RE.findall(normalized)}


def _overlap(frame: QueryFrame, text: str) -> float:
    query = set(
        frame.content_terms + frame.participant_terms + frame.temporal_terms
    )
    return len(query & _tokens(text)) / max(1, len(query))


def _focused_text(text: str, frame: QueryFrame, limit: int) -> str:
    if len(text) <= limit:
        return text
    segments = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    ]
    query_terms = set(frame.content_terms)
    segment_tokens = [_tokens(value) for value in segments]
    document_frequency = {
        term: sum(term in terms for terms in segment_tokens)
        for term in query_terms
    }
    value_pattern = (
        r"\b(?:19|20)\d{2}(?:[-\/]\d{1,2}(?:[-\/]\d{1,2})?)?\b"
        if frame.answer_form == "date"
        else r"(?<!\w)(?:[$]\s?)?\d+(?:[.,]\d+)?(?:\s?%)?(?!\w)"
    )
    expects_value = frame.answer_form in {"date", "number", "duration"}
    ranked = sorted(
        enumerate(segments),
        key=lambda item: (
            int(expects_value and bool(re.search(value_pattern, item[1]))),
            sum(1.0 / max(1, document_frequency[term])
                for term in query_terms & segment_tokens[item[0]]),
            _overlap(frame, item[1]),
            -item[0],
        ),
        reverse=True,
    )
    selected: list[int] = []
    for index, _value in ranked:
        if index not in selected:
            selected.append(index)
        # Preserve a structured section heading together with its first value row.
        # This keeps tables, lists, code, scores, and symbolic sequences grounded.
        if (
            index + 1 < len(segments)
            and len(segments[index]) <= 80
            and segments[index].rstrip().endswith(":")
            and index + 1 not in selected
        ):
            selected.append(index + 1)
        if len(selected) >= 6:
            break
    return " ".join(segments[index] for index in selected[:6])[:limit]


def render_block(frame: QueryFrame, kind: str, node: Any) -> str:
    if isinstance(node, TurnNode):
        return (
            f"[TURN {node.node_id} | session={node.session_id} | "
            f"date={node.session_date or 'unknown'} | speaker={node.speaker}]\n"
            f"{_focused_text(node.text, frame, 1200)}"
        )
    if isinstance(node, ClaimNode):
        return (
            f"[CLAIM {node.node_id} | "
            f"event_time={node.event_time or 'unknown'} | "
            f"observed_at={node.observed_at or 'unknown'} | "
            f"modality={node.modality} | polarity={node.polarity} | "
            f"sources={','.join(node.source_turn_ids[:2])}]\n"
            f"{node.subject} | {node.predicate} | "
            f"{_focused_text(node.object, frame, 600)}"
        )
    if isinstance(node, EventNode):
        semantic_types = ",".join(node.semantic_type_keys) or "unknown"
        return (
            f"[EVENT {node.node_id} | time={node.event_time or 'unknown'} | "
            f"status={node.status} | types={semantic_types} | sources={','.join(node.source_turn_ids[:2])}]\n"
            f"{_focused_text(node.label, frame, 480)}"
        )
    if isinstance(node, EventEntityNode):
        return (
            f"[EVENT_ENTITY {node.node_id} | lifecycle={node.lifecycle_status} | "
            f"time_start={node.time_start or 'unknown'} | time_end={node.time_end or 'unknown'} | "
            f"members={','.join(node.member_event_ids)} | "
            f"anchors={','.join(node.anchor_terms)}]\n"
            f"{_focused_text(node.canonical_label, frame, 360)}"
        )
    if isinstance(node, OperandRecordV3):
        recurrence = ",".join(node.recurrence_days) or "none"
        event_types = ",".join(node.event_type_keys) or "unknown"
        return (
            f"[OPERAND {node.operand_id} | event_time={node.event_time or 'unknown'} | "
            f"observed_at={node.observed_at or 'unknown'} | "
            f"modality={node.modality} | polarity={node.polarity} | recurrence={recurrence} | "
            f"event_types={event_types} | "
            f"sources={', '.join(node.source_turn_ids[:2])}]\n"
            f"{node.subject_key} | {node.predicate_key} | "
            f"{_focused_text(node.object_text, frame, 480)}"
        )
    if isinstance(node, EventFrameV3):
        semantic_types = ",".join(node.semantic_type_keys) or "unknown"
        return (
            f"[EVENT_FRAME {node.frame_id} | status={node.status} | "
            f"event_time={node.event_time or 'unknown'} | "
            f"observed_at={node.observed_at or 'unknown'} | types={semantic_types} | "
            f"sessions={', '.join(node.session_ids[:3])} | "
            f"sources={', '.join(node.source_turn_ids[:2])}]\n"
            f"{_focused_text(node.retrieval_text, frame, 520)}"
        )
    if isinstance(node, EpisodeNode):
        return (
            f"[EPISODE {node.node_id} | session={node.session_id} | "
            f"turn_count={len(node.turn_ids)} | "
            f"sources={','.join(node.turn_ids[:2])}]\n"
            f"{_focused_text(node.retrieval_text, frame, 520)}"
        )
    if isinstance(node, ThemeNode):
        return (
            f"[THEME {node.node_id} | episode_count={len(node.episode_ids)} | "
            f"sources={','.join(node.episode_ids[:3])}]\n"
            f"{_focused_text(node.retrieval_text, frame, 360)}"
        )
    return f"[{kind.upper()}]\n{_focused_text(str(node), frame, 500)}"


def pack_context(
    frame: QueryFrame,
    ordered: list[tuple[str, Any, float, str]],
    budget: int,
) -> tuple[list[tuple[str, Any, float, str]], str, list[dict[str, Any]]]:
    prepared = []
    for kind, node, score, source in ordered:
        block = render_block(frame, kind, node)
        cost = rough_token_count(block)
        priority = (
            2.75 if source == "protected_direct" else
            1.85 if source == "relation_focus" else
            2.85 if source == "catalog_operator_provenance" else
            2.85 if source == "temporal_operator_provenance" else
            2.85 if source == "location_operator_provenance" else
            2.85 if source == "lifecycle_operator_provenance" else
            2.85 if source == "contrast_operator_provenance" else
            2.85 if source == "relation_operator_provenance" else
            2.85 if source == "reference_chain_provenance" else
            2.85 if source == "event_identity_provenance" else
            2.90 if source == "event_entity_provenance" else
            2.85 if source == "recommendation_resource_provenance" else
            2.90 if source == "focused_provenance_expansion" else
            2.65 if source == "scope_total_index" else
            2.65 if source == "scope_lossless_event" else
            1.08 if source == "protected_catalog" else
            1.06 if source == "protected_catalog_dense" else
            1.10 if source == "scope_local_turn_primary" else
            1.22 if source == "scope_local_turn_adjacent" else
            1.15 if (
                source == "scope_local_turn_secondary"
                and frame.requested_operation == "ordering"
            ) else
            0.45 if source == "scope_local_turn_secondary" else
            1.35 if source == "scope_local_turn_unprojected" else
            1.35 if source == "scope_local_projection_primary" else
            1.15 if source == "scope_local_projection_secondary" else
            1.55 if source == "coarse_fine_projection" else
            0.82 if source == "protected_graph_rescue" else
            0.42 if source == "catalog_provenance" else
            0.26 if source == "provenance_expansion" else 0.0
        )
        priority += 0.55 * _overlap(
            frame, getattr(node, "retrieval_text", "")
        )
        if kind in {"claim", "event", "event_entity", "operand"}:
            priority += 0.18
        elif kind in {"episode", "theme"}:
            priority += 0.10
        priority += min(score, 2.0) * 0.12
        priority -= min(cost, 1200) / 6000.0
        prepared.append((priority, kind, node, score, source, block, cost))
    prepared.sort(
        key=lambda item: (
            item[0],
            -item[6],
            getattr(item[2], "node_id", ""),
        ),
        reverse=True,
    )
    kept: list[tuple[str, Any, float, str]] = []
    blocks: list[str] = []
    decisions: list[dict[str, Any]] = []
    used = 0
    for priority, kind, node, score, source, block, cost in prepared:
        decision = "keep" if used + cost <= budget else "drop_budget"
        decisions.append({
            "node_id": getattr(node, "node_id", ""),
            "decision": decision,
            "rough_tokens": cost,
            "source": source,
            "pack_priority": round(priority, 6),
        })
        if decision != "keep":
            continue
        kept.append((kind, node, score, source))
        blocks.append(block)
        used += cost
    return kept, "\n\n".join(blocks), decisions
