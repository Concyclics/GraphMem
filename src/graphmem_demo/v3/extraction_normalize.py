from __future__ import annotations

from typing import Any


def _first(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return []


def _claim_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return [
        row.get("subject"),
        row.get("predicate"),
        _first(row, "object", "value", "fact", "content"),
        row.get("kind", "general"),
        row.get("polarity", "positive"),
        row.get("modality", "asserted"),
        row.get("state_op", row.get("operation", "none")),
        _first(row, "context", "context_key"),
        _first(row, "event_time", "time", "when"),
        _first(
            row,
            "source_turn_ids",
            "source_turns",
            "source_ids",
            "sources",
            "turn_ids",
            "evidence_turn_ids",
            "source_turn_id",
            "source",
            default=[],
        ),
        row.get("quantity"),
        row.get("unit"),
        row.get("confidence"),
    ]


def _event_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return [
        _first(row, "label", "event", "name"),
        row.get("status", "unknown"),
        _first(row, "event_time", "time", "when"),
        _first(row, "participant_names", "participants", default=[]),
        _first(row, "claim_indices", "claims", default=[]),
        _first(
            row,
            "source_turn_ids",
            "source_turns",
            "source_ids",
            "sources",
            "turn_ids",
            "evidence_turn_ids",
            "source_turn_id",
            "source",
            default=[],
        ),
        row.get("confidence"),
        row.get("semantic_type_keys", row.get("event_types", row.get("types", []))),
    ]


def normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept compact arrays and semantically equivalent object-shaped output."""
    nested = _first(payload, "memory_graph", "graph", "extraction", "result", "data")
    result = dict(nested) if isinstance(nested, dict) else dict(payload)
    claims = _first(result, "claims", "facts", "atomic_facts", default=[])
    result["claims"] = [
        _claim_row(row) for row in _rows(claims)
    ]
    events = _first(result, "events", "event_nodes", default=[])
    result["events"] = [
        _event_row(row) for row in _rows(events)
    ]
    episodes = _first(result, "episodes", "episode_nodes", default=[])
    result["episodes"] = _rows(episodes)
    return result
