from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


GRAPHMEM_V4_SCHEMA = "graphmem_v4_0"
V4_BUILD_VERSION = "4.0.0"
V4_RETRIEVAL_VERSION = "4.0.0"


@dataclass
class CapabilityViewV4:
    """Question-independent projections over the single physical role graph."""

    topology_mode: str
    speaker_keys: list[str]
    memory_source_turn_ids: list[str]
    dialogue_context_turn_ids: list[str]
    frame_ids_by_capability: dict[str, list[str]] = field(default_factory=dict)
    group_ids_by_kind: dict[str, list[str]] = field(default_factory=dict)
    turn_ids_by_speaker: dict[str, list[str]] = field(default_factory=dict)
    state_chain_ids: list[str] = field(default_factory=list)
    source_coverage_complete: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = GRAPHMEM_V4_SCHEMA


def capability_view_from_dict(payload: dict[str, Any]) -> CapabilityViewV4:
    return CapabilityViewV4(
        topology_mode=str(payload.get("topology_mode") or "assistant_mediated"),
        speaker_keys=[str(value) for value in payload.get("speaker_keys", [])],
        memory_source_turn_ids=[
            str(value) for value in payload.get("memory_source_turn_ids", [])
        ],
        dialogue_context_turn_ids=[
            str(value) for value in payload.get("dialogue_context_turn_ids", [])
        ],
        frame_ids_by_capability={
            str(key): [str(value) for value in values]
            for key, values in (payload.get("frame_ids_by_capability") or {}).items()
        },
        group_ids_by_kind={
            str(key): [str(value) for value in values]
            for key, values in (payload.get("group_ids_by_kind") or {}).items()
        },
        turn_ids_by_speaker={
            str(key): [str(value) for value in values]
            for key, values in (payload.get("turn_ids_by_speaker") or {}).items()
        },
        state_chain_ids=[
            str(value) for value in payload.get("state_chain_ids", [])
        ],
        source_coverage_complete=bool(
            payload.get("source_coverage_complete", False)
        ),
        diagnostics=dict(payload.get("diagnostics") or {}),
        schema_version=str(payload.get("schema_version") or GRAPHMEM_V4_SCHEMA),
    )
