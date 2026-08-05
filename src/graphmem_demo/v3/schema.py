from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .catalog_schema import EventFrameV3, OperandRecordV3


GRAPHMEM_V3_SCHEMA = "graphmem_v3"


@dataclass
class TurnNode:
    node_id: str
    question_id: str
    session_id: str
    session_date: str | None
    turn_index: int
    speaker: str
    speaker_key: str
    listener: str
    transport_role: str
    text: str
    retrieval_text: str
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class ClaimNode:
    node_id: str
    question_id: str
    session_id: str
    subject: str
    subject_key: str
    predicate: str
    predicate_key: str
    object: str
    object_key: str
    kind: Literal[
        "state", "event", "preference", "quantity", "general"
    ] = "general"
    polarity: Literal["positive", "negative", "unknown"] = "positive"
    modality: Literal[
        "asserted", "planned", "possible", "conditional", "unknown"
    ] = "asserted"
    state_op: Literal[
        "assert", "retract", "replace", "add", "remove", "complete", "cancel", "none"
    ] = "none"
    context_key: str = ""
    event_time: str | None = None
    observed_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    quantity: float | None = None
    unit: str = ""
    source_turn_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    retrieval_text: str = ""
    embedding: list[float] | None = None
    observation_order: int = -1
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class EventNode:
    node_id: str
    question_id: str
    session_id: str
    label: str
    label_key: str
    status: Literal[
        "asserted", "planned", "possible", "complete", "cancelled", "unknown"
    ] = "asserted"
    participant_keys: list[str] = field(default_factory=list)
    event_time: str | None = None
    claim_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    semantic_type_keys: list[str] = field(default_factory=list)
    confidence: float = 1.0
    retrieval_text: str = ""
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class EventEntityNode:
    """Query-independent identity carried across multiple event mentions."""

    node_id: str
    question_id: str
    canonical_label: str
    canonical_key: str
    member_event_ids: list[str]
    anchor_terms: list[str] = field(default_factory=list)
    participant_keys: list[str] = field(default_factory=list)
    semantic_type_keys: list[str] = field(default_factory=list)
    lifecycle_status: Literal[
        "asserted", "planned", "possible", "complete", "cancelled", "unknown"
    ] = "unknown"
    current_event_id: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    source_turn_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    retrieval_text: str = ""
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class EpisodeNode:
    node_id: str
    question_id: str
    session_id: str
    session_date: str | None
    label: str
    participant_keys: list[str]
    time_start: str | None
    time_end: str | None
    turn_ids: list[str]
    claim_ids: list[str]
    event_ids: list[str]
    retrieval_text: str
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class ThemeNode:
    node_id: str
    question_id: str
    labels: list[str]
    participant_keys: list[str]
    time_start: str | None
    time_end: str | None
    episode_ids: list[str]
    claim_ids: list[str]
    event_ids: list[str]
    source_turn_ids: list[str]
    retrieval_text: str
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class HyperIncidence:
    node_id: str
    role: str
    order: int | None = None


@dataclass
class HyperEdge:
    edge_id: str
    question_id: str
    relation: Literal[
        "episode_member",
        "theme_member",
        "participant",
        "same_event",
        "event_entity_member",
        "state_history",
        "temporal_scope",
        "quantity_collection",
        "supports",
        "refers_to",
        "contradiction",
        "semantic_cluster",
        "event_frame_member",
        "operand_projection",
    ]
    incidences: list[HyperIncidence]
    directed: bool = False
    confidence: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    retrieval_text: str = ""
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class StateChainV3:
    chain_id: str
    question_id: str
    subject_key: str
    predicate_key: str
    context_key: str
    current_claim_ids: list[str]
    history_claim_ids: list[str]
    update_order: list[str]
    valid_from: str | None = None
    valid_to: str | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class QueryFrame:
    raw_question: str
    content_terms: list[str]
    participant_terms: list[str]
    temporal_terms: list[str]
    explicit_dates: list[str]
    requested_operation: Literal[
        "lookup", "list", "count", "recurrence", "latest", "earliest", "date", "duration", "state", "recommendation", "ordering", "location", "counterfactual", "preference_list", "planned_date"
    ]
    answer_form: Literal[
        "span", "entity", "list", "number", "frequency", "date", "duration", "state", "recommendation"
    ]
    hypotheses: list[str]
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class ClosureCertificate:
    requested_operation: str
    complete: bool
    visited_hyperedge_ids: list[str]
    operand_node_ids: list[str]
    contradiction_node_ids: list[str]
    missing_requirements: list[str]
    truncated: bool
    provenance_complete: bool
    scope_description: str
    schema_version: str = GRAPHMEM_V3_SCHEMA


@dataclass
class V3Index:
    turns: list[TurnNode] = field(default_factory=list)
    claims: list[ClaimNode] = field(default_factory=list)
    events: list[EventNode] = field(default_factory=list)
    event_entities: list[EventEntityNode] = field(default_factory=list)
    episodes: list[EpisodeNode] = field(default_factory=list)
    themes: list[ThemeNode] = field(default_factory=list)
    hyperedges: list[HyperEdge] = field(default_factory=list)
    state_chains: list[StateChainV3] = field(default_factory=list)
    event_frames: list[EventFrameV3] = field(default_factory=list)
    operands: list[OperandRecordV3] = field(default_factory=list)
    schema_version: str = GRAPHMEM_V3_SCHEMA


def index_from_dict(payload: dict[str, Any]) -> V3Index:
    return V3Index(
        turns=[TurnNode(**row) for row in payload.get("turns", [])],
        claims=[ClaimNode(**row) for row in payload.get("claims", [])],
        events=[EventNode(**row) for row in payload.get("events", [])],
        event_entities=[
            EventEntityNode(**row) for row in payload.get("event_entities", [])
        ],
        episodes=[EpisodeNode(**row) for row in payload.get("episodes", [])],
        themes=[ThemeNode(**row) for row in payload.get("themes", [])],
        hyperedges=[
            HyperEdge(
                **{
                    **row,
                    "incidences": [
                        HyperIncidence(**incidence)
                        for incidence in row.get("incidences", [])
                    ],
                }
            )
            for row in payload.get("hyperedges", [])
        ],
        state_chains=[
            StateChainV3(**row) for row in payload.get("state_chains", [])
        ],
        event_frames=[
            EventFrameV3(**row) for row in payload.get("event_frames", [])
        ],
        operands=[
            OperandRecordV3(**row) for row in payload.get("operands", [])
        ],
        schema_version=str(payload.get("schema_version") or GRAPHMEM_V3_SCHEMA),
    )
