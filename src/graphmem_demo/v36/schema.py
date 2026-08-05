from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


GRAPHMEM_V36_SCHEMA = "graphmem_v3_6"

FrameKind = Literal[
    "fact", "event", "state", "preference", "quantity", "dialogue_answer"
]
GroupKind = Literal[
    "single_fact", "dialogue_pair", "state_transition", "collection",
    "temporal_pair", "contrast", "reference_chain",
]
EdgeRelation = Literal[
    "source", "next_turn", "dialogue_pair", "reference", "same_event",
    "state_transition", "collection_member", "temporal_endpoint", "contrast",
    "routing_contains", "semantic_neighbor",
]


@dataclass
class TurnNodeV36:
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
    schema_version: str = GRAPHMEM_V36_SCHEMA


@dataclass
class QuantityValue:
    value: float | None = None
    unit: str = ""
    multiplier: float | None = None


@dataclass
class TemporalValue:
    event_time: str | None = None
    observed_at: str | None = None
    start: str | None = None
    end: str | None = None
    precision: str = "unknown"
    anchor_source: str | None = None


@dataclass
class RoleFrameNode:
    frame_id: str
    question_id: str
    session_ids: list[str]
    frame_kind: FrameKind
    owner_key: str
    entity_key: str
    predicate_key: str
    object_key: str
    context_key: str = ""
    polarity: Literal["positive", "negative", "unknown"] = "positive"
    modality: Literal[
        "asserted", "planned", "possible", "conditional", "unknown"
    ] = "asserted"
    lifecycle_status: Literal[
        "proposed", "planned", "ongoing", "completed", "cancelled", "unknown"
    ] = "unknown"
    state_op: Literal[
        "set", "add", "remove", "increment", "decrement", "cancel",
        "complete", "none",
    ] = "none"
    quantity: QuantityValue = field(default_factory=QuantityValue)
    temporal: TemporalValue = field(default_factory=TemporalValue)
    event_identity_key: str = ""
    semantic_type_keys: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    retrieval_text: str = ""
    coverage_mask: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    observation_order: int = -1
    schema_version: str = GRAPHMEM_V36_SCHEMA

    @property
    def node_id(self) -> str:
        return self.frame_id


@dataclass
class RoutingCard:
    card_id: str
    question_id: str
    session_id: str
    speaker_keys: list[str]
    canonical_entities: list[str]
    relations: list[str]
    key_events: list[str]
    current_states: list[str]
    time_range: str
    frame_ids: list[str]
    turn_ids: list[str]
    routing_text: str
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V36_SCHEMA

    @property
    def node_id(self) -> str:
        return self.card_id

    @property
    def retrieval_text(self) -> str:
        return self.routing_text


@dataclass
class CoverageEntry:
    turn_id: str
    coverage_class: Literal[
        "memory_frame", "dialogue_context", "non_durable", "boilerplate",
        "lossless_only",
    ]
    frame_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceGroup:
    group_id: str
    question_id: str
    group_kind: GroupKind
    member_frame_ids: list[str]
    source_turn_ids: list[str]
    required_roles: list[str]
    completeness_mask: dict[str, bool]
    provenance_complete: bool
    confidence: float
    retrieval_text: str
    session_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V36_SCHEMA

    @property
    def node_id(self) -> str:
        return self.group_id


@dataclass
class GraphEdgeV36:
    edge_id: str
    question_id: str
    src: str
    dst: str
    relation: EdgeRelation
    directed: bool
    confidence: float
    provenance: dict[str, Any]
    role: str = ""
    schema_version: str = GRAPHMEM_V36_SCHEMA


@dataclass
class StateVersionV36:
    frame_id: str
    state_op: str
    object_key: str
    event_time: str | None
    observed_at: str | None
    valid_from: str | None
    valid_to: str | None
    source_turn_ids: list[str]


@dataclass
class StateChainV36:
    chain_id: str
    question_id: str
    owner_key: str
    entity_key: str
    attribute_key: str
    context_key: str
    versions: list[StateVersionV36]
    current_frame_ids: list[str]
    schema_version: str = GRAPHMEM_V36_SCHEMA


@dataclass
class QueryIR:
    raw_question: str
    target_entities: list[str]
    target_relation: str
    target_owner: str
    requested_value_type: Literal[
        "span", "entity", "list", "count", "date", "duration", "state",
        "preference", "recommendation", "temporal_order", "aggregate",
        "boolean",
    ]
    temporal_constraints: list[str]
    state_constraints: list[str]
    collection_constraints: list[str]
    polarity: Literal["positive", "negative", "unknown"]
    required_roles: list[str]
    comparison_targets: list[str] = field(default_factory=list)
    aggregation_op: Literal[
        "none", "sum", "average", "difference",
    ] = "none"
    operand_targets: list[str] = field(default_factory=list)
    schema_version: str = GRAPHMEM_V36_SCHEMA


@dataclass
class SourceSpanCandidate:
    source_turn_id: str
    session_id: str
    span_index: int
    text: str
    speaker_key: str
    transport_role: str
    target_terms: list[str]
    relation_terms: list[str]
    action_families: list[str]
    roles: list[str]
    lifecycle_status: str
    polarity: str
    event_time_text: str
    identity_keys: list[str]
    score: float
    provenance_complete: bool = True


@dataclass
class SourceSpanClosure:
    candidates: list[SourceSpanCandidate]
    selected_source_turn_ids: list[str]
    present_roles: list[str]
    missing_roles: list[str]
    target_support: dict[str, list[str]]
    complete: bool
    schema_version: str = GRAPHMEM_V36_SCHEMA


@dataclass
class CompletenessCertificate:
    entity_match: bool
    relation_match: bool
    scope_match: bool
    provenance_complete: bool
    present_roles: list[str]
    missing_roles: list[str]
    excluded_near_matches: list[str]
    complete: bool
    expansion_rounds: int = 0
    schema_version: str = GRAPHMEM_V36_SCHEMA


@dataclass
class V36Index:
    turns: list[TurnNodeV36] = field(default_factory=list)
    frames: list[RoleFrameNode] = field(default_factory=list)
    routing_cards: list[RoutingCard] = field(default_factory=list)
    evidence_groups: list[EvidenceGroup] = field(default_factory=list)
    edges: list[GraphEdgeV36] = field(default_factory=list)
    state_chains: list[StateChainV36] = field(default_factory=list)
    coverage: list[CoverageEntry] = field(default_factory=list)
    inverted_indexes: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    schema_version: str = GRAPHMEM_V36_SCHEMA


def index_from_dict(payload: dict[str, Any]) -> V36Index:
    def frame(row: dict[str, Any]) -> RoleFrameNode:
        value = dict(row)
        value["quantity"] = QuantityValue(**(value.get("quantity") or {}))
        value["temporal"] = TemporalValue(**(value.get("temporal") or {}))
        return RoleFrameNode(**value)

    def chain(row: dict[str, Any]) -> StateChainV36:
        value = dict(row)
        value["versions"] = [
            StateVersionV36(**item) for item in value.get("versions", [])
        ]
        return StateChainV36(**value)

    return V36Index(
        turns=[TurnNodeV36(**row) for row in payload.get("turns", [])],
        frames=[frame(row) for row in payload.get("frames", [])],
        routing_cards=[
            RoutingCard(**row) for row in payload.get("routing_cards", [])
        ],
        evidence_groups=[
            EvidenceGroup(**row) for row in payload.get("evidence_groups", [])
        ],
        edges=[GraphEdgeV36(**row) for row in payload.get("edges", [])],
        state_chains=[chain(row) for row in payload.get("state_chains", [])],
        coverage=[CoverageEntry(**row) for row in payload.get("coverage", [])],
        inverted_indexes={
            str(name): {
                str(key): list(ids) for key, ids in values.items()
            }
            for name, values in (payload.get("inverted_indexes") or {}).items()
        },
        schema_version=str(
            payload.get("schema_version") or GRAPHMEM_V36_SCHEMA
        ),
    )
