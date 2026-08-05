from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "graphmem-v5"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, StrEnum):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )


def stable_id(kind: str, *parts: Any) -> str:
    """Return a deterministic, namespace-labelled 128-bit content ID."""
    namespace = kind.strip().casefold().replace("_", "-")
    if not namespace or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in namespace):
        raise ValueError(f"invalid ID namespace: {kind!r}")
    payload = canonical_json([namespace, *parts]).encode("utf-8")
    return f"{namespace}:{hashlib.sha256(payload).hexdigest()[:32]}"


class NodeType(StrEnum):
    SCENE = "scene"
    ENTITY_MENTION = "entity_mention"
    CANONICAL_ENTITY = "canonical_entity"
    EVENT_SKELETON = "event_skeleton"
    COLLECTION_SCOPE = "collection_scope"
    STATE_HEAD = "state_head"
    ROUTING_CARD = "routing_card"
    EVENT_FRAME = "event_frame"
    ROLE_FRAME = "role_frame"
    ENTITY = "entity"
    TIME_ANCHOR = "time_anchor"
    STATE_VALUE = "state_value"
    SESSION = "session"
    EVIDENCE_GROUP_REF = "evidence_group_ref"
    CANONICAL_FACT = "canonical_fact"
    CANONICAL_VALUE = "canonical_value"
    FACT_MENTION = "fact_mention"
    VIRTUAL_REGION = "virtual_region"


class RelationType(StrEnum):
    PORTAL = "portal"
    SCENE_CONTAINS = "scene_contains"
    MEMBER_OF = "member_of"
    RESOLVES_TO = "resolves_to"
    PARTICIPATES_IN = "participates_in"
    TEMPORAL_AFTER = "temporal_after"
    REFINES_TO = "refines_to"
    CONTAINS = "contains"
    HAS_ACTOR = "has_actor"
    HAS_OBJECT = "has_object"
    AT_TIME = "at_time"
    AT_LOCATION = "at_location"
    HAS_STATE = "has_state"
    MENTIONS = "mentions"
    SAME_EVENT = "same_event"
    TEMPORAL_BEFORE = "temporal_before"
    STATE_TRANSITION = "state_transition"
    REPLACEMENT = "replacement"
    DIALOGUE_PAIR = "dialogue_pair"
    COREFERENCE = "coreference"
    OWNED_BY = "owned_by"
    HAS_EVIDENCE = "has_evidence"
    HAS_FACT = "has_fact"
    FACT_VALUE = "fact_value"
    IN_SCOPE = "in_scope"
    SHARED_ENTITY = "shared_entity"
    SHARED_VALUE = "shared_value"
    SAME_ACTIVITY = "same_activity"
    SAME_PREFERENCE_DOMAIN = "same_preference_domain"
    STATE_NEXT = "state_next"
    COLLECTION_CO_MEMBER = "collection_co_member"


@dataclass(frozen=True, slots=True)
class Conversation:
    memory_id: str
    dataset: str
    source_id: str
    content_hash: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    memory_id: str
    ordinal: int
    timestamp: str | None
    content_hash: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SourceTurn:
    turn_id: str
    memory_id: str
    session_id: str
    turn_index: int
    speaker: str
    listener: str
    role: str
    timestamp: str | None
    raw_text: str
    content_hash: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class EvidenceMember:
    turn_id: str
    span_start: int
    span_end: int
    support_type: str

    def __post_init__(self) -> None:
        if self.span_start < 0 or self.span_end <= self.span_start:
            raise ValueError("invalid evidence span")


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    evidence_group_id: str
    memory_id: str
    members: tuple[EvidenceMember, ...]
    content_hash: str
    min_time: str | None = None
    max_time: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("evidence group must contain at least one source span")


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    memory_id: str
    session_id: str
    start_turn_index: int
    end_turn_index: int
    turn_ids: tuple[str, ...]
    summary: str
    evidence_group_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.start_turn_index < 0 or self.end_turn_index < self.start_turn_index:
            raise ValueError("invalid scene turn interval")
        if not self.turn_ids or not self.evidence_group_ids:
            raise ValueError("scene requires turns and provenance")


@dataclass(frozen=True, slots=True)
class EntityMention:
    mention_id: str
    memory_id: str
    scene_id: str
    turn_id: str
    surface: str
    normalized: str
    entity_type: str
    span_start: int
    span_end: int
    speaker_key: str | None = None
    candidate_entity_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class CanonicalEntity:
    entity_id: str
    memory_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...]
    mention_ids: tuple[str, ...]
    evidence_group_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class EventSkeleton:
    event_id: str
    memory_id: str
    scene_id: str
    predicate: str
    actor_entity_ids: tuple[str, ...]
    object_entity_ids: tuple[str, ...]
    time_values: tuple[str, ...]
    state_values: tuple[str, ...]
    negative_scope: bool
    summary: str
    evidence_group_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class CollectionScope:
    scope_id: str
    memory_id: str
    scene_id: str
    member_event_ids: tuple[str, ...]
    scope_kind: str
    evidence_group_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class StateHead:
    state_head_id: str
    memory_id: str
    entity_id: str
    state_key: str
    current_value: str
    prior_values: tuple[str, ...]
    evidence_group_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FactMention:
    mention_id: str
    scene_id: str
    turn_id: str
    span_start: int
    span_end: int
    surface: str


@dataclass(frozen=True, slots=True)
class CanonicalValue:
    value_id: str
    memory_id: str
    normalized: str
    value_type: str
    evidence_group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalFact:
    fact_id: str
    memory_id: str
    owner_id: str
    predicate_id: str
    value_id: str
    scope_id: str
    polarity: str
    evidence_group_ids: tuple[str, ...]
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class SemanticExtractionManifest:
    memory_id: str
    scene_count: int
    parsed_scene_count: int
    fallback_scene_count: int
    fact_count: int
    valid_evidence_refs: int
    invalid_evidence_refs: int
    token_usage: Mapping[str, int]
    prompt_hash: str


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    memory_id: str
    mode: str
    seed_node_ids: tuple[str, ...]
    visited_node_ids: tuple[str, ...]
    candidate_turn_ids: tuple[str, ...]
    candidate_session_ids: tuple[str, ...]
    proof: tuple["ProofStep", ...]
    relation_counts: Mapping[str, int]
    first_hit_relations: Mapping[str, str]
    budget_exhausted: bool


@dataclass(frozen=True, slots=True)
class DiagnosticRunManifest:
    run_id: str
    graph_checksum: str
    modes: tuple[str, ...]
    question_count: int
    config_hash: str


@dataclass(frozen=True, slots=True)
class RoutingCard:
    card_id: str
    memory_id: str
    level: int
    parent_card_id: str | None
    child_ids: tuple[str, ...]
    portal_ids: tuple[str, ...]
    summary: str
    entity_keys: tuple[str, ...]
    time_keys: tuple[str, ...]
    evidence_group_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    memory_id: str
    node_type: NodeType
    level: int
    summary: str
    evidence_group_id: str
    evidence_group_ids: tuple[str, ...] = ()
    entity_id: str | None = None
    event_time: str | None = None
    state: str | None = None
    confidence: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("node level cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("node confidence must be in [0, 1]")
        if not self.evidence_group_id:
            raise ValueError("graph node requires provenance")

    @property
    def all_evidence_group_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.evidence_group_id, *self.evidence_group_ids)))


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    memory_id: str
    src_id: str
    relation: RelationType
    dst_id: str
    evidence_group_id: str
    directed: bool
    confidence: float
    source: str
    evidence_group_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("edge confidence must be in [0, 1]")
        if not self.evidence_group_id:
            raise ValueError("graph edge requires provenance")

    @property
    def all_evidence_group_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.evidence_group_id, *self.evidence_group_ids)))


@dataclass(frozen=True, slots=True)
class QueryBudget:
    max_hops: int = 2
    max_iterations: int = 4
    max_visited_nodes: int = 96
    max_visited_edges: int = 192
    max_frontier: int = 32
    max_evidence_turns: int = 16
    max_evidence_tokens: int = 5000
    max_llm_reranks: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0 or (name != "max_llm_reranks" and value == 0):
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ProofStep:
    edge_id: str
    src_id: str
    relation: RelationType
    dst_id: str
    evidence_group_id: str


@dataclass(frozen=True, slots=True)
class CandidateScore:
    turn_id: str
    session_id: str
    exact_score: float
    bm25_score: float
    dense_score: float
    graph_score: float
    role_gain: float
    slot_gain: float
    token_cost: int
    fused_score: float
    source_channels: tuple[str, ...]
    session_score: float = 0.0
    adjacency_score: float = 0.0
    graph_path_ids: tuple[str, ...] = ()
    relation_contributions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceCertificate:
    question_kind: str
    required_slots: tuple[str, ...]
    covered_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    complete: bool
    iterations: int
    negative_scope_required: bool = False


@dataclass(frozen=True, slots=True)
class NavigationResult:
    question_id: str
    memory_id: str
    graph_artifact_id: str
    retrieved_session_ids: tuple[str, ...]
    retrieved_turn_ids: tuple[str, ...]
    proof: tuple[ProofStep, ...]
    visited_nodes: int
    visited_edges: int
    frontier_peak: int
    evidence_tokens: int
    budget_exhausted: bool
    trace: Mapping[str, Any] = field(default_factory=dict)
    seed_node_ids: tuple[str, ...] = ()
    visited_path_node_ids: tuple[str, ...] = ()
    slot_coverage: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    certificate: EvidenceCertificate | None = None
    candidate_scores: tuple[CandidateScore, ...] = ()
    packed_turn_ids: tuple[str, ...] = ()
    dropped_turn_ids: tuple[str, ...] = ()
    stage_latency_ms: Mapping[str, float] = field(default_factory=dict)
    graph_only_candidate_turn_ids: tuple[str, ...] = ()
    relation_trace: tuple[ProofStep, ...] = ()
    first_hit_relations: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class GraphArtifactManifest:
    graph_artifact_id: str
    memory_id: str
    dataset_hash: str
    config_hash: str
    graph_checksum: str
    graph_version: int
    node_count: int
    edge_count: int
    evidence_group_count: int
    model_ids: Mapping[str, str]
    prompt_hashes: Mapping[str, str]
    build_token_usage: Mapping[str, int]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    git_commit: str
    dataset_hash: str
    config_hash: str
    graph_artifact_ids: tuple[str, ...]
    model_ids: Mapping[str, str]
    prompt_hashes: Mapping[str, str]
    random_seed: int
    started_at: str
    hardware: Mapping[str, Any]
    software_versions: Mapping[str, str]
    schema_version: str = SCHEMA_VERSION


def dataclass_dict(value: Any) -> dict[str, Any]:
    return _canonical(asdict(value))


def logical_graph_checksum(
    nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
) -> str:
    node_rows = sorted(canonical_json(dataclass_dict(node)) for node in nodes)
    edge_rows = sorted(canonical_json(dataclass_dict(edge)) for edge in edges)
    payload = "\n".join([*node_rows, "--edges--", *edge_rows]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
