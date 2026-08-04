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
    ROUTING_CARD = "routing_card"
    EVENT_FRAME = "event_frame"
    ROLE_FRAME = "role_frame"
    ENTITY = "entity"
    TIME_ANCHOR = "time_anchor"
    STATE_VALUE = "state_value"
    SESSION = "session"
    EVIDENCE_GROUP_REF = "evidence_group_ref"


class RelationType(StrEnum):
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
class GraphNode:
    node_id: str
    memory_id: str
    node_type: NodeType
    level: int
    summary: str
    evidence_group_id: str
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
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("edge confidence must be in [0, 1]")
        if not self.evidence_group_id:
            raise ValueError("graph edge requires provenance")


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
