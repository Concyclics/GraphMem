from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
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
    COLLECTION_MANIFEST = "collection_manifest"


class RelationType(StrEnum):
    PORTAL = "portal"
    SCENE_CONTAINS = "scene_contains"
    MEMBER_OF = "member_of"
    RESOLVES_TO = "resolves_to"
    PARTICIPATES_IN = "participates_in"
    TEMPORAL_AFTER = "temporal_after"
    REFINES_TO = "refines_to"
    COARSE_RELATED = "coarse_related"
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
    SAME_ENTITY_STATE = "same_entity_state"
    TEMPORAL_CONTINUATION = "temporal_continuation"
    CAUSAL = "causal"
    CONTRADICTION_UPDATE = "contradiction_update"
    #: Two scenes in different sessions that share several low-document-frequency
    #: terms.  Derived from the raw text rather than from anything the extractor
    #: wrote, which is the point: every extractor-supplied join key degenerates
    #: (92.1% of predicates occur once), and the measured result is a graph whose
    #: edges are 100% single-session.  See scripts/build_shared_referent_edges.py.
    SHARED_REFERENT = "shared_referent"


class QueryOperator(StrEnum):
    LOOKUP = "lookup"
    UNION_DISTINCT = "union_distinct"
    INTERSECTION_DISTINCT = "intersection_distinct"
    GROUP_BY_OWNER = "group_by_owner"
    COUNT_DISTINCT = "count_distinct"
    EXISTS_ALL = "exists_all"
    ARGMIN_TIME = "argmin_time"
    ARGMAX_TIME = "argmax_time"
    DATE_DIFFERENCE = "date_difference"
    LATEST_STATE = "latest_state"
    ORDINAL = "ordinal"


class CertificateStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE_BINDING = "incomplete_binding"
    INCOMPLETE_OPERAND = "incomplete_operand"
    INCOMPLETE_COLLECTION = "incomplete_collection"
    INCOMPLETE_SCOPE = "incomplete_scope"
    INCOMPLETE_PROVENANCE = "incomplete_provenance"
    PACK_INCOMPLETE = "pack_incomplete"
    NO_PROGRESS = "no_progress"
    BUDGET_EXHAUSTED = "budget_exhausted"


class CertificatePhase(StrEnum):
    """Ordered stages a V5.6 certificate must clear before it may claim closure.

    V5.5 collapsed all of these into "the operand has at least one binding",
    which is why a certificate could report complete while the packer had
    already dropped the evidence that proves it.
    """

    BINDING = "binding_complete"
    SCOPE = "scope_or_collection_saturated"
    PROVENANCE = "provenance_hydrated"
    POST_PACK = "post_pack_complete"


CERTIFICATE_PHASE_ORDER: tuple[CertificatePhase, ...] = (
    CertificatePhase.BINDING,
    CertificatePhase.SCOPE,
    CertificatePhase.PROVENANCE,
    CertificatePhase.POST_PACK,
)


class StopReason(StrEnum):
    CERTIFICATE = "certificate"
    NO_PROGRESS = "no_progress"
    NODE_CAP = "node_cap"
    EDGE_CAP = "edge_cap"
    HOP_CAP = "hop_cap"
    FRONTIER_TRUNCATED = "frontier_truncated"
    PACK_TURN_CAP = "pack_turn_cap"
    PACK_TOKEN_CAP = "pack_token_cap"
    PACK_INCOMPLETE = "pack_incomplete"
    CANDIDATES_EXHAUSTED = "candidates_exhausted"


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
class TemporalInterval:
    raw_text: str
    start: str | None
    end: str | None
    precision: str
    kind: str
    anchor_turn_id: str
    confidence: float

    def __post_init__(self) -> None:
        if self.kind not in {"absolute", "relative", "observed", "unresolved"}:
            raise ValueError("invalid temporal interval kind")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("temporal interval confidence must be in [0, 1]")


PRECISION_RANK: Mapping[str, int] = {
    "second": 0, "minute": 1, "day": 2, "week": 3, "month": 4, "year": 5, "unknown": 6,
}
TEMPORAL_KIND_RANK: Mapping[str, int] = {
    "absolute": 0, "relative": 1, "observed": 2, "unresolved": 3,
}


@dataclass(frozen=True, slots=True, order=False)
class TemporalKey:
    """A comparable projection of a ``TemporalInterval`` for relation algebra.

    V5.5 stored the whole interval mapping on a graph node and then compared
    ``str(dict)``.  Canonical JSON sorts keys, so that string starts with
    ``anchor_turn_id`` and every temporal ordering degenerated into a sort over
    turn-id hashes.  Ordering must be driven by the normalized endpoints.
    """

    start: str | None = None
    end: str | None = None
    precision: str = "unknown"
    kind: str = "unresolved"
    confidence: float = 0.0
    raw_text: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.start)

    @property
    def sort_key(self) -> tuple[int, str, str, int, int, float]:
        """Total order: resolved intervals first, then by start/end, then precision."""
        return (
            0 if self.resolved else 1,
            self.start or "",
            self.end or "",
            PRECISION_RANK.get(self.precision, len(PRECISION_RANK)),
            TEMPORAL_KIND_RANK.get(self.kind, len(TEMPORAL_KIND_RANK)),
            -self.confidence,
        )

    def days_between(self, other: "TemporalKey") -> int | None:
        """Whole days from ``other``'s start to this key's start.

        Returns ``None`` unless both endpoints are resolved to at least day
        precision: a date difference computed from a year-precision endpoint
        would be off by up to a year while looking exact.
        """
        if not (self.resolved and other.resolved):
            return None
        try:
            left = date.fromisoformat(str(self.start)[:10])
            right = date.fromisoformat(str(other.start)[:10])
        except ValueError:
            return None
        return (left - right).days

    def overlaps(self, other: "TemporalKey") -> bool:
        if not (self.resolved and other.resolved):
            return False
        left_end = self.end or self.start
        right_end = other.end or other.start
        return not (left_end < (other.start or "") or right_end < (self.start or ""))

    @classmethod
    def from_attribute(cls, value: Any) -> "TemporalKey | None":
        """Build a key from a ``dataclass_dict(TemporalInterval)`` attribute.

        Returns ``None`` for a genuinely absent interval so callers can tell
        "no time" apart from "time that would not normalize".
        """
        if isinstance(value, TemporalKey):
            return value
        if isinstance(value, TemporalInterval):
            value = dataclass_dict(value)
        if not isinstance(value, Mapping):
            return None
        start = value.get("start") or None
        end = value.get("end") or None
        if start is None and end is None and not value.get("raw_text"):
            return None
        try:
            confidence = float(value.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            start=str(start) if start else None,
            end=str(end) if end else None,
            precision=str(value.get("precision", "unknown") or "unknown"),
            kind=str(value.get("kind", "unresolved") or "unresolved"),
            confidence=confidence,
            raw_text=str(value.get("raw_text", "") or ""),
        )


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
class GraphChecksumState:
    """Incrementally composable digest state for a logical graph.

    Each canonical row contributes a domain-separated SHA-256 value.  XOR and
    modular sum make insert/delete/upsert O(1) per touched row; counts prevent
    cancellation from making graphs with different cardinality equivalent.
    The published checksum hashes the complete state again, so callers only see
    the usual opaque 256-bit artifact identifier.
    """

    node_xor: int = 0
    node_sum: int = 0
    node_count: int = 0
    edge_xor: int = 0
    edge_sum: int = 0
    edge_count: int = 0

    @property
    def checksum(self) -> str:
        payload = canonical_json({
            "algorithm": "graphmem-multiset-sha256-v1",
            "node_xor": f"{self.node_xor:064x}",
            "node_sum": f"{self.node_sum:064x}",
            "node_count": self.node_count,
            "edge_xor": f"{self.edge_xor:064x}",
            "edge_sum": f"{self.edge_sum:064x}",
            "edge_count": self.edge_count,
        }).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_HASH_MODULUS = 1 << 256


def logical_graph_row_digest(kind: str, value: GraphNode | GraphEdge) -> int:
    if kind not in {"node", "edge"}:
        raise ValueError(f"unsupported graph row kind: {kind!r}")
    payload = (kind + "\0" + canonical_json(dataclass_dict(value))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def logical_graph_checksum_state(
    nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
) -> GraphChecksumState:
    node_xor = node_sum = node_count = 0
    for node in nodes:
        digest = logical_graph_row_digest("node", node)
        node_xor ^= digest
        node_sum = (node_sum + digest) % _HASH_MODULUS
        node_count += 1
    edge_xor = edge_sum = edge_count = 0
    for edge in edges:
        digest = logical_graph_row_digest("edge", edge)
        edge_xor ^= digest
        edge_sum = (edge_sum + digest) % _HASH_MODULUS
        edge_count += 1
    return GraphChecksumState(
        node_xor, node_sum, node_count,
        edge_xor, edge_sum, edge_count,
    )


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
    # V5.6: the reservoir holds ids only, so it can be far wider than the
    # hydrated evidence pack without costing context tokens.  The legacy
    # navigator's widest measured pool is 524 turns, so anything smaller would
    # reintroduce the coverage regression this reservoir exists to remove.
    max_candidate_reservoir: int = 576
    max_query_views_per_operand: int = 6
    max_answer_tokens: int = 10000
    max_answer_tokens_hard: int = 13000
    # The fact reservoir is id-only and wide; only this many facts reach binding.
    max_active_facts: int = 96
    max_active_facts_per_operand: int = 32
    # Seeding and traversal used to share ``max_visited_nodes``: seeding filled
    # it, the scheduler popped those seeds until the same cap was reached, and
    # expanded nothing.  These split the cap.  Both are read through the
    # ``seed_nodes`` / ``traversal_nodes`` properties below, where 0 means
    # "inherit max_visited_nodes" -- so grepping the field names alone makes them
    # look dead when they are not.
    max_seed_nodes: int = 64
    max_traversal_nodes: int = 0

    #: Fields where 0 is meaningful rather than invalid: no reranks, or "inherit
    #: max_visited_nodes" for the budgets split out of it.
    _ZERO_ALLOWED = ("max_llm_reranks", "max_traversal_nodes", "max_seed_nodes")

    @property
    def traversal_nodes(self) -> int:
        """Node cap for graph traversal, independent of how many seeds there are."""
        return self.max_traversal_nodes or self.max_visited_nodes

    @property
    def seed_nodes(self) -> int:
        """Node cap for seeding, so a full seed list cannot starve traversal."""
        return self.max_seed_nodes or self.max_visited_nodes

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0 or (name not in self._ZERO_ALLOWED and value == 0):
                raise ValueError(f"{name} must be positive")
        if self.max_answer_tokens_hard < self.max_answer_tokens:
            raise ValueError("max_answer_tokens_hard cannot be below max_answer_tokens")
        # No guard on seed_nodes >= traversal_nodes: it was measured not to
        # matter.  Seeding returns 15-64 nodes in practice, so the shared cap
        # rarely bound, and widening the traversal budget to 160 actually
        # *reduced* edges walked (37.7 vs 44.0 per query) because traversal is
        # seed-bound, not budget-bound.  The split is kept for correctness --
        # the two stages should not compete -- not as a performance fix.


@dataclass(frozen=True, slots=True)
class ProofStep:
    edge_id: str
    src_id: str
    relation: RelationType
    dst_id: str
    evidence_group_id: str
    # ``GraphReadView.neighbors`` traverses inverse adjacency by default.  Without
    # recording the direction, a proof rendered from src->dst misreports how the
    # node was actually reached.
    inverse: bool = False

    @property
    def from_id(self) -> str:
        return self.dst_id if self.inverse else self.src_id

    @property
    def to_id(self) -> str:
        return self.src_id if self.inverse else self.dst_id


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
    operand_ids: tuple[str, ...] = ()
    binding_score: float = 0.0
    relation_path_score: float = 0.0
    obligation_gain: float = 0.0
    provenance_novelty: float = 0.0
    mandatory: bool = False
    proof_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OperandSpec:
    operand_id: str
    owner_aliases: tuple[str, ...] = ()
    predicate_candidates: tuple[str, ...] = ()
    scope_candidates: tuple[str, ...] = ()
    value_type: str | None = None
    temporal_constraint: str | None = None
    polarity: str | None = None
    modalities: tuple[str, ...] = ()
    multiplicity: str = "one"
    distinct_by: str = "value"
    required_provenance: bool = True


@dataclass(frozen=True, slots=True)
class ProofObligation:
    obligation_id: str
    operand_id: str | None
    kind: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class FactBinding:
    binding_id: str
    operand_id: str
    fact_node_id: str
    owner_id: str | None
    predicate: str
    scope: str
    value_key: str
    event_instance_id: str | None
    time_interval: TemporalKey | None
    evidence_group_ids: tuple[str, ...]
    relation_path: tuple[str, ...] = ()
    confidence: float = 0.0
    value: str = ""
    value_type: str = "text"
    polarity: str = "positive"
    modality: str = "asserted"
    session_id: str = ""
    turn_index: int = -1
    collection_key: str = ""
    node_type: str = "canonical_fact"


@dataclass(frozen=True, slots=True)
class AnswerMember:
    """One distinct element of an algebra result, with the witnesses proving it.

    ``member_key`` is whatever the operator distinguishes by: a value key, an
    event-instance id, a normalized date, or an ``owner:value`` pair.
    """

    member_key: str
    value: str
    value_key: str
    owner_id: str | None = None
    value_type: str = "text"
    witness_binding_ids: tuple[str, ...] = ()
    operand_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalEndpoint:
    role: str
    key: TemporalKey
    binding_id: str


@dataclass(frozen=True, slots=True)
class StateResult:
    owner_id: str | None
    predicate: str
    current_value: str
    current_binding_id: str
    prior_binding_ids: tuple[str, ...] = ()
    superseded: bool = False
    contradiction_status: str = "none"
    interval_uncertainty: str = "unknown"


@dataclass(frozen=True, slots=True)
class AlgebraResult:
    """Structured output of the relation algebra.

    V5.5 returned only ``output_binding_ids``, so the packer could not tell an
    answer member from a mere candidate and made every binding mandatory.
    """

    operator: QueryOperator
    bindings: tuple[FactBinding, ...]
    output_binding_ids: tuple[str, ...]
    members: tuple[AnswerMember, ...] = ()
    count: int | None = None
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    temporal_endpoints: tuple[TemporalEndpoint, ...] = ()
    state_result: StateResult | None = None
    witness_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    complete_operands: tuple[str, ...] = ()
    scope_complete: bool = False
    answer_kind: str = "lookup"
    degradations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    unit_id: str
    obligation_ids: tuple[str, ...]
    binding_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    relation_path_ids: tuple[str, ...]
    token_cost: int
    mandatory: bool = False
    operand_ids: tuple[str, ...] = ()
    member_key: str = ""
    spans: tuple[EvidenceMember, ...] = ()
    # Units are atomic by default: a shared value proved by two speakers is
    # worthless if only one half survives the budget.
    atomic: bool = True
    rank: int = 0


@dataclass(frozen=True, slots=True)
class ProofBundle:
    """The packer's input contract: ordered, atomic units plus operand quotas."""

    units: tuple[EvidenceUnit, ...]
    mandatory_unit_ids: tuple[str, ...] = ()
    operand_quota: Mapping[str, int] = field(default_factory=dict)
    answer_kind: str = "lookup"


@dataclass(frozen=True, slots=True)
class EvidenceCertificate:
    question_kind: str
    required_slots: tuple[str, ...]
    covered_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    complete: bool
    iterations: int
    negative_scope_required: bool = False
    status: CertificateStatus = CertificateStatus.INCOMPLETE_OPERAND
    operand_status: Mapping[str, str] = field(default_factory=dict)
    stop_reason: StopReason | None = None
    # V5.6 four-phase closure.  ``complete`` stays the legacy pre-pack flag so
    # existing metrics keep their meaning; ``post_pack_complete`` is the one a
    # report may call an evidence certificate.
    phase_status: Mapping[str, bool] = field(default_factory=dict)
    post_pack_complete: bool = False
    unsatisfied_phase: str | None = None
    dropped_mandatory_unit_ids: tuple[str, ...] = ()

    @property
    def false_complete(self) -> bool:
        """True when a pre-pack certificate claimed closure the pack did not deliver."""
        return bool(self.complete and not self.post_pack_complete)


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
    search_exhaustion: Mapping[str, bool] = field(default_factory=dict)
    pack_exhaustion: Mapping[str, bool] = field(default_factory=dict)
    operand_coverage: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    proof_units: tuple[EvidenceUnit, ...] = ()
    stop_reason: StopReason | None = None
    # Diagnostics only: which facts actually bound, and which the algebra kept.
    # Without these the proof funnel cannot tell "no fact existed" apart from
    # "a fact existed but nothing bound to it".
    reservoir_fact_node_ids: tuple[str, ...] = ()
    reached_fact_node_ids: tuple[str, ...] = ()
    bound_fact_node_ids: tuple[str, ...] = ()
    selected_fact_node_ids: tuple[str, ...] = ()
    # Set only by AST-executing profiles.  The answer stage needs answer members
    # and their witnesses to emit a closed-form count or list; without this the
    # composer has nothing to read and fired on 0 of 200 questions.
    algebra: "AlgebraResult | None" = None
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
    build_diagnostics: Mapping[str, Any] = field(default_factory=dict)
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
    return logical_graph_checksum_state(nodes, edges).checksum
