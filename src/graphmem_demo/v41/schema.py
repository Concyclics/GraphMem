from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


GRAPHMEM_V41_SCHEMA = "graphmem_v4_1_query"
V41_POLICY_VERSION = "4.1.88"


@dataclass(frozen=True)
class QueryPolicyV41:
    normal_context_target: int = 8400
    complex_context_target: int = 9200
    planner_prompt_max: int = 700
    planner_output_max: int = 256
    answer_output_max: int = 512
    query_target: int = 10_000
    query_hard_limit: int = 13_000
    answer_prompt_reserve: int = 4500
    anchor_limit: int = 36
    initial_session_quota: int = 4
    gap_session_quota: int = 8
    dialogue_closure_limit: int = 8
    relation_limit: int = 4
    collection_relation_limit: int = 8
    expansion_rounds: int = 2
    expansion_depth: int = 2
    policy_version: str = V41_POLICY_VERSION


@dataclass
class QueryAugmentationV41:
    domain_hints: list[str] = field(default_factory=list)
    required_roles: list[str] = field(default_factory=list)
    alternative_entities: list[str] = field(default_factory=list)
    event_identity_terms: list[str] = field(default_factory=list)
    scope_terms: list[str] = field(default_factory=list)
    answer_algebra: str = "direct_fact"
    expanded_terms: list[str] = field(default_factory=list)
    planner_required: bool = False


@dataclass
class EvidenceCertificateV41:
    entity_match: bool
    relation_match: bool
    scope_match: bool
    provenance_complete: bool
    lifecycle_complete: bool
    temporal_complete: bool
    dialogue_complete: bool
    present_roles: list[str]
    missing_roles: list[str]
    source_turn_ids: list[str]
    complete: bool
    excluded_near_matches: list[str] = field(default_factory=list)


@dataclass
class PlannerResultV41:
    alternative_entities: list[str] = field(default_factory=list)
    event_aliases: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    temporal_constraints: list[str] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    selected_source_ids: list[str] = field(default_factory=list)
    member_candidates: list[dict[str, str]] = field(default_factory=list)
    slot_candidates: list[dict[str, str]] = field(default_factory=list)
    inference_candidates: list[str] = field(default_factory=list)
    valid: bool = False
    error: str | None = None


@dataclass
class SidecarDocumentV41:
    node_id: str
    node_type: str
    session_ids: list[str]
    source_turn_ids: list[str]
    text: str
    fields: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class QuerySidecarV41:
    index_hash: str
    policy_version: str
    documents: dict[str, SidecarDocumentV41]
    inverted: dict[str, dict[str, list[str]]]
    adjacency: dict[str, dict[str, list[str]]]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = GRAPHMEM_V41_SCHEMA
