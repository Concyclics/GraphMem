from __future__ import annotations

import json
from dataclasses import replace

import pytest

from graphmem.config import CacheIdentity, GraphMemV5Config, config_from_dict, config_hash
from graphmem.domain import (
    EvidenceGroup,
    EvidenceMember,
    GraphEdge,
    GraphNode,
    NodeType,
    QueryBudget,
    RelationType,
    logical_graph_checksum,
    stable_id,
)


def test_stable_id_is_order_stable_for_mapping_payloads() -> None:
    assert stable_id("node", {"b": 2, "a": 1}) == stable_id(
        "node", {"a": 1, "b": 2}
    )
    assert stable_id("node", "a") != stable_id("edge", "a")


def test_config_hash_is_stable_and_sensitive() -> None:
    first = GraphMemV5Config()
    round_trip = config_from_dict(json.loads(json.dumps({
        "schema_version": first.schema_version,
        "profile": first.profile,
        "random_seed": first.random_seed,
    })))
    assert config_hash(first) == config_hash(round_trip)
    assert config_hash(first) != config_hash(replace(first, random_seed=43))


def test_cache_identity_requires_every_reproducibility_dimension() -> None:
    identity = CacheIdentity(
        dataset_hash="d", model_id="m", prompt_hash="p",
        schema_version="s", config_hash="c", stage="build",
    )
    assert len(identity.key()) == 64
    with pytest.raises(ValueError, match="non-empty"):
        replace(identity, prompt_hash="").key()


def test_thinking_cannot_be_enabled() -> None:
    with pytest.raises(ValueError, match="thinking"):
        config_from_dict({"models": {"thinking_enabled": True}})


def test_query_budget_rejects_unbounded_or_zero_limits() -> None:
    with pytest.raises(ValueError):
        QueryBudget(max_visited_nodes=0)


def test_graph_objects_require_provenance_and_checksum_is_order_independent() -> None:
    member = EvidenceMember("turn:1", 0, 5, "fact")
    group = EvidenceGroup("evidence:1", "m", (member,), "hash")
    a = GraphNode("node:a", "m", NodeType.EVENT_FRAME, 0, "a", group.evidence_group_id)
    b = GraphNode("node:b", "m", NodeType.ENTITY, 0, "b", group.evidence_group_id)
    edge = GraphEdge(
        "edge:1", "m", a.node_id, RelationType.HAS_OBJECT, b.node_id,
        group.evidence_group_id, True, 1.0, "test",
    )
    assert logical_graph_checksum([a, b], [edge]) == logical_graph_checksum(
        [b, a], [edge]
    )
    with pytest.raises(ValueError):
        EvidenceGroup("evidence:empty", "m", (), "hash")
