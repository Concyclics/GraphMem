from __future__ import annotations

from graphmem_demo.v3.retrieval import (
    _relation_prior,
    _scope_posteriors,
    build_query_frame,
)
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, session_id: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id=session_id,
        session_date="2026-01-01",
        turn_index=0,
        speaker="A",
        speaker_key="a",
        listener="B",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_scope_posterior_rewards_joint_query_coverage_without_hard_filtering() -> None:
    frame = build_query_frame("Which camera did Alice buy in Berlin?")
    nodes = {
        "target": _turn("target", "target-session", "Alice bought a Leica camera in Berlin."),
        "partial-a": _turn("partial-a", "other-session", "Alice visited Berlin."),
        "partial-b": _turn("partial-b", "third-session", "Bob bought a camera."),
    }
    channels = {
        "dense": ["partial-a", "target", "partial-b"],
        "bm25": ["target", "partial-a", "partial-b"],
        "exact": ["target", "partial-b", "partial-a"],
    }
    scores = {"target": 0.04, "partial-a": 0.05, "partial-b": 0.03}

    rows = _scope_posteriors(frame, nodes, channels, scores)

    assert rows[0]["session_id"] == "target-session"
    assert rows[0]["query_coverage"] > rows[1]["query_coverage"]
    assert {row["session_id"] for row in rows} == {
        "target-session",
        "other-session",
        "third-session",
    }


def test_relation_prior_is_operation_conditioned_and_not_an_allowlist() -> None:
    temporal = build_query_frame("When did Alice finish the trip?")
    collection = build_query_frame("How many cameras did Alice buy?")

    assert _relation_prior(temporal, "temporal_scope") > _relation_prior(
        temporal, "quantity_collection"
    )
    assert _relation_prior(collection, "quantity_collection") > _relation_prior(
        collection, "temporal_scope"
    )
    assert _relation_prior(temporal, "participant") > 0
