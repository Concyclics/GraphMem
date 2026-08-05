from graphmem_demo.v3.recommendation_resources import (
    recommendation_resource_turn_ids, recommendation_scope_session_ids,
    resource_evidence_text,
)
from graphmem_demo.v3.retrieval import _recommendation_constraints, build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, session: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id=session,
        session_date="2026-01-01",
        turn_index=int(node_id[-1]),
        speaker="participant_1",
        speaker_key="participant 1",
        listener="",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_resource_projection_is_bounded_to_routed_sessions() -> None:
    frame = build_query_frame(
        "I am anxious about navigating Meridian. Do you have helpful tips?"
    )
    pass_turn = _turn("t0", "travel", "I currently use my prepaid transit pass.")
    app_turn = _turn("t1", "travel", "I have downloaded RouteKit to organize the trip.")
    distractor = _turn("t2", "hobby", "I currently own a tournament racket.")
    ids = recommendation_resource_turn_ids(
        frame,
        [distractor, pass_turn, app_turn],
        ["travel"],
        semantic_similarity=lambda _turn: 0.5,
    )
    assert set(ids) == {"t0", "t1"}
    assert "t2" not in ids


def test_routed_resource_can_supply_constraint_without_repeating_query_topic() -> None:
    frame = build_query_frame(
        "I am anxious about navigating Meridian. Do you have helpful tips?"
    )
    resource = _turn("t0", "travel", "I have downloaded RouteKit to organize the trip.")
    rows = _recommendation_constraints(
        frame,
        [("turn", resource, 0.4, "recommendation_resource_provenance")],
        [{"session_id": "travel", "query_coverage": 1.0, "posterior": 0.8}],
    )
    assert [row["node_id"] for row in rows] == ["t0"]
    assert rows[0]["selection_source"] == "recommendation_resource_provenance"


def test_recommendation_scope_requires_maximum_anchor_coverage() -> None:
    frame = build_query_frame(
        "I am anxious about navigating Meridian. Do you have helpful tips?"
    )
    ids = recommendation_scope_session_ids(
        frame,
        [
            {"session_id": "travel", "covered_terms": ["meridian", "navigat"], "query_coverage": 0.6, "posterior": 0.7},
            {"session_id": "hobby", "covered_terms": ["tips"], "query_coverage": 0.7, "posterior": 0.8},
        ],
    )
    assert ids == ["travel"]


def test_resource_evidence_text_preserves_late_resource_sentences() -> None:
    text = (
        "I am planning a guided walk. Can you explain the route using my transit pass? "
        "I have downloaded RouteKit to organize it."
    )
    evidence = resource_evidence_text(text)
    assert "transit pass" in evidence
    assert "RouteKit" in evidence
