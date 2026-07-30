from graphmem_demo.v3.retrieval import (
    _recommendation_constraints,
    build_query_frame,
)
from graphmem_demo.v3.schema import TurnNode


def _turn(index: int, text: str) -> TurnNode:
    return TurnNode(
        node_id=f"q:s{index}:turn:0",
        question_id="q",
        session_id=f"s{index}",
        session_date="2026-01-01",
        turn_index=0,
        speaker="person",
        speaker_key="person",
        listener="",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_recommendation_constraints_require_non_generic_target_overlap() -> None:
    frame = build_query_frame(
        "Can you recommend resources where I can learn video editing?"
    )
    relevant = _turn(
        0,
        "The person prefers advanced video editing resources for a specific editor.",
    )
    irrelevant = _turn(
        1,
        "The person asked for recommended online resources to learn aquarium care.",
    )
    rows = _recommendation_constraints(
        frame,
        [
            ("turn", irrelevant, 2.0, "protected_direct"),
            ("turn", relevant, 1.0, "protected_direct"),
        ],
    )
    assert [row["node_id"] for row in rows] == [relevant.node_id]


def test_recommendation_constraints_stay_inside_highest_coverage_scope() -> None:
    frame = build_query_frame(
        "Can you recommend accessories that complement my current hiking setup?"
    )
    primary = _turn(
        0,
        "I currently use a lightweight hiking pack and want compatible hiking accessories.",
    )
    distractor = _turn(
        1,
        "I want hiking accessories mainly as decorative gifts for someone else.",
    )
    rows = _recommendation_constraints(
        frame,
        [
            ("turn", distractor, 2.0, "protected_graph_rescue"),
            ("turn", primary, 1.0, "scope_local_turn_primary"),
        ],
        [
            {"session_id": "s0", "query_coverage": 0.75, "posterior": 0.30},
            {"session_id": "s1", "query_coverage": 0.50, "posterior": 0.40},
        ],
    )
    assert [row["node_id"] for row in rows] == [primary.node_id]


def test_self_recommendation_constraints_exclude_assistant_suggestions() -> None:
    frame = build_query_frame(
        "Can you recommend resources for my current video editing setup?"
    )
    user_state = _turn(0, "I currently use a desktop editor for video editing.")
    assistant = _turn(0, "I recommend a general video editing course.")
    assistant.node_id = "q:s0:turn:1"
    assistant.turn_index = 1
    assistant.transport_role = "assistant"
    rows = _recommendation_constraints(
        frame,
        [
            ("turn", assistant, 2.0, "protected_direct"),
            ("turn", user_state, 1.0, "scope_local_turn_primary"),
        ],
        [{"session_id": "s0", "query_coverage": 1.0, "posterior": 0.5}],
    )
    assert [row["node_id"] for row in rows] == [user_state.node_id]
