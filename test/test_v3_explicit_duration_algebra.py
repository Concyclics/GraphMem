from __future__ import annotations

from graphmem_demo.v3.operators import duration_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id=node_id.split(":")[0],
        session_date="2023-05-28",
        turn_index=0,
        speaker="participant_1",
        speaker_key="participant 1",
        listener="participant_2",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def _hint(question: str, texts: list[str]):
    frame = build_query_frame(question)
    kept = [
        ("turn", _turn(f"s{index}:turn", text), 1.0, "test")
        for index, text in enumerate(texts)
    ]
    return duration_hint(
        frame,
        kept,
        tokenize=lambda text: text.casefold().replace("'", "").split(),
        node_text=lambda node: node.retrieval_text,
        evidence_time=lambda node: node.session_date,
        query_overlap=lambda query, text: sum(
            term in text.casefold() for term in query.content_terms
        ) / max(1, len(query.content_terms)),
    )


def test_simple_explicit_duration_beats_unrelated_calendar_dates() -> None:
    result = _hint(
        "How long have I been collecting vintage cameras?",
        ["I've been collecting vintage cameras for three months now."],
    )
    assert result is not None
    assert result["operation"] == "explicit_relative_duration"
    assert result["value"] == 3
    assert result["unit"] == "month"


def test_relative_offsets_resolve_state_duration_at_event_time() -> None:
    result = _hint(
        "How long had I been taking guitar lessons when I bought the amp?",
        [
            "I've been taking guitar lessons for six weeks.",
            "I bought the new guitar amp two weeks ago.",
        ],
    )
    assert result is not None
    assert result["value"] == 4
    assert result["unit"] == "week"


def test_two_relative_event_offsets_produce_elapsed_membership() -> None:
    result = _hint(
        "How long had I been a member when I attended the meetup?",
        [
            "I became a member of the group three weeks ago.",
            "I attended the meetup last week.",
        ],
    )
    assert result is not None
    assert result["value"] == 2
    assert result["unit"] == "week"
