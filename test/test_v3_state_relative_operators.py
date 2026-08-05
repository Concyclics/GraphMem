from __future__ import annotations

from graphmem_demo.v3.retrieval import (
    _evidence_time,
    _node_text,
    _tokens,
    build_query_frame,
)
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.schema import TurnNode
from graphmem_demo.v3.state_temporal_operators import (
    latest_state_hint,
    relative_age_hint,
    relative_time_hint,
)


def _turn(node_id: str, day: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id=node_id,
        session_date=day,
        turn_index=0,
        speaker="A",
        speaker_key="a",
        listener="B",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_latest_state_hint_prefers_newest_attribute_match() -> None:
    frame = build_query_frame("What is my current volleyball record?")
    old = _turn("old", "2026-01-01", "My volleyball record is 3-2.")
    new = _turn("new", "2026-01-08", "My volleyball record is 5-2.")
    hint = latest_state_hint(
        frame,
        [("turn", old, 1.0, "test"), ("turn", new, 1.0, "test")],
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["observed_at"] == "2026-01-08"
    assert "5-2" in hint["evidence"]


def test_latest_state_hint_ignores_invalid_calendar_dates() -> None:
    frame = build_query_frame("What is my current volleyball record?")
    invalid = _turn("invalid", "2026-02-31", "My volleyball record is 9-2.")
    valid = _turn("valid", "2026-02-28", "My volleyball record is 5-2.")
    hint = latest_state_hint(
        frame,
        [("turn", invalid, 2.0, "test"), ("turn", valid, 1.0, "test")],
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["observed_at"] == "2026-02-28"
    assert "5-2" in hint["evidence"]


def test_relative_time_hint_selects_nearest_dated_scope() -> None:
    frame = build_query_frame("Did I visit the museum two months ago?")
    old = _turn("old", "2025-10-11", "I visited a science museum.")
    target = _turn("target", "2026-01-11", "I attended a museum lecture.")
    hint = relative_time_hint(
        frame,
        [("turn", old, 1.0, "test"), ("turn", target, 1.0, "test")],
        "2026-03-11",
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["target_date"] == "2026-01-11"
    assert hint["selected_evidence_date"] == "2026-01-11"
    assert hint["within_tolerance"]



def test_relative_time_accepts_an_indefinite_article_as_one() -> None:
    frame = build_query_frame("What event did I attend a week ago?")
    target = _turn("target", "2026-03-04", "I attended a family ceremony.")
    hint = relative_time_hint(
        frame, [("turn", target, 1.0, "test")], "2026-03-11",
        tokenize=_tokens, node_text=_node_text, evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["target_date"] == "2026-03-04"
    assert hint["supporting_node_ids"] == ["target"]


def test_relative_time_honors_requested_relationship_category() -> None:
    frame = build_query_frame(
        "What life event of one of my relatives did I participate in a week ago?"
    )
    friend = _turn("friend", "2026-03-04", "My friend had a baby today.")
    cousin = _turn(
        "cousin", "2026-03-04",
        "I walked down the aisle at my cousin's wedding.",
    )
    hint = relative_time_hint(
        frame, [("turn", friend, 2.0, "test"), ("turn", cousin, 1.0, "test")],
        "2026-03-11", tokenize=_tokens, node_text=_node_text,
        evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["supporting_node_ids"] == ["cousin"]


def test_relative_age_anchors_today_to_operand_observation_date() -> None:
    frame = build_query_frame("How many days ago did I calibrate the kiln?")
    operand = OperandRecordV3(
        "q:operand:0", "q", "a", "calibrated", "kiln", "kiln",
        event_time="today", observed_at="2026-03-02",
        source_claim_ids=["c0"], source_turn_ids=["t0"],
        retrieval_text="A calibrated kiln today",
    )
    hint = relative_age_hint(
        frame,
        [("operand", operand, 1.0, "test")],
        "2026-03-11",
        tokenize=_tokens,
        node_text=_node_text,
    )
    assert hint is not None
    assert hint["event_date"] == "2026-03-02"
    assert hint["value"] == 9
