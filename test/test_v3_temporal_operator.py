from __future__ import annotations

from graphmem_demo.v3.answer_hints import (
    before_after_relation_hint, calendar_window_hint,
)
from graphmem_demo.v3.operators import duration_hint
from graphmem_demo.v3.retrieval import (
    _evidence_time,
    _node_text,
    _query_overlap,
    _tokens,
    build_query_frame,
)
from graphmem_demo.v3.catalog_schema import EventFrameV3
from graphmem_demo.v3.schema import TurnNode


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


def test_duration_hint_selects_two_query_covering_dated_endpoints() -> None:
    frame = build_query_frame(
        "How many days passed between the modern art museum visit and the ancient exhibit?"
    )
    left = _turn("left", "2026-01-02", "I just visited the modern art museum.")
    right = _turn("right", "2026-01-09", "I attended the ancient exhibit today.")
    distractor = _turn("noise", "2026-01-08", "I visited a coffee shop.")

    hint = duration_hint(
        frame,
        [
            ("turn", left, 1.0, "test"),
            ("turn", right, 1.0, "test"),
            ("turn", distractor, 1.0, "test"),
        ],
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
        query_overlap=_query_overlap,
    )

    assert hint is not None
    assert hint["elapsed_days"] == 7
    assert hint["inclusive_days"] == 8
    assert {hint["left"]["date"], hint["right"]["date"]} == {
        "2026-01-02",
        "2026-01-09",
    }


def test_calendar_window_binds_first_weekend_inside_explicit_month() -> None:
    frame = build_query_frame("Where did Alex go during the first weekend of August 2023?")
    target = _turn(
        "target", "11 am on 4 August, 2023",
        "Alex plans to go camping during the coming weekend.",
    )
    later = _turn(
        "later", "11 am on 19 August, 2023",
        "Alex plans to visit a nature reserve this weekend.",
    )
    hint = calendar_window_hint(
        frame,
        [("turn", later, 2.0, "relation_focus"), ("turn", target, 0.2, "protected_direct")],
        query_overlap=_query_overlap,
    )
    assert hint is not None
    assert hint["event_window_start"] == "2023-08-05"
    assert hint["event_window_end"] == "2023-08-06"
    assert hint["source_turn_ids"] == ["target"]


def test_before_after_relation_binds_anchor_and_excludes_it() -> None:
    before_frame = build_query_frame("Which city was Alex in before traveling to Harbor City?")
    turns = [
        _turn("old", "4:21 pm on 16 July, 2023", "Alex was excited about a game in Cedar City."),
        _turn("anchor", "1:08 pm on 11 August, 2023", "Alex traveled to Harbor City."),
        _turn("noise", "10 August, 2023", "Alex repaired a pump."),
    ]
    frames = [
        EventFrameV3(
            "f-old", "q", "Cedar destination", "cedar destination",
            participant_keys=["alex"], status="complete",
            observed_at="4:21 pm on 16 July, 2023", source_turn_ids=["old"],
            semantic_type_keys=["travel", "future event"],
            retrieval_text="Cedar destination travel Alex",
        ),
        EventFrameV3(
            "f-anchor", "q", "travel to Harbor City", "travel to harbor city",
            participant_keys=["alex"], status="complete",
            observed_at="1:08 pm on 11 August, 2023", source_turn_ids=["anchor"],
            semantic_type_keys=["travel", "city"],
            retrieval_text="travel to Harbor City travel city Alex",
        ),
        EventFrameV3(
            "f-noise", "q", "pump repair", "pump repair",
            participant_keys=["alex"], status="complete",
            observed_at="2023-08-10", source_turn_ids=["noise"],
            semantic_type_keys=["maintenance"],
            retrieval_text="Alex repaired pump maintenance",
        ),
    ]
    kept = [
        *( ("turn", turn, 1.0, "test") for turn in turns ),
        *( ("event_frame", frame, 1.0, "test") for frame in frames ),
    ]
    hint = before_after_relation_hint(
        before_frame, kept, query_overlap=_query_overlap,
    )
    assert hint is not None and hint["complete"] is True
    assert hint["anchor_event"]["node_id"] == "f-anchor"
    assert hint["nearest_qualifying_event"]["node_id"] == "f-old"
    assert hint["source_turn_ids"] == ["anchor", "old"]
    assert hint["distance_days"] == 26


def test_before_after_relation_requires_requested_slot_and_shared_participant() -> None:
    frame = build_query_frame("Which city was Alex in after traveling to Harbor City?")
    anchor = EventFrameV3(
        "anchor-frame", "q", "travel to Harbor City", "travel to harbor city",
        participant_keys=["alex"], observed_at="2023-08-11",
        semantic_type_keys=["travel", "city"], retrieval_text="Alex Harbor City travel city",
    )
    wrong_person = EventFrameV3(
        "wrong-person", "q", "Lake City trip", "lake city trip",
        participant_keys=["blair"], observed_at="2023-08-12",
        semantic_type_keys=["travel", "city"], retrieval_text="Blair Lake City travel city",
    )
    wrong_slot = EventFrameV3(
        "wrong-slot", "q", "pump repair", "pump repair",
        participant_keys=["alex"], observed_at="2023-08-12",
        semantic_type_keys=["maintenance"], retrieval_text="Alex pump repair",
    )
    target = EventFrameV3(
        "target-frame", "q", "Maple City visit", "maple city visit",
        participant_keys=["alex"], observed_at="2023-08-13",
        semantic_type_keys=["travel", "city"], retrieval_text="Alex Maple City travel city",
    )
    hint = before_after_relation_hint(
        frame,
        [("event_frame", value, 1.0, "test") for value in (
            anchor, wrong_person, wrong_slot, target,
        )],
        query_overlap=_query_overlap,
    )
    assert hint is not None
    assert hint["nearest_qualifying_event"]["node_id"] == "target-frame"



def test_before_today_is_a_temporal_window_not_a_binary_event_relation() -> None:
    frame = build_query_frame(
        "What is the order of airlines I flew with from earliest to latest before today?"
    )
    frames = [
        EventFrameV3(
            "first", "q", "United flight", "united flight",
            participant_keys=["alex"], observed_at="2023-01-01",
            semantic_type_keys=["travel", "airline"],
            retrieval_text="Alex flew United Airlines",
        ),
        EventFrameV3(
            "second", "q", "Delta flight today", "delta flight today",
            participant_keys=["alex"], observed_at="2023-02-01",
            semantic_type_keys=["travel", "airline"],
            retrieval_text="Alex flew Delta Airlines today",
        ),
    ]
    assert before_after_relation_hint(
        frame,
        [("event_frame", value, 1.0, "test") for value in frames],
        query_overlap=_query_overlap,
    ) is None
