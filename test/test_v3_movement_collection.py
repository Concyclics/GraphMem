from graphmem_demo.v3.catalog_schema import EventFrameV3, OperandRecordV3
from graphmem_demo.v3.movement_collection import (
    movement_location_collection_hint,
)
from graphmem_demo.v3.retrieval import build_query_frame


def _operand(
    operand_id: str,
    subject: str,
    predicate: str,
    object_text: str,
    *,
    modality: str = "asserted",
    observed_at: str = "4 May, 2023",
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key=subject,
        predicate_key=predicate,
        object_key=object_text.casefold(),
        object_text=object_text,
        modality=modality,
        polarity="positive",
        observed_at=observed_at,
        source_turn_ids=[f"{operand_id}:turn"],
    )


def test_movement_collection_filters_plans_other_subjects_and_media() -> None:
    frame = build_query_frame("Which cities did Alex travel to in 2023?")
    rows = [
        _operand("o1", "alex", "returned from", "Port Indigo"),
        _operand("o2", "alex", "attended", "a workshop in Lake City"),
        _operand("o3", "alex", "will show Morgan", "favorite places in Coaston",
                 modality="planned"),
        _operand("o4", "morgan", "visited", "Hilltown"),
        _operand("o5", "alex", "shared photo", "a street in River City"),
    ]
    hint = movement_location_collection_hint(frame, rows)
    assert hint is not None
    assert hint["values"] == ["Lake City", "Port Indigo"]
    assert hint["complete"] is False
    assert hint["source_turn_ids"] == ["o2:turn", "o1:turn"]


def test_movement_collection_respects_explicit_year() -> None:
    frame = build_query_frame("Which places did Alex visit in 2023?")
    rows = [
        _operand("old", "alex", "visited", "Oldtown",
                 observed_at="4 May, 2022"),
        _operand("new", "alex", "visited", "Newtown",
                 observed_at="4 May, 2023"),
    ]
    hint = movement_location_collection_hint(frame, rows)
    assert hint is not None
    assert hint["values"] == ["Newtown"]


def test_movement_collection_uses_coarse_event_when_operand_is_missing() -> None:
    frame = build_query_frame("Which cities did Alex travel to in 2023?")
    coarse = EventFrameV3(
        frame_id="f1",
        question_id="q",
        label="Alex attended a conference in Harbor City",
        label_key="alex attended conference harbor city",
        participant_keys=["alex"],
        status="asserted",
        event_time="recently",
        observed_at="4 May, 2023",
        session_ids=["s1"],
        source_turn_ids=["s1:t1"],
        retrieval_text="Alex attended a conference in Harbor City",
    )
    future = _operand(
        "future", "alex", "wants to visit", "Dreamland",
        observed_at="4 May, 2023",
    )
    future.event_time = "future"
    hint = movement_location_collection_hint(frame, [future], [coarse])
    assert hint is not None
    assert hint["values"] == ["Harbor City"]
    assert hint["frame_ids"] == ["f1"]
