from graphmem_demo.v3.alternative_comparison import (
    earliest_alternative_from_sources,
)
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode
from graphmem_demo.v3.temporal_normalize import resolve_evidence_time


def _turn(node_id: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id="s",
        session_date="2026-07-21",
        turn_index=0,
        speaker="participant_1",
        speaker_key="participant 1",
        listener="",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def _operand(
    node_id: str, predicate: str, obj: str, event_time: str, source: str
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=node_id,
        question_id="q",
        subject_key="participant 1",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        event_time=event_time,
        observed_at="2026-07-21",
        source_turn_ids=[source],
        retrieval_text=f"participant 1 | {predicate} | {obj}",
    )


def test_relative_time_normalization_orders_distinct_relation_sources() -> None:
    frame = build_query_frame(
        "Who did I meet first, the artisan selling herbs at the plaza "
        "or the traveler from Borealis?"
    )
    turns = [
        _turn("t1", "I met a herb artisan at the plaza two weeks ago."),
        _turn("t2", "I met a traveler from Borealis last Thursday."),
    ]
    hint = earliest_alternative_from_sources(
        frame,
        [
            _operand("o1", "conversation", "herb artisan at plaza", "two weeks ago", "t1"),
            _operand("o2", "met", "Borealis traveler", "last Thursday", "t2"),
        ],
        turns,
    )
    assert hint is not None
    assert hint["operation"] == "earliest_named_alternative"
    assert hint["value"] == "artisan selling herbs at the plaza"
    assert hint["source_turn_ids"] == ["t1", "t2"]
    assert hint["completion_basis"] == "distinct_entity_relation_time_source_closure"


def test_missing_relation_for_one_named_entity_remains_incomplete() -> None:
    frame = build_query_frame("Who became a mentor first, Rowan or Casey?")
    hint = earliest_alternative_from_sources(
        frame,
        [
            _operand("o1", "visited", "Rowan museum", "2026-01-01", "t1"),
            _operand("o2", "became mentor", "Casey mentor", "2026-02-01", "t2"),
        ],
        [_turn("t1", "Rowan visited a museum."), _turn("t2", "Casey became a mentor.")],
    )
    assert hint is not None
    assert hint["operation"] == "named_alternative_incomplete"
    assert hint["missing_alternatives"] == ["rowan"]


def test_last_weekday_is_anchored_before_observation_date() -> None:
    value, basis = resolve_evidence_time("last Thursday", "2026-07-21 (Tue)")
    assert value is not None
    assert value.date().isoformat() == "2026-07-16"
    assert basis == "anchored_weekday"


def test_lossless_relative_times_override_coarse_session_date() -> None:
    frame = build_query_frame(
        "Which event happened first, receiving the case or losing the charger?"
    )
    case = _operand("o1", "received", "case", "2026-07-21", "t1")
    charger = _operand("o2", "lost", "charger", "2026-07-21", "t2")
    turns = [
        _turn("t1", "I received the case about one month ago."),
        _turn("t2", "I lost the charger two weeks ago."),
    ]
    hint = earliest_alternative_from_sources(frame, [case, charger], turns)
    assert hint is not None
    assert hint["operation"] == "earliest_named_alternative"
    assert hint["value"] == "receiving the case"
    assert hint["proofs"][0]["time_basis"].startswith("lossless_")


def test_indefinite_relative_month_and_yearless_dates_use_source_anchor() -> None:
    value, basis = resolve_evidence_time(
        "I received it about a month ago", "2026-07-21"
    )
    assert value is not None
    assert value.date().isoformat() == "2026-06-21"
    assert basis == "anchored_relative"

    named, named_basis = resolve_evidence_time("on January 15th", "2026-03-28")
    numeric, numeric_basis = resolve_evidence_time("on 2/10", "2026-03-28")
    assert named is not None and named.date().isoformat() == "2026-01-15"
    assert numeric is not None and numeric.date().isoformat() == "2026-02-10"
    assert named_basis == numeric_basis == "anchored_partial_date"

def test_relative_month_resolves_against_natural_language_session_date() -> None:
    value, basis = resolve_evidence_time(
        "last month", "3:56 pm on 6 June, 2023"
    )
    assert value is not None
    assert value.strftime("%B %Y") == "May 2023"
    assert basis == "anchored_relative"
