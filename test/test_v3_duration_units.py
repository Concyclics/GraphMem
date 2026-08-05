from __future__ import annotations

from graphmem_demo.v3.catalog_duration import duration_from_operands, duration_from_turns
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.operators import duration_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, date: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id=node_id.split(":")[0],
        session_date=date,
        turn_index=0,
        speaker="participant_1",
        speaker_key="participant 1",
        listener="participant_2",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def _operand(
    node_id: str, predicate: str, value: str, date: str
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=node_id,
        question_id="q",
        subject_key="participant 1",
        predicate_key=predicate,
        object_key=value.casefold(),
        object_text=value,
        polarity="positive",
        modality="asserted",
        observed_at=date,
        event_time=date,
        source_turn_ids=[f"{node_id}:turn"],
        session_ids=[node_id],
        confidence=0.9,
        retrieval_text=f"{predicate} {value}",
    )


def test_local_duration_returns_requested_week_unit() -> None:
    frame = build_query_frame(
        "How many weeks passed between starting the course and finishing it?"
    )
    kept = [
        (
            "turn",
            _turn("s1:turn", "2023-01-01", "I started the course today."),
            1.0,
            "test",
        ),
        (
            "turn",
            _turn("s2:turn", "2023-01-22", "I finished the course today."),
            1.0,
            "test",
        ),
    ]
    hint = duration_hint(
        frame,
        kept,
        tokenize=lambda text: text.casefold().replace("?", "").split(),
        node_text=lambda node: node.retrieval_text,
        evidence_time=lambda node: node.session_date,
        query_overlap=lambda query, text: sum(
            term in text.casefold() for term in query.content_terms
        )
        / max(1, len(query.content_terms)),
    )
    assert hint is not None
    assert hint["elapsed_days"] == 21
    assert hint["value"] == 3
    assert hint["unit"] == "week"


def test_total_time_spent_query_is_duration_not_entity_count() -> None:
    frame = build_query_frame(
        "How many weeks in total did I spend reading one book and listening to another?"
    )
    assert frame.requested_operation == "duration"
    assert frame.answer_form == "duration"


def test_catalog_duration_returns_requested_week_unit() -> None:
    frame = build_query_frame(
        "How many weeks passed between buying the camera and receiving it?"
    )
    operands = [
        _operand("o1", "bought", "camera", "2023-01-01"),
        _operand("o2", "received", "camera", "2023-01-22"),
    ]
    hint = duration_from_operands(
        frame,
        operands,
        query_overlap=lambda _frame, _text: 1.0,
    )
    assert hint is not None
    assert hint["elapsed_days"] == 21
    assert hint["value"] == 3
    assert hint["unit"] == "week"


def test_lossless_between_endpoints_survive_missing_atomic_extraction() -> None:
    frame = build_query_frame(
        "How many months passed between the completion of my undergraduate "
        "degree and the submission of my master's thesis?"
    )
    hint = duration_from_turns(
        frame,
        [
            _turn(
                "s1:turn", "2022-11-17",
                "I just completed my undergraduate degree in computer science.",
            ),
            _turn(
                "s2:turn", "2023-05-15",
                "I submitted my master's thesis today.",
            ),
        ],
        query_overlap=lambda _frame, _text: 1.0,
    )
    assert hint is not None
    assert hint["value"] == 6
    assert hint["unit"] == "month"
    assert hint["source_turn_ids"] == ["s1:turn", "s2:turn"]


def test_since_consecutive_event_sequence_uses_sequence_endpoint() -> None:
    frame = build_query_frame(
        "How many months have passed since I participated in two charity events "
        "in a row, on consecutive days?"
    )
    operands = [
        _operand("o1", "attended", "charity bike ride", "2023-02-14"),
        _operand("o2", "volunteered", "charity book drive", "2023-02-15"),
        _operand("o3", "attended", "charity walk", "2023-03-19"),
    ]
    kept = [("operand", item, 1.0, "test") for item in operands]
    hint = duration_hint(
        frame, kept,
        tokenize=lambda text: text.casefold().replace("?", "").split(),
        node_text=lambda node: node.retrieval_text,
        evidence_time=lambda node: node.observed_at,
        query_overlap=lambda _frame, _text: 1.0,
        question_date="2023/04/18 (Tue) 03:31",
    )
    assert hint is not None
    assert hint["operation"] == "duration_since_consecutive_event_sequence"
    assert hint["sequence_dates"] == ["2023-02-14", "2023-02-15"]
    assert hint["value"] == 2
    assert hint["unit"] == "month"
    assert duration_from_operands(
        frame, operands, query_overlap=lambda _frame, _text: 1.0,
    ) is None


def test_days_ago_uses_question_date_as_reference_endpoint() -> None:
    frame = build_query_frame(
        "How many days ago did I attend a networking event?"
    )
    kept = [
        (
            "turn",
            _turn(
                "event:turn",
                "2022-03-09",
                "I attended a networking event today.",
            ),
            1.0,
            "test",
        )
    ]
    hint = duration_hint(
        frame,
        kept,
        tokenize=lambda text: text.casefold().replace("?", "").split(),
        node_text=lambda node: node.retrieval_text,
        evidence_time=lambda node: node.session_date,
        query_overlap=lambda _frame, _text: 1.0,
        question_date="2022-04-04",
    )
    assert hint is not None
    assert hint["operation"] == "duration_since_bound_event"
    assert hint["value"] == 26
