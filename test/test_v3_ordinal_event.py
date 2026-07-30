from types import SimpleNamespace

from graphmem_demo.v3.ordinal_event import ordinal_event_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.catalog_schema import OperandRecordV3


def _operand(
    index: int,
    session: str,
    observed: str,
    predicate: str,
    value: str,
    source: str,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=f"o{index}",
        question_id="q",
        subject_key="nate",
        predicate_key=predicate,
        object_key=value.casefold(),
        object_text=value,
        polarity="positive",
        modality="asserted",
        observed_at=observed,
        event_time="last week",
        source_turn_ids=[source],
        session_ids=[session],
        confidence=0.9,
        retrieval_text=f"Nate {predicate} {value} last week",
    )


def test_ordinal_event_binds_specific_attribute_inside_selected_occurrence() -> None:
    frame = build_query_frame(
        "What game was the second tournament that Nate won based on?"
    )
    operands = [
        _operand(0, "s1", "2022-01-01", "won", "first tournament", "t1"),
        _operand(1, "s2", "2022-02-01", "won", "tournament", "t2"),
        _operand(
            2, "s2", "2022-02-01", "won",
            "Street Fighter tournament", "t2",
        ),
        _operand(3, "s3", "2022-03-01", "won", "third tournament", "t3"),
    ]
    turns = [
        SimpleNamespace(
            node_id="t2",
            text=(
                "I usually play another game, but won the Street Fighter "
                "tournament this time."
            ),
        )
    ]
    hint = ordinal_event_hint(frame, operands, turns)
    assert hint is not None
    assert hint["operation"] == "ordinal_event_attribute"
    assert hint["ordinal"] == 2
    assert hint["value"] == "Street Fighter tournament"
    assert hint["source_turn_ids"] == ["t2"]


def test_ordinal_event_date_preserves_relative_expression() -> None:
    frame = build_query_frame("When did Nate win his third tournament?")
    operands = [
        _operand(0, "s1", "2022-01-01", "won", "tournament", "t1"),
        _operand(1, "s2", "2022-02-01", "won", "tournament", "t2"),
        _operand(2, "s3", "2022-03-01", "won", "tournament", "t3"),
    ]
    turns = [
        SimpleNamespace(node_id="t3", text="I won another tournament last week.")
    ]
    hint = ordinal_event_hint(frame, operands, turns)
    assert hint is not None
    assert hint["value"] == "last week"
    assert hint["session_id"] == "s3"
