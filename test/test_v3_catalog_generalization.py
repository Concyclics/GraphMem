from graphmem_demo.v3.catalog_arithmetic import arithmetic_hint
from graphmem_demo.v3.catalog_duration import duration_from_operands
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame


def _operand(
    operand_id: str,
    predicate: str,
    obj: str,
    *,
    event_time: str,
    quantity: float | None = None,
    unit: str = "",
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="participant 1",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        event_time=event_time,
        observed_at=event_time,
        quantity=quantity,
        unit=unit,
        polarity="positive",
        modality="asserted",
        source_turn_ids=[f"{operand_id}:turn"],
        retrieval_text=f"participant 1 {predicate} {obj} {event_time}",
    )


def _overlap(frame, text: str) -> float:
    query = set(frame.content_terms)
    words = set(text.casefold().replace("-", " ").split())
    return len(query & words) / max(1, len(query))


def test_duration_binds_two_named_endpoints_without_action_words() -> None:
    frame = build_query_frame(
        "How many days passed between the spring festival and Sunday service?"
    )
    hint = duration_from_operands(
        frame,
        [
            _operand("a", "attended", "spring festival", event_time="2023-02-26"),
            _operand("b", "attended", "Sunday service", event_time="2023-03-19"),
            _operand("noise", "visited", "unrelated market", event_time="2023-03-12"),
        ],
        _overlap,
    )
    assert hint is not None
    assert hint["elapsed_days"] == 21
    assert {hint["left"]["operand_id"], hint["right"]["operand_id"]} == {"a", "b"}


def test_duration_binds_it_took_after_endpoints_and_object_date() -> None:
    frame = build_query_frame(
        "How many days did it take for me to finish the search "
        "after starting to work with Morgan?"
    )
    start = _operand(
        "start", "started working Morgan", "2026/02/15", event_time="none"
    )
    start.observed_at = "2026/03/02"
    finish = _operand(
        "finish", "finished", "the search", event_time="2026/03/01"
    )
    distractor = _operand(
        "noise", "plans find", "an unrelated partner", event_time="2026/03/02"
    )
    hint = duration_from_operands(frame, [start, finish, distractor], _overlap)
    assert hint is not None
    assert hint["elapsed_days"] == 14


def test_count_prefers_explicit_windowed_scalar_snapshot() -> None:
    frame = build_query_frame(
        "How many signed balls did I have in the first three months of collecting?"
    )
    hint = arithmetic_hint(
        frame,
        [
            _operand(
                "first",
                "collection",
                "15 signed balls since I started collecting three months ago",
                event_time="2023-07-11",
                quantity=15,
                unit="balls",
            ),
            _operand(
                "later",
                "added",
                "20 signed balls in the past few months",
                event_time="2023-12-30",
                quantity=20,
                unit="balls",
            ),
        ],
        [],
        _overlap,
    )
    assert hint is not None
    assert hint["operation"] == "scalar_snapshot"
    assert hint["value"] == 15
    assert hint["operand_ids"] == ["first"]


def test_ratio_percent_uses_two_query_grounded_amounts() -> None:
    frame = build_query_frame(
        "What percentage of the rural property's price is the renovation cost?"
    )
    hint = arithmetic_hint(
        frame,
        [
            _operand(
                "price",
                "listed",
                "rural property price",
                event_time="2023-05-23",
                quantity=200_000,
                unit="USD",
            ),
            _operand(
                "cost",
                "estimated cost",
                "renovation cost",
                event_time="2023-05-26",
                quantity=20_000,
                unit="USD",
            ),
        ],
        [],
        _overlap,
    )
    assert hint is not None
    assert hint["operation"] == "ratio_percent"
    assert hint["value"] == 10.0
