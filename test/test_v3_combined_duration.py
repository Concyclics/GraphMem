from graphmem_demo.v3.combined_duration import combined_named_duration_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(
    node_id: str,
    text: str,
    *,
    role: str = "user",
    date: str = "2026-01-01",
) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id=node_id,
        session_date=date,
        turn_index=0,
        speaker="participant_1" if role == "user" else "participant_2",
        speaker_key="participant 1" if role == "user" else "participant 2",
        listener="",
        transport_role=role,
        text=text,
        retrieval_text=text,
    )


def test_combined_duration_requires_independent_named_sources() -> None:
    frame = build_query_frame(
        "How long did I take to finish 'Project Cedar' and 'Project Quartz' combined?"
    )
    hint = combined_named_duration_hint(frame, [
        _turn("t1", "I finished Project Cedar, which took me two and a half weeks."),
        _turn("t2", "I recently finished Project Quartz; it took three weeks."),
        _turn("noise", "I expect Project Amber would take seven weeks."),
    ])
    assert hint is not None
    assert hint["operation"] == "combined_named_duration"
    assert hint["value"] == 5.5
    assert hint["unit"] == "week"
    assert hint["source_turn_ids"] == ["t1", "t2"]
    assert [proof["entity"] for proof in hint["proofs"]] == [
        "Project Cedar", "Project Quartz",
    ]


def test_combined_duration_rejects_missing_or_planned_operand() -> None:
    frame = build_query_frame(
        'How long did I take to finish "Route Alpha" and "Route Beta" combined?'
    )
    hint = combined_named_duration_hint(frame, [
        _turn("t1", "I completed Route Alpha, which took two days."),
        _turn("t2", "I expect Route Beta would take three days."),
    ])
    assert hint is not None
    assert hint["operation"] == "combined_named_duration_incomplete"
    assert hint["complete"] is False
    assert hint["missing_or_ambiguous_entity"] == "Route Beta"


def test_combined_duration_does_not_borrow_assistant_estimate() -> None:
    frame = build_query_frame(
        "How long did I take to finish 'Model North' and 'Model South' combined?"
    )
    hint = combined_named_duration_hint(frame, [
        _turn("t1", "I finished Model North, and it took one day."),
        _turn("t2", "Model South should take two days to finish.", role="assistant"),
    ])
    assert hint is not None
    assert hint["complete"] is False


def test_combined_duration_normalizes_exact_mixed_units() -> None:
    frame = build_query_frame(
        "How long did I take to finish 'Pass One' and 'Pass Two' altogether?"
    )
    hint = combined_named_duration_hint(frame, [
        _turn("t1", "I finished Pass One; it took one day."),
        _turn("t2", "I completed Pass Two in 24 hours."),
    ])
    assert hint is not None
    assert hint["value"] == 48
    assert hint["unit"] == "hour"


def test_combined_duration_derives_each_named_interval_from_lossless_dates() -> None:
    frame = build_query_frame(
        "How many weeks in total did I spend reading 'Book One' and "
        "listening to 'Book Two' and 'Book Three'?"
    )
    hint = combined_named_duration_hint(frame, [
        _turn("one-start", "I started reading Book One.", date="2026-01-01"),
        _turn("one-end", "I finished Book One.", date="2026-01-15"),
        _turn("two-start", "I started listening to Book Two.", date="2026-02-01"),
        _turn("two-end", "I finished listening to Book Two.", date="2026-03-01"),
        _turn("three-start", "I began listening to Book Three.", date="2026-03-06"),
        _turn("three-end", "I completed Book Three.", date="2026-03-20"),
    ])
    assert hint is not None
    assert hint["operation"] == "combined_named_duration"
    assert hint["value"] == 8
    assert hint["unit"] == "week"
    assert hint["source_turn_ids"] == [
        "one-start", "one-end", "two-start", "two-end",
        "three-start", "three-end",
    ]
