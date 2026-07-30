from graphmem_demo.v3.catalog_schema import EventFrameV3
from graphmem_demo.v3.planned_event_count import planned_event_identity_count
from graphmem_demo.v3.retrieval import build_query_frame


def _frame(
    frame_id: str,
    session_id: str,
    event_time: str,
    observed_at: str,
    *,
    participants: list[str] | None = None,
    label: str = "Alex and Morgan plan a workshop",
) -> EventFrameV3:
    return EventFrameV3(
        frame_id=frame_id,
        question_id="q",
        label=label,
        label_key=label.casefold(),
        participant_keys=participants or ["alex", "morgan"],
        status="planned",
        event_time=event_time,
        observed_at=observed_at,
        session_ids=[session_id],
        source_turn_ids=[f"{session_id}:t0"],
        retrieval_text=label,
    )


def test_plan_count_merges_repeated_target_intervals_and_session_followup() -> None:
    frame = build_query_frame(
        "How many times did Alex and Morgn plan a workshop together?"
    )
    rows = [
        _frame("f1", "s1", "next month", "8 July, 2023"),
        _frame("f2", "s2", "next month", "11 July, 2023"),
        _frame("f3", "s3", "next month", "1 October, 2023"),
        _frame("f4", "s4", "next month", "19 October, 2023"),
        _frame("f5", "s5", "Saturday after 2023-10-28", "28 October, 2023"),
        _frame("f6", "s5", "future after Saturday", "28 October, 2023"),
    ]
    hint = planned_event_identity_count(frame, rows)
    assert hint is not None
    assert hint["value"] == 3
    assert hint["complete"] is True
    assert len(hint["groups"]) == 3


def test_plan_count_requires_all_participants_and_matching_action() -> None:
    frame = build_query_frame(
        "How many times did Alex and Morgan plan a workshop together?"
    )
    rows = [
        _frame("solo", "s1", "next month", "8 July, 2023",
               participants=["alex"]),
        _frame("other", "s2", "next month", "8 July, 2023",
               label="Alex and Morgan plan a concert"),
    ]
    assert planned_event_identity_count(frame, rows) is None


def test_plan_count_refuses_completed_events_and_non_plan_queries() -> None:
    frame = build_query_frame(
        "How many times did Alex and Morgan attend a workshop?"
    )
    row = _frame("f1", "s1", "next month", "8 July, 2023")
    assert planned_event_identity_count(frame, [row]) is None
    plan_frame = build_query_frame(
        "How many times did Alex and Morgan plan a workshop?"
    )
    row.status = "complete"
    assert planned_event_identity_count(plan_frame, [row]) is None
