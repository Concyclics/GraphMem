from graphmem_demo.v3.clock_arithmetic import arrival_clock_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, session: str, index: int, text: str, date: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id=session,
        session_date=date,
        turn_index=index,
        speaker="Alex",
        speaker_key="alex",
        listener="Sam",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_arrival_clock_binds_departure_duration_and_destination() -> None:
    frame = build_query_frame("What time did I reach the studio on Monday?")
    turns = [
        _turn("t0", "s0", 0, "I left home at 7:15 AM on Monday.", "2026-07-20 (Mon)"),
        _turn("t1", "s0", 1, "The trip took one hour and 30 minutes to get to the studio.", "2026-07-20 (Mon)"),
        _turn("n0", "s1", 0, "The studio opens at 12 PM.", "2026-07-19 (Sun)"),
    ]
    hint = arrival_clock_hint(frame, turns)
    assert hint is not None
    assert hint["operation"] == "arrival_clock_time"
    assert hint["value"] == "8:45 AM"
    assert hint["source_turn_ids"] == ["t0", "t1"]


def test_arrival_clock_abstains_without_relation_bound_duration() -> None:
    frame = build_query_frame("What time did I arrive at the library?")
    turns = [
        _turn("t0", "s0", 0, "I left home at 8 AM.", "2026-07-20"),
        _turn("t1", "s0", 1, "The library closes at 5 PM.", "2026-07-20"),
    ]
    assert arrival_clock_hint(frame, turns) is None


def test_arrival_clock_ignores_future_duration_preference_in_same_turn() -> None:
    frame = build_query_frame("What time did I reach the clinic on Monday?")
    turns = [
        _turn("t0", "old", 0, "I left home at 7 AM on Monday.", "2026-07-20"),
        _turn(
            "t1", "new", 0,
            "It took me two hours to get to the clinic last time, so if I could "
            "find something within an hour's drive, that would be better.",
            "2026-07-30",
        ),
    ]
    hint = arrival_clock_hint(frame, turns)
    assert hint is not None
    assert hint["operation"] == "arrival_clock_time"
    assert hint["value"] == "9:00 AM"
