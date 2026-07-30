from graphmem_demo.v3.event_onset import event_onset_from_sources
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(
    index: int,
    speaker: str,
    text: str,
    day: str = "19 August, 2023",
) -> TurnNode:
    return TurnNode(
        node_id=f"s:t{index}",
        question_id="q",
        session_id="s",
        session_date=day,
        turn_index=index,
        speaker=speaker,
        speaker_key=speaker.casefold(),
        listener="",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_direct_vague_onset_is_anchored_to_lossless_session_date() -> None:
    frame = build_query_frame("When did Mira start taking ceramics classes?")
    hint = event_onset_from_sources(frame, [
        _turn(0, "Mira", "I started making pottery years ago."),
        _turn(1, "Mira", "I started taking ceramics classes a few days ago."),
    ])
    assert hint is not None
    assert hint["value"] == "a few days before August 19, 2023"
    assert hint["onset_date"] == "2023-08-16"
    assert hint["source_turn_ids"] == ["s:t1"]


def test_present_perfect_duration_backshifts_anaphoric_activity_window() -> None:
    frame = build_query_frame("When did Rowan resume repairing sensors?")
    hint = event_onset_from_sources(frame, [
        _turn(0, "Rowan", "I repair environmental sensors.", "27 March, 2022"),
        _turn(1, "Mira", "How long have you been doing that?", "27 March, 2022"),
        _turn(2, "Rowan", "I've been doing it for a month now.", "27 March, 2022"),
    ])
    assert hint is not None
    assert hint["value"] == "February 2022"
    assert hint["completion_basis"] == "present_perfect_duration_backshift"
    assert hint["source_turn_ids"] == ["s:t0", "s:t2"]


def test_onset_does_not_borrow_other_speakers_relative_expression() -> None:
    frame = build_query_frame("When did Rowan start taking ceramics classes?")
    assert event_onset_from_sources(frame, [
        _turn(0, "Mira", "I started taking ceramics classes a few days ago."),
        _turn(1, "Rowan", "That sounds enjoyable."),
    ]) is None
