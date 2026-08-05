from types import SimpleNamespace

from graphmem_demo.v3.event_interval import event_lifecycle_duration_hint
from graphmem_demo.v3.retrieval import build_query_frame


def _turn(
    node_id: str,
    session: str,
    index: int,
    date_text: str,
    speaker: str,
    text: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        session_id=session,
        turn_index=index,
        session_date=date_text,
        speaker=speaker,
        speaker_key=speaker.casefold(),
        text=text,
    )


def test_project_duration_uses_received_to_completed_lifecycle() -> None:
    frame = build_query_frame(
        "How long did Alex work on the robotics project from the professor?"
    )
    turns = [
        _turn(
            "s1:t0", "s1", 0, "1 February, 2023", "Alex",
            "My professor gave me a huge robotics project.",
        ),
        _turn(
            "s2:t0", "s2", 0, "6 June, 2023", "Alex",
            "I finally wrapped the engineering project up last month.",
        ),
        _turn(
            "s3:t0", "s3", 0, "1 January, 2020", "Alex",
            "I started yoga and finished a class.",
        ),
    ]
    hint = event_lifecycle_duration_hint(frame, turns)
    assert hint is not None
    assert hint["unit"] == "months"
    assert 3 <= hint["value"] <= 4
    assert hint["source_turn_ids"] == ["s1:t0", "s2:t0"]


def test_relationship_duration_uses_dialogue_bridge_and_relative_weeks() -> None:
    frame = build_query_frame(
        "How long did Evan and his partner date before getting married?"
    )
    turns = [
        _turn(
            "s1:t0", "s1", 0, "7 August, 2023", "Evan",
            "Last week I met a Canadian woman and we fell for each other.",
        ),
        _turn(
            "s2:t0", "s2", 0, "26 December, 2023", "Evan",
            "I got married last week.",
        ),
        _turn(
            "s2:t1", "s2", 1, "26 December, 2023", "Morgan",
            "Was it the same Canadian woman?",
        ),
    ]
    hint = event_lifecycle_duration_hint(frame, turns)
    assert hint is not None
    assert hint["unit"] == "months"
    assert 4 <= hint["value"] <= 5
    assert hint["source_turn_ids"] == ["s1:t0", "s2:t0"]


def test_lifecycle_duration_refuses_different_subject() -> None:
    frame = build_query_frame("How long did Alex work on the project?")
    turns = [
        _turn(
            "s1:t0", "s1", 0, "1 February, 2023", "Morgan",
            "I received the project.",
        ),
        _turn(
            "s2:t0", "s2", 0, "1 May, 2023", "Alex",
            "I finished the project.",
        ),
    ]
    assert event_lifecycle_duration_hint(frame, turns) is None
