from graphmem_demo.v3.dialogue_answer import dialogue_answer_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(index: int, speaker: str, text: str, *, session: str = "s") -> TurnNode:
    return TurnNode(
        node_id=f"{session}:t{index}",
        question_id="q",
        session_id=session,
        session_date="2026-04-01",
        turn_index=index,
        speaker=speaker,
        speaker_key=speaker.casefold(),
        listener="",
        transport_role="speaker",
        text=text,
        retrieval_text=f"{speaker} {text}",
    )


def test_dialogue_answer_binds_matching_question_next_turn_and_speaker() -> None:
    frame = build_query_frame("What is Rowan's preferred calibration profile?")
    turns = [
        _turn(0, "Mira", "Which calibration profile do you prefer?"),
        _turn(
            1, "Rowan",
            'I use the quiet option called "Delta Quiet" because it is stable.',
        ),
        _turn(2, "Mira", "What profile did you test yesterday?"),
        _turn(3, "Rowan", 'It was called "Alpha Fast".'),
        _turn(0, "Rowan", "Which calibration profile do you prefer?", session="x"),
        _turn(1, "Mira", 'Mine is called "Mira Default".', session="x"),
    ]
    hint = dialogue_answer_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is True
    assert hint["value"] == "Delta Quiet"
    assert hint["source_turn_ids"] == ["s:t0", "s:t1"]
    assert hint["answer_speaker"] == "Rowan"


def test_dialogue_answer_requires_explicit_named_span() -> None:
    frame = build_query_frame("What is Rowan's preferred calibration profile?")
    assert dialogue_answer_hint(
        frame,
        [
            _turn(0, "Mira", "Which calibration profile do you prefer?"),
            _turn(1, "Rowan", "The quiet one works well for me."),
        ],
    ) is None


def test_dialogue_answer_does_not_override_collection_query() -> None:
    frame = build_query_frame("What are Rowan's preferred calibration profiles?")
    assert frame.requested_operation == "list"
    assert dialogue_answer_hint(
        frame,
        [
            _turn(0, "Mira", "Which calibration profile do you prefer?"),
            _turn(1, "Rowan", 'It is called "Delta Quiet".'),
        ],
    ) is None
