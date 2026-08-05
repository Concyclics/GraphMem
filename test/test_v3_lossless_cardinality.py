from graphmem_demo.v3.lossless_cardinality import latest_cardinality_from_turns
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, date: str, text: str, role: str = "user") -> TurnNode:
    return TurnNode(
        node_id=node_id, question_id="q", session_id=node_id,
        session_date=date, turn_index=0,
        speaker="participant_1" if role == "user" else "participant_2",
        speaker_key="participant 1" if role == "user" else "participant 2",
        listener="", transport_role=role, text=text, retrieval_text=text,
    )


def test_latest_lossless_cardinality_supersedes_older_total() -> None:
    frame = build_query_frame(
        "How many sessions of the support group did I attend?"
    )
    hint = latest_cardinality_from_turns(frame, [
        _turn("old", "2023-05-11", "I attended three sessions of the support group."),
        _turn("new", "2023-10-30", "I remember attending five sessions of the support group."),
        _turn("assistant", "2023-11-01", "You attended seven sessions.", "assistant"),
    ])
    assert hint is not None
    assert hint["value"] == 5
    assert hint["source_turn_ids"] == ["new"]


def test_cardinality_does_not_transfer_from_sibling_entity() -> None:
    frame = build_query_frame(
        "How many sessions of the support group did I attend?"
    )
    assert latest_cardinality_from_turns(frame, [
        _turn("noise", "2023-10-30", "I attended five sessions of yoga class."),
    ]) is None


def test_cardinality_resolves_adjacent_entity_reference() -> None:
    frame = build_query_frame(
        "How many sessions of the support group did I attend?"
    )
    hint = latest_cardinality_from_turns(frame, [
        _turn(
            "new",
            "2023-10-30",
            "I was thinking about the support group I attended. "
            "I remember attending five sessions.",
        ),
    ])
    assert hint is not None
    assert hint["value"] == 5


def test_scale_denominator_is_not_a_collection_cardinality() -> None:
    frame = build_query_frame(
        "How many model kits have I worked on or bought?"
    )
    assert latest_cardinality_from_turns(frame, [
        _turn(
            "scale", "2023-05-27",
            "I want advanced techniques for a 1/72 scale model like this.",
        ),
    ]) is None
