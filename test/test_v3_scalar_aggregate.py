from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.scalar_aggregate import named_scalar_average_hint
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id, question_id="q", session_id=node_id,
        session_date="2024-01-01", turn_index=0, speaker="participant_1",
        speaker_key="participant 1", listener="", transport_role="user",
        text=text, retrieval_text=text,
    )


def test_named_scalar_average_binds_each_partition_independently() -> None:
    frame = build_query_frame(
        "What is the average score of my undergraduate and graduate studies?"
    )
    hint = named_scalar_average_hint(frame, [
        _turn("grad", "I completed my Master's degree, where my score was 3.8."),
        _turn(
            "undergrad",
            "During my undergraduate studies, my score was 3.86.",
        ),
    ])
    assert hint is not None
    assert hint["value"] == 3.83
    assert {row["partition"] for row in hint["proofs"]} == {
        "undergraduate", "graduate studies",
    }


def test_named_scalar_average_refuses_a_missing_partition() -> None:
    frame = build_query_frame(
        "What is the average score of my undergraduate and graduate studies?"
    )
    assert named_scalar_average_hint(frame, [
        _turn("grad", "My Master's degree score was 3.8."),
    ]) is None
