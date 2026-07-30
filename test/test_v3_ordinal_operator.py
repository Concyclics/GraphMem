from graphmem_demo.v3.ordinal_operator import ordinal_list_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _overlap(frame, text: str) -> float:
    query = set(frame.content_terms)
    words = set(text.casefold().replace("-", " ").split())
    return len(query & words) / max(1, len(query))


def test_ordinal_item_uses_neighboring_turn_for_list_scope() -> None:
    frame = build_query_frame("What was the 3rd role in the remote work list?")
    turns = [
        TurnNode(
            node_id="u",
            question_id="q",
            session_id="s",
            session_date="2023-01-01",
            turn_index=0,
            speaker="a",
            speaker_key="a",
            listener="b",
            transport_role="user",
            text="Please list remote work roles.",
            retrieval_text="remote work roles",
        ),
        TurnNode(
            node_id="a",
            question_id="q",
            session_id="s",
            session_date="2023-01-01",
            turn_index=1,
            speaker="b",
            speaker_key="b",
            listener="a",
            transport_role="assistant",
            text="1. Editor\n2. Tutor\n3. Transcriptionist\n4. Bookkeeper",
            retrieval_text="Editor Tutor Transcriptionist Bookkeeper",
        ),
    ]
    hint = ordinal_list_hint(frame, turns, query_overlap=_overlap)
    assert hint is not None
    assert hint["operation"] == "ordinal_list_item"
    assert hint["position"] == 3
    assert hint["value"] == "Transcriptionist"
    assert hint["source_turn_ids"] == ["a"]


def test_temporal_first_comparison_is_not_treated_as_list_ordinal() -> None:
    frame = build_query_frame(
        "Which item did I purchase first, the lamp or the desk?"
    )
    turn = TurnNode(
        node_id="a",
        question_id="q",
        session_id="s",
        session_date="2023-01-01",
        turn_index=0,
        speaker="b",
        speaker_key="b",
        listener="a",
        transport_role="assistant",
        text="1. Unrelated advice\n2. Another suggestion",
        retrieval_text="lamp desk advice",
    )
    assert ordinal_list_hint(frame, [turn], query_overlap=_overlap) is None
