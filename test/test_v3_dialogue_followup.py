from graphmem_demo.v3.dialogue_followup import dialogue_followup_plan_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(index: int, speaker: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=f"s:t{index}",
        question_id="q",
        session_id="s",
        session_date="2026-01-01",
        turn_index=index,
        speaker=speaker,
        speaker_key=speaker.casefold(),
        listener="",
        transport_role="user" if speaker == "Mira" else "assistant",
        text=text,
        retrieval_text=text,
    )


def test_commitment_response_plan_follows_lossless_adjacency() -> None:
    frame = build_query_frame(
        "What did Mira plan to do with the blueprint Rowan promised to share?"
    )
    hint = dialogue_followup_plan_hint(frame, [
        _turn(0, "Mira", "Could you send me the bridge blueprint?"),
        _turn(1, "Rowan", "Sure, I can give it to you tomorrow."),
        _turn(2, "Mira", "Great! I'm going to review it with my team this weekend."),
        _turn(3, "Rowan", "Let me know how it goes."),
    ])
    assert hint is not None
    assert hint["operation"] == "dialogue_followup_plan"
    assert hint["value"] == "review it with my team this weekend"
    assert hint["source_turn_ids"] == ["s:t0", "s:t1", "s:t2"]


def test_commitment_response_plan_rejects_wrong_recipient() -> None:
    frame = build_query_frame(
        "What did Mira plan to do with the blueprint Rowan promised to share?"
    )
    assert dialogue_followup_plan_hint(frame, [
        _turn(0, "Mira", "Could you send me the bridge blueprint?"),
        _turn(1, "Rowan", "Sure, I can give it to you tomorrow."),
        _turn(2, "Kai", "I'm going to review it with my team."),
    ]) is None
