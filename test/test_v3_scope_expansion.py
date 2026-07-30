from types import SimpleNamespace

from graphmem_demo.v3.action_semantics import (
    action_family_overlap,
    has_completed_participation,
)
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.scope_expansion import (
    lossless_event_turn_candidates,
    total_scope_candidates,
)


def _tokens(value: str) -> list[str]:
    return value.casefold().replace("-", " ").split()


def test_action_family_equates_buy_and_got_but_not_got_back() -> None:
    assert action_family_overlap("What did I buy?", "I got a new smoker") == 1
    assert action_family_overlap("What did I buy?", "I got back from a concert") == 0


def test_total_scope_uses_subject_unit_and_session_semantics() -> None:
    frame = build_query_frame("How many hours have I spent playing games in total?")
    operands = [
        SimpleNamespace(
            operand_id=f"o{index}", quantity=value, unit="hours",
            polarity="positive", modality="asserted", subject_key="participant 1",
            session_ids=[session], retrieval_text=f"participant completed {game}",
        )
        for index, (session, game, value) in enumerate([
            ("s1", "game Odyssey", 70),
            ("s2", "game Celeste", 10),
            ("s3", "game Finch", 5),
        ])
    ]
    turns = [
        SimpleNamespace(
            session_id=item.session_ids[0],
            retrieval_text=f"I played a game for {item.quantity} hours",
        )
        for item in operands
    ]
    sessions, operand_ids = total_scope_candidates(
        frame, operands, turns, tokenize=_tokens, similarity=lambda _item: 0.1,
    )
    assert set(sessions) == {"s1", "s2", "s3"}
    assert set(operand_ids) == {"o0", "o1", "o2"}


def test_lossless_event_beam_keeps_completed_first_person_turns() -> None:
    frame = build_query_frame(
        "How many different art-related events did I attend in the past month?"
    )
    turns = [
        SimpleNamespace(
            node_id="history", text="I went on a guided art tour at the History Museum.",
            speaker_key="participant 1",
        ),
        SimpleNamespace(
            node_id="planned", text="I plan to attend an art workshop.",
            speaker_key="participant 1",
        ),
    ]
    assert has_completed_participation(turns[0].text)
    assert not has_completed_participation("I will visit an art gallery this weekend.")
    assert lossless_event_turn_candidates(
        frame, turns, tokenize=_tokens, similarity=lambda _item: 0.1,
    ) == ["history"]
