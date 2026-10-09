from __future__ import annotations

from graphmem.answer.incremental import (
    build_incremental_answer_messages, build_merged_expansion_messages,
    plan_incremental_evidence,
)
from graphmem.domain import SourceTurn


def _turn(turn_id: str, text: str) -> SourceTurn:
    return SourceTurn(
        turn_id=turn_id, memory_id="memory:1", session_id="session_1",
        turn_index=0, speaker="Alice", listener="Bob", role="user",
        timestamp="10 May, 2024", raw_text=text, content_hash=turn_id,
    )


def test_incremental_plan_transmits_only_delta_and_bounded_anchors() -> None:
    plan = plan_incremental_evidence(
        ("a", "b", "c", "d"), ("b", "c", "e", "f"),
        preferred_anchor_ids=("a", "c", "c"), max_anchors=2)

    assert plan.added_turn_ids == ("e", "f")
    assert plan.anchor_turn_ids == ("a", "c")
    assert plan.removed_turn_ids == ("a", "d")
    assert plan.retained_turn_ids == ("b", "c")
    assert plan.transmitted_turn_ids == ("a", "c", "e", "f")


def test_incremental_prompt_marks_proposals_as_non_evidence() -> None:
    messages = build_incremental_answer_messages(
        question="When did Alice travel?",
        first_pass_candidates=(
            {"prediction": "Last Friday."},
            {"prediction": "last friday"},
            {"prediction": "On 3 May."},
        ),
        anchor_turns=(_turn("a", "I planned a trip."),),
        added_turns=(_turn("b", "I travelled last Friday."),),
        route="temporal", missing_obligation_count=1,
        escalation_reasons=("packed_witness_incomplete",),
    )

    assert len(messages) == 2
    assert "not new source evidence" in messages[0]["content"]
    assert "absence here is not a contradiction" in messages[0]["content"]
    user = messages[1]["content"]
    assert user.count("[PROPOSAL") == 2
    assert "[BASE ANCHOR 1]" in user
    assert "[NEW EVIDENCE 1]" in user
    assert "unresolved_bindings=1" in user
    assert "Question (answer this exact relation)" in user


def test_incremental_prompt_rejects_empty_delta() -> None:
    try:
        build_incremental_answer_messages(
            question="What happened?", first_pass_candidates=("Nothing",),
            added_turns=())
    except ValueError as error:
        assert "at least one new source turn" in str(error)
    else:
        raise AssertionError("empty incremental evidence should fail")


def test_merged_expansion_preserves_base_prompt_and_calls_model_once() -> None:
    base = (
        {"role": "system", "content": "Use supplied memories."},
        {"role": "user", "content": "Question: Where did Alice go?\nCore."},
    )
    messages = build_merged_expansion_messages(
        base_messages=base, question="Where did Alice go?",
        added_turns=(_turn("b", "I went to Kyoto."),),
        route="multi_hop", missing_obligation_count=1,
        escalation_reasons=("packed_witness_incomplete",),
    )

    assert len(messages) == 2
    assert messages[0]["content"].startswith(base[0]["content"])
    assert messages[1]["content"].startswith(base[1]["content"])
    assert "[EXPANDED SOURCE 1]" in messages[1]["content"]
    assert messages[1]["content"].count("Kyoto") == 1
    assert "Question (answer this exact relation)" in messages[1]["content"]
