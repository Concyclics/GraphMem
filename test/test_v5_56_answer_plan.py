from __future__ import annotations

import hashlib

from graphmem.answer.answer_plan import apply_answer_plan, compile_answer_plan
from graphmem.answer.stage import PreparedAnswer
from graphmem.domain import canonical_json
from graphmem.tokenization import HeuristicTokenCounter


COUNTER = HeuristicTokenCounter()


def _prepared(question: str) -> PreparedAnswer:
    messages = (
        {"role": "system", "content": "Use only supplied memories."},
        {"role": "user", "content": (
            f"Question: {question}\n\nConversation memories:\n"
            "[AUX 1 rank=1] [s1 @ 2023-02-11] user: I started sculpting classes today. "
            "[source-time \"today\" => 2023-02-11; anchor=2023-02-11]\n"
            "[AUX 1 rank=2] [s1 @ 2023-03-04] user: I bought sculpting tools today. "
            "[source-time \"today\" => 2023-03-04; anchor=2023-03-04]\n"
            "[AUX 2 rank=3] [s2 @ 2023-04-01] user: I visited a museum.\n\n"
            f"Answer the original Question now: {question}")},
    )
    payload = hashlib.sha256(canonical_json(messages).encode()).hexdigest()
    return PreparedAnswer(
        question_id="q1", memory_id="m1", messages=messages,
        evidence_turn_ids=("t1", "t2", "t3"), dropped_turn_ids=(),
        evidence_tokens=60, packing_prompt_tokens=sum(
            COUNTER.count(row["content"]) for row in messages),
        closed_form=False, draft_text="", draft_certified=False,
        budget_relaxed=False, prompt_hash="base", prompt_payload_hash=payload,
        trace={"typed_readout_kind": "temporal",
               "prompt_version": "base-v1"})


def test_duration_plan_binds_two_verbatim_temporal_candidates() -> None:
    source = _prepared(
        "How many weeks passed between starting sculpting classes and buying tools?")

    plan = compile_answer_plan(source, max_candidates=3)

    assert plan is not None and plan.kind == "date_difference"
    assert [row.turn_id for row in plan.candidates] == ["t1", "t2"]
    assert all(row.has_time_anchor for row in plan.candidates)
    assert "museum" not in plan.render()
    assert "derived-from-two-bound-endpoints" in plan.render()


def test_plan_appends_without_changing_evidence_identity_or_order() -> None:
    source = _prepared("When did I buy sculpting tools?")

    updated = apply_answer_plan(
        source, COUNTER, max_candidates=3,
        enabled_kinds=("temporal_lookup",))

    assert updated.evidence_turn_ids == source.evidence_turn_ids
    assert updated.dropped_turn_ids == source.dropped_turn_ids
    assert source.messages[-1]["content"] in updated.messages[-1]["content"]
    assert updated.prompt_payload_hash != source.prompt_payload_hash
    assert updated.packing_prompt_tokens > source.packing_prompt_tokens
    assert updated.trace["answer_plan"]["kind"] == "temporal_lookup"
    assert "gold" not in updated.messages[-1]["content"].casefold()


def test_non_temporal_lookup_is_byte_identical() -> None:
    source = _prepared("What color is my bicycle?")
    source = PreparedAnswer.from_record({
        **source.to_record(), "trace": {"prompt_version": "base-v1"}})

    assert apply_answer_plan(source, COUNTER) is source


def test_last_weekend_is_not_treated_as_explicit_temporal_order() -> None:
    source = _prepared("Which game did I finally beat last weekend?")

    assert apply_answer_plan(source, COUNTER) is source


def test_explicit_first_comparison_remains_eligible() -> None:
    source = _prepared(
        "Which did I do first, starting sculpting classes or buying sculpting tools?")

    plan = compile_answer_plan(
        source, enabled_kinds=("temporal_order",))

    assert plan is not None and plan.kind == "temporal_order"
    assert len(plan.candidates) <= 3


def test_plan_falls_back_when_the_hard_prompt_ceiling_is_too_small() -> None:
    source = _prepared("When did I buy sculpting tools?")

    updated = apply_answer_plan(
        source, COUNTER, max_candidates=5,
        max_prompt_tokens=source.packing_prompt_tokens,
        enabled_kinds=("temporal_lookup",))

    assert updated is source
