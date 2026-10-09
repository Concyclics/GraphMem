from __future__ import annotations

from pathlib import Path

from graphmem.config import load_runtime_config
from graphmem.domain import (
    CertificateStatus,
    EvidenceCertificate,
    OperandSpec,
    ProofObligation,
    QueryOperator,
)
from graphmem.retrieval.adaptive_recall import plan_adaptive_recall
from graphmem.retrieval.query_ir import QueryIR


def _certificate(
    obligation: ProofObligation, *, complete: bool,
    post_pack_complete: bool,
) -> EvidenceCertificate:
    return EvidenceCertificate(
        question_kind="date_difference",
        required_slots=(obligation.obligation_id,),
        covered_slots=((obligation.obligation_id,) if complete else ()),
        missing_slots=(() if complete else (obligation.obligation_id,)),
        complete=complete,
        iterations=1,
        status=(CertificateStatus.COMPLETE if post_pack_complete
                else CertificateStatus.INCOMPLETE_OPERAND),
        post_pack_complete=post_pack_complete,
    )


def test_adaptive_recall_keeps_complete_pack_at_baseline() -> None:
    obligation = ProofObligation("endpoint", "left", "time_endpoint")
    ir = QueryIR(
        "How long was it between the two events?",
        QueryOperator.DATE_DIFFERENCE,
        (OperandSpec("left"), OperandSpec("right")),
        (obligation,),
        compile_confidence=0.65,
        soft_fallback_applied=True,
    )
    complete = _certificate(
        obligation, complete=True, post_pack_complete=True)

    plan = plan_adaptive_recall(
        ir, complete, complete, hard_turn_limit=128,
        candidate_count=300)

    assert not plan.triggered
    assert not plan.active
    assert plan.target_turns == 64


def test_adaptive_recall_uses_maximum_for_uncertain_multihop_gap() -> None:
    obligation = ProofObligation("endpoint", "left", "time_endpoint")
    ir = QueryIR(
        "How long was it between the two events?",
        QueryOperator.DATE_DIFFERENCE,
        (OperandSpec("left"), OperandSpec("right")),
        (obligation,),
        compile_confidence=0.65,
        soft_fallback_applied=True,
    )
    incomplete = _certificate(
        obligation, complete=False, post_pack_complete=False)

    plan = plan_adaptive_recall(
        ir, incomplete, incomplete, hard_turn_limit=128,
        candidate_count=300)

    assert plan.triggered
    assert plan.target_turns == 96
    assert "queryir_uncertain" in plan.reasons
    assert "missing_time_endpoint" in plan.reasons


def test_adaptive_recall_does_not_expand_confident_scalar_lookup() -> None:
    obligation = ProofObligation("binding", "item", "binding")
    ir = QueryIR(
        "What is Alice's dog's name?", QueryOperator.LOOKUP,
        (OperandSpec("item"),), (obligation,))
    incomplete = _certificate(
        obligation, complete=False, post_pack_complete=False)

    plan = plan_adaptive_recall(
        ir, incomplete, incomplete, hard_turn_limit=128,
        candidate_count=300)

    assert not plan.triggered


def test_accuracy_profile_can_expand_observable_lookup_witness_gap() -> None:
    obligation = ProofObligation("binding", "item", "binding")
    ir = QueryIR(
        "What is Alice's dog's name?", QueryOperator.LOOKUP,
        (OperandSpec("item"),), (obligation,))
    incomplete = _certificate(
        obligation, complete=False, post_pack_complete=False)

    plan = plan_adaptive_recall(
        ir, incomplete, incomplete, hard_turn_limit=80,
        base_turns=32, medium_turns=64, maximum_turns=80,
        lookup_minimum_severity=2, candidate_count=300)

    assert plan.triggered
    assert plan.target_turns == 64
    assert "lookup_witness_expansion" in plan.reasons


def test_accuracy_profile_can_expand_soft_fallback_before_answer() -> None:
    obligation = ProofObligation("binding", "item", "binding")
    ir = QueryIR(
        "What is Alice's dog's name?", QueryOperator.LOOKUP,
        (OperandSpec("item"),), (obligation,),
        compile_confidence=0.65, soft_fallback_applied=True)
    incomplete = _certificate(
        obligation, complete=False, post_pack_complete=False)

    plan = plan_adaptive_recall(
        ir, incomplete, incomplete, hard_turn_limit=80,
        base_turns=32, medium_turns=64, maximum_turns=80,
        expand_queryir_soft_fallback=True, candidate_count=300)

    assert plan.triggered
    assert "queryir_soft_fallback_expansion" in plan.reasons


def test_adaptive_recall_ignores_single_weak_complex_deficit() -> None:
    obligation = ProofObligation("endpoint", "left", "time_endpoint")
    ir = QueryIR(
        "When did Alice arrive?", QueryOperator.ARGMIN_TIME,
        (OperandSpec("left"),), (obligation,))
    closure = _certificate(
        obligation, complete=True, post_pack_complete=True)
    packed = _certificate(
        obligation, complete=False, post_pack_complete=False)

    plan = plan_adaptive_recall(
        ir, closure, packed, hard_turn_limit=128,
        candidate_count=300)

    assert plan.severity == 2
    assert not plan.triggered


def test_adaptive_recall_uses_medium_below_maximum_severity() -> None:
    obligation = ProofObligation("collection", "items", "collection")
    ir = QueryIR(
        "What items did Alice and Bob mention?", QueryOperator.UNION_DISTINCT,
        (OperandSpec("items"),), (obligation,))
    incomplete = _certificate(
        obligation, complete=False, post_pack_complete=False)

    plan = plan_adaptive_recall(
        ir, incomplete, incomplete, hard_turn_limit=128,
        candidate_count=300)

    assert plan.severity == 4
    assert plan.triggered
    assert plan.target_turns == 80


def test_adaptive_recall_can_repair_tail_without_growing_budget() -> None:
    obligation = ProofObligation("source", "item", "provenance")
    ir = QueryIR(
        "What is Alice's dog's name?", QueryOperator.LOOKUP,
        (OperandSpec("item"),), (obligation,),
        compile_confidence=0.65, soft_fallback_applied=True)
    closure = _certificate(
        obligation, complete=True, post_pack_complete=True)
    packed = _certificate(
        obligation, complete=False, post_pack_complete=False)

    plan = plan_adaptive_recall(
        ir, closure, packed, hard_turn_limit=80,
        base_turns=32, medium_turns=64, maximum_turns=80,
        enable_in_budget_repair=True, repair_minimum_severity=2,
        candidate_count=300)

    assert plan.active
    assert not plan.triggered
    assert plan.in_budget_repair
    assert plan.target_turns == 32


def test_adaptive_recall_expands_tokens_only_after_observed_cap() -> None:
    obligation = ProofObligation("collection", "items", "collection")
    ir = QueryIR(
        "What items did Alice mention?", QueryOperator.UNION_DISTINCT,
        (OperandSpec("items"),), (obligation,))
    incomplete = _certificate(
        obligation, complete=False, post_pack_complete=False)

    without_cap = plan_adaptive_recall(
        ir, incomplete, incomplete, hard_turn_limit=80,
        base_turns=32, medium_turns=64, maximum_turns=80,
        hard_token_limit=5000, base_tokens=2200,
        medium_tokens=3400, maximum_tokens=4500,
        base_token_cap_reached=False, candidate_count=300)
    with_cap = plan_adaptive_recall(
        ir, incomplete, incomplete, hard_turn_limit=80,
        base_turns=32, medium_turns=64, maximum_turns=80,
        hard_token_limit=5000, base_tokens=2200,
        medium_tokens=3400, maximum_tokens=4500,
        base_token_cap_reached=True, candidate_count=300)

    assert without_cap.target_turns == with_cap.target_turns == 64
    assert without_cap.target_tokens == 2200
    assert with_cap.target_tokens == 3400


def test_v574_runtime_declares_baseline_and_hard_caps() -> None:
    config = load_runtime_config(
        Path("configs/v5/runtime_v5_74_adaptive64.json"))
    assert config.retrieval.adaptive_recall
    assert config.retrieval.adaptive_recall_base_turns == 64
    assert config.retrieval.adaptive_recall_medium_turns == 80
    assert config.retrieval.adaptive_recall_max_turns == 96
    assert config.retrieval.adaptive_recall_minimum_severity == 3
    assert config.retrieval.adaptive_recall_maximum_tier_severity == 6
    assert config.query_budget.max_evidence_turns == 128
    assert config.retrieval.navigator_options()["adaptive_recall"] is True


def test_v576_runtime_declares_independent_turn_and_token_tiers() -> None:
    config = load_runtime_config(
        Path("configs/v5/runtime_v5_76_adaptive_budget.json"))
    retrieval = config.retrieval
    assert retrieval.adaptive_recall_in_budget_repair
    assert retrieval.adaptive_recall_repair_minimum_severity == 2
    assert (
        retrieval.adaptive_recall_base_tokens,
        retrieval.adaptive_recall_medium_tokens,
        retrieval.adaptive_recall_max_tokens,
    ) == (2200, 3800, 4500)
    assert config.query_budget.max_evidence_tokens == 5000
    options = retrieval.navigator_options()
    assert options["adaptive_recall_base_tokens"] == 2200
    assert options["adaptive_recall_in_budget_repair"] is True
    assert options["adaptive_recall_lookup_minimum_severity"] == 2
    assert options["adaptive_recall_expand_queryir_soft_fallback"] is True
