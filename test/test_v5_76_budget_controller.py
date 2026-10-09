from __future__ import annotations

from graphmem.answer.budget_controller import (
    answer_signature, decide_answer_budget, modal_candidate_choice,
    reliable_candidate_choice, stage_normalized_candidate_choice,
)


def _candidates(*answers: str) -> list[dict[str, str]]:
    return [{"prediction": answer} for answer in answers]


def test_surface_only_variation_does_not_expand_easy_query() -> None:
    decision = decide_answer_budget(
        {"adaptive_recall_severity": 0},
        _candidates("Alice's dog is Luna.", "alice’s dog is luna"),
        disagreement_prefix=2)

    assert answer_signature("Alice's dog is Luna.") == answer_signature(
        "alice’s dog is luna")
    assert not decision.expand
    assert decision.target_turns == 32
    assert decision.target_tokens == 2200
    assert decision.target_candidates == 2


def test_answer_disagreement_triggers_additive_second_pass() -> None:
    decision = decide_answer_budget(
        {"adaptive_recall_severity": 0},
        _candidates("Luna", "Oliver"), disagreement_prefix=2)

    assert decision.expand
    assert decision.retain_first_pass
    assert decision.target_turns == 64
    assert decision.target_tokens == 2200
    assert decision.target_candidates == 8
    assert decision.reasons == ("answer_disagreement",)


def test_accuracy_policy_uses_severity_two_without_token_growth() -> None:
    decision = decide_answer_budget(
        {"adaptive_recall_severity": 2},
        _candidates("Luna", "Luna"), policy="accuracy")

    assert decision.expand
    assert decision.target_turns == 64
    assert decision.target_tokens == 2200
    assert "closure_severity_ge_2" in decision.reasons


def test_pareto_policy_expands_severity_two_lookup_but_not_temporal() -> None:
    candidates = _candidates("Luna", "Luna")
    lookup = decide_answer_budget(
        {"adaptive_recall_severity": 2, "adaptive_recall_route": "lookup"},
        candidates, policy="pareto")
    temporal = decide_answer_budget(
        {"adaptive_recall_severity": 2, "adaptive_recall_route": "temporal"},
        candidates, policy="pareto")

    assert lookup.expand
    assert "lookup_witness_severity_ge_2" in lookup.reasons
    assert not temporal.expand


def test_token_budget_grows_only_after_measured_cap() -> None:
    decision = decide_answer_budget(
        {
            "adaptive_recall_severity": 6,
            "adaptive_recall_triggered": True,
            "pack_token_cap_reached": True,
        },
        _candidates("Luna", "Luna"))

    assert decision.target_turns == 80
    assert decision.target_tokens == 4500


def test_cross_stage_consensus_has_a_deterministic_choice() -> None:
    rows = [
        {"prediction": "Luna", "budget_stage": "base32", "family": "direct"},
        {"prediction": "Oliver", "budget_stage": "base32", "family": "direct"},
        {"prediction": "Luna.", "budget_stage": "expanded64", "family": "audit"},
    ]
    assert reliable_candidate_choice(rows) == 0


def test_weak_single_stage_split_requires_verification() -> None:
    rows = [
        {"prediction": "Luna", "budget_stage": "base32"},
        {"prediction": "Luna", "budget_stage": "base32"},
        {"prediction": "Oliver", "budget_stage": "base32"},
        {"prediction": "Milo", "budget_stage": "base32"},
    ]
    assert reliable_candidate_choice(rows) is None


def test_stage_normalized_vote_does_not_overweight_larger_stage() -> None:
    rows = [
        {"prediction": "Luna", "budget_stage": "base32"},
        {"prediction": "Luna", "budget_stage": "base32"},
        {"prediction": "Oliver", "budget_stage": "expanded64"},
        {"prediction": "Oliver", "budget_stage": "expanded64"},
        {"prediction": "Oliver", "budget_stage": "expanded64"},
        {"prediction": "Milo", "budget_stage": "expanded64"},
    ]
    assert modal_candidate_choice(rows) == 2
    assert stage_normalized_candidate_choice(rows) == 0
