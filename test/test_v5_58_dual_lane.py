from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from graphmem.config import load_runtime_config
from graphmem.domain import QueryOperator
from graphmem.retrieval.navigator import operator_aware_legacy_lane


ROOT = Path(__file__).resolve().parents[1]


def test_v558_dual_lane_profiles_only_change_evidence_budget() -> None:
    turn64 = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_58_dual64.json")
    turn128 = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_58_dual128.json")

    assert turn64.retrieval.dual_lane_packing
    assert turn64.retrieval.dual_lane_precision_head == 32
    assert turn64.retrieval.dual_lane_rrf_k == 60
    assert not turn64.retrieval.dual_lane_proof_reserve
    assert not turn64.retrieval.obligation_aware_relations
    assert not turn64.retrieval.dialogue_response_closure
    assert turn64.query_budget.max_evidence_turns == 64
    assert turn128.query_budget.max_evidence_turns == 128
    assert turn128.query_budget.max_evidence_tokens == 24000
    assert replace(
        turn128.query_budget,
        max_evidence_turns=64,
        max_evidence_tokens=12000,
        max_answer_tokens=10000,
        max_answer_tokens_hard=13000,
    ) == turn64.query_budget
    assert turn128.retrieval == turn64.retrieval


def test_operator_aware_lane_is_query_and_transcript_shape_driven() -> None:
    assert operator_aware_legacy_lane(
        QueryOperator.LOOKUP, named_transcript=False, enabled=True)
    assert operator_aware_legacy_lane(
        QueryOperator.COUNT_DISTINCT, named_transcript=False, enabled=True)
    assert not operator_aware_legacy_lane(
        QueryOperator.DATE_DIFFERENCE, named_transcript=False, enabled=True)
    assert not operator_aware_legacy_lane(
        QueryOperator.LOOKUP, named_transcript=True, enabled=True)
    assert not operator_aware_legacy_lane(
        QueryOperator.LOOKUP, named_transcript=False, enabled=False)
