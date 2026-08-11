from __future__ import annotations

import importlib.util
from pathlib import Path

from graphmem.config import load_runtime_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_v5_6_answer", ROOT / "scripts" / "run_v5_6_answer.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_v5_54_index_ablation",
    ROOT / "scripts" / "audit_v5_54_index_ablation.py")
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def test_v5_54_32_turn_profile_only_changes_the_turn_budget() -> None:
    low = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_54_accuracy32.json")
    high = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_54_accuracy64.json")

    assert low.query_budget.max_evidence_turns == 32
    assert high.query_budget.max_evidence_turns == 64
    assert low.query_budget.max_evidence_tokens == 12_000
    assert low.retrieval.navigator_options(compiled_cache_dir=None) == \
        high.retrieval.navigator_options(compiled_cache_dir=None)


def test_runtime_structural_overrides_are_independent_and_effective() -> None:
    runtime = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_54_accuracy64.json")

    seed_only = RUNNER.effective_runtime_navigator_options(
        runtime, disable_hierarchical_routing=True,
        disable_graph_traversal=True)
    hierarchy_only = RUNNER.effective_runtime_navigator_options(
        runtime, disable_graph_traversal=True)
    flat_graph = RUNNER.effective_runtime_navigator_options(
        runtime, disable_hierarchical_routing=True)
    full = RUNNER.effective_runtime_navigator_options(runtime)

    assert (seed_only["hierarchical_routing"], seed_only["h10_traversal"]) \
        == (False, False)
    assert (hierarchy_only["hierarchical_routing"],
            hierarchy_only["h10_traversal"]) == (True, False)
    assert (flat_graph["hierarchical_routing"], flat_graph["h10_traversal"]) \
        == (False, True)
    assert (full["hierarchical_routing"], full["h10_traversal"]) \
        == (True, True)


def test_final_audit_recomputes_nearest_rank_token_statistics() -> None:
    statistics = AUDIT.usage_statistics(list(range(1, 101)))

    assert statistics == {
        "count": 100, "mean": 50.5, "p50": 50,
        "p95": 95, "p99": 99, "max": 100,
    }
    assert AUDIT.prediction_sha256({"prediction": "same bytes"}) \
        == AUDIT.prediction_sha256({"prediction": "same bytes"})
    assert AUDIT.prediction_sha256({"prediction": "same bytes"}) \
        != AUDIT.prediction_sha256({"prediction": "different"})


def test_final_audit_indexes_question_rows_by_default() -> None:
    indexed = AUDIT.by_id([
        {"question_id": "q2", "value": 2},
        {"question_id": "q1", "value": 1},
    ])

    assert set(indexed) == {"q1", "q2"}
    assert indexed["q1"]["value"] == 1
