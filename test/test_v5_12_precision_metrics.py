from __future__ import annotations

import pytest

from graphmem.eval.metrics import (
    aggregate_metrics,
    ranked_retrieval_metrics,
    retrieval_set_metrics,
)


def test_full_memory_recall_does_not_hide_low_precision() -> None:
    gold = {"g1", "g2"}
    full_memory = {"g1", "g2", *(f"noise-{index}" for index in range(98))}

    row = retrieval_set_metrics(gold, full_memory, prefix="candidate_")

    assert row["candidate_turn_all_hit"]
    assert row["candidate_turn_recall"] == 1.0
    assert row["candidate_turn_precision"] == pytest.approx(0.02)
    assert row["candidate_turn_f1"] < 0.04


def test_top_k_metrics_expose_bad_ranking_inside_a_saturated_reservoir() -> None:
    gold = {"g1", "g2"}
    ranked = [*(f"noise-{index}" for index in range(40)), "g1", "g2"]

    row = ranked_retrieval_metrics(gold, ranked)

    assert row["candidate_top32_turn_recall"] == 0.0
    assert not row["candidate_top32_turn_all_hit"]
    assert row["candidate_last_gold_rank"] == 42
    assert row["candidate_average_precision"] == pytest.approx(
        ((1 / 41) + (2 / 42)) / 2)
    assert row["candidate_r_precision"] == 0.0
    assert row["candidate_ndcg_at_32"] == 0.0


def test_average_precision_rewards_gold_items_near_the_front() -> None:
    row = ranked_retrieval_metrics(
        {"g1", "g2"}, ["g1", "noise", "g2", "noise-2"])

    assert row["candidate_average_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert row["candidate_r_precision"] == pytest.approx(0.5)
    assert 0 < row["candidate_ndcg_at_8"] < 1


def test_aggregate_reports_macro_and_micro_precision() -> None:
    rows = [
        {
            "stratum": "s", "gold_turns": 1,
            "turns": 2, "turn_hits": 1,
            "turn_precision": 0.5, "turn_recall": 1.0, "turn_f1": 2 / 3,
        },
        {
            "stratum": "s", "gold_turns": 3,
            "turns": 10, "turn_hits": 1,
            "turn_precision": 0.1, "turn_recall": 1 / 3, "turn_f1": 2 / 13,
        },
    ]

    overall = aggregate_metrics(rows)["overall"]

    assert overall["turn_precision"] == pytest.approx(0.3)
    assert overall["micro_turn_precision"] == pytest.approx(2 / 12)
    assert overall["micro_turn_recall"] == pytest.approx(2 / 4)
