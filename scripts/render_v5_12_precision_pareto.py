#!/usr/bin/env python3
"""Render report-ready Recall--Precision Pareto tables and figures.

The input is the per-question output of ``measure_v5_12_precision_gate.py``.
Precision is annotation-scoped because benchmark gold turns are sufficient but
not necessarily exhaustive; paired comparisons remain valid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=WORKSPACE / "artifacts/report/v5_12/precision_gate_dev200")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def ranked_row(gold: set[tuple[Any, ...]], ranked: list[list[Any]], cutoff: int) -> dict[str, float]:
    ordered = [tuple(item) for item in ranked[:cutoff]]
    predicted = set(ordered)
    hits = len(gold & predicted)
    precision = ratio(hits, len(predicted))
    recall = ratio(hits, len(gold))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "all_hit": float(bool(gold) and gold <= predicted),
        "turns": float(len(predicted)),
    }


def ranking_quality(gold: set[tuple[Any, ...]], ranked: list[list[Any]]) -> dict[str, float]:
    ordered = [tuple(item) for item in ranked]
    hit_ranks = [rank for rank, item in enumerate(ordered, 1) if item in gold]
    average_precision = (
        sum(hit_index / rank for hit_index, rank in enumerate(hit_ranks, 1)) / len(gold)
        if gold else 0.0)
    first_rr = 1 / min(hit_ranks) if hit_ranks else 0.0
    r = len(gold)
    r_precision = ratio(len(gold & set(ordered[:r])), r)
    result = {
        "map": average_precision,
        "mrr": first_rr,
        "r_precision": r_precision,
    }
    for cutoff in (8, 16, 32):
        dcg = sum(1 / math.log2(rank + 1)
                  for rank, item in enumerate(ordered[:cutoff], 1) if item in gold)
        ideal = sum(1 / math.log2(rank + 1)
                    for rank in range(1, min(len(gold), cutoff) + 1))
        result[f"ndcg_at_{cutoff}"] = ratio(dcg, ideal)
    return result


def macro(items: list[dict[str, float]]) -> dict[str, float]:
    return {key: fmean(row[key] for row in items) for key in items[0]}


def same_budget_diagnostic(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    scored = []
    for row in rows:
        gold = {tuple(item) for item in row["gold_refs"]}
        packed = {tuple(item) for item in row[arm]["packed_refs"]}
        ranked = [tuple(item) for item in row[arm]["candidate_refs"]]
        top_n = set(ranked[:len(packed)])
        top_hits = gold & top_n
        pack_hits = gold & packed
        top_precision = ratio(len(top_hits), len(top_n))
        top_recall = ratio(len(top_hits), len(gold))
        scored.append({
            "pack_gold_hits": float(len(pack_hits)),
            "topn_gold_hits": float(len(top_hits)),
            "gold_hits_delta": float(len(pack_hits) - len(top_hits)),
            "gold_turns_lost": float(len(top_hits - packed)),
            "gold_turns_rescued": float(len(pack_hits - top_n)),
            "pack_churn_vs_topn": 1.0 - ratio(len(packed & top_n), len(packed)),
            "topn_precision": top_precision,
            "topn_recall": top_recall,
            "topn_f1": f1(top_precision, top_recall),
        })
    return macro(scored)


def main() -> None:
    args = parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in
            (args.input / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    summary = json.loads((args.input / "summary.json").read_text(encoding="utf-8"))
    arms = tuple(summary["overall"])
    cutoffs = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
    curve = []
    for cutoff in cutoffs:
        scored = [ranked_row(
            {tuple(item) for item in row["gold_refs"]},
            row["baseline"]["candidate_refs"], cutoff) for row in rows]
        aggregate = macro(scored)
        curve.append({"point": f"Top-{cutoff}", "cutoff": cutoff, **aggregate})
    full_scored = [ranked_row(
        {tuple(item) for item in row["gold_refs"]},
        row["baseline"]["candidate_refs"],
        len(row["baseline"]["candidate_refs"])) for row in rows]
    curve.append({"point": "Full reservoir", "cutoff": 0, **macro(full_scored)})

    pack_points = [{
        "arm": arm,
        "precision": summary["overall"][arm]["precision"],
        "recall": summary["overall"][arm]["recall"],
        "f1": summary["overall"][arm]["f1"],
        "all_hit": summary["overall"][arm]["all_hit"],
        "turns": summary["overall"][arm]["pack_turns"],
        "tokens": summary["overall"][arm]["evidence_tokens"],
        "latency_ms": summary["overall"][arm]["latency_ms"],
    } for arm in arms]
    same_budget = {arm: same_budget_diagnostic(rows, arm) for arm in arms}
    rank_quality = macro([
        ranking_quality({tuple(item) for item in row["gold_refs"]},
                        row["baseline"]["candidate_refs"])
        for row in rows])

    with (output / "recall_precision_curve.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)
    with (output / "pack_operating_points.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pack_points[0].keys())
        writer.writeheader()
        writer.writerows(pack_points)

    payload = {
        "schema_version": "graphmem-v5.12-precision-pareto-v1",
        "precision_scope": "official_gold_turns_only_lower_bound",
        "questions": len(rows),
        "ranking_quality": rank_quality,
        "candidate_curve": curve,
        "pack_points": pack_points,
        "same_budget_diagnostic": same_budget,
    }
    (output / "pareto.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V5.12 Recall--Precision Pareto",
        "",
        "> Precision 仅相对官方 gold turn 计算。标注未必穷尽全部相关证据，",
        "> 因而它是 annotation-scoped lower bound；同一标注上的配对比较有效。",
        "",
        "## 候选排序质量",
        "",
        f"- mAP: {rank_quality['map']:.2%}",
        f"- MRR: {rank_quality['mrr']:.2%}",
        f"- R-Precision: {rank_quality['r_precision']:.2%}",
        f"- nDCG@8 / @16 / @32: {rank_quality['ndcg_at_8']:.2%} / "
        f"{rank_quality['ndcg_at_16']:.2%} / {rank_quality['ndcg_at_32']:.2%}",
        "",
        "## 候选池工作点",
        "",
        "| Point | Turns | All-hit | Recall | Precision | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        if row["point"] in {"Top-8", "Top-16", "Top-32", "Top-64",
                            "Top-128", "Top-256", "Full reservoir"}:
            lines.append(
                f"| {row['point']} | {row['turns']:.1f} | {row['all_hit']:.2%} | "
                f"{row['recall']:.2%} | {row['precision']:.2%} | {row['f1']:.2%} |")
    lines.extend([
        "", "## 最终证据包工作点", "",
        "| Arm | Turns | Tokens | Latency (ms) | All-hit | Recall | Precision | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in pack_points:
        lines.append(
            f"| {row['arm']} | {row['turns']:.1f} | {row['tokens']:.1f} | "
            f"{row['latency_ms']:.1f} | {row['all_hit']:.2%} | {row['recall']:.2%} | "
            f"{row['precision']:.2%} | {row['f1']:.2%} |")
    lines.extend([
        "", "## 同证据条数 pack vs. candidate Top-N", "",
        "| Arm | Top-N Recall | Top-N Precision | Top-N F1 | Pack-TopN gold hits | Lost gold | Rescued gold | Selection churn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm, row in same_budget.items():
        lines.append(
            f"| {arm} | {row['topn_recall']:.2%} | {row['topn_precision']:.2%} | "
            f"{row['topn_f1']:.2%} | {row['gold_hits_delta']:+.3f} | "
            f"{row['gold_turns_lost']:.3f} | {row['gold_turns_rescued']:.3f} | "
            f"{row['pack_churn_vs_topn']:.2%} |")
    (output / "pareto.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphmem-matplotlib")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    axis.plot([100 * row["recall"] for row in curve],
              [100 * row["precision"] for row in curve], "o-", linewidth=1.8,
              color="#2563eb", label="Candidate Top-K")
    for row in curve:
        if row["point"] in {"Top-8", "Top-16", "Top-32", "Top-64",
                            "Top-128", "Top-256", "Full reservoir"}:
            axis.annotate(row["point"],
                          (100 * row["recall"], 100 * row["precision"]),
                          xytext=(4, 5), textcoords="offset points", fontsize=8)
    colors = ("#dc2626", "#059669", "#d97706", "#7c3aed")
    for color, row in zip(colors, pack_points):
        axis.scatter(100 * row["recall"], 100 * row["precision"], s=70,
                     marker="D", color=color, edgecolor="white", linewidth=0.8,
                     label=f"Pack: {row['arm']}")
    axis.set_xlabel("Gold Recall (%)")
    axis.set_ylabel("Annotation Precision lower bound (%)")
    axis.set_title("GraphMem bounded retrieval: Recall--Precision Pareto")
    axis.legend(fontsize=7.5, loc="best")
    fig.savefig(output / "recall_precision_pareto.pdf")
    fig.savefig(output / "recall_precision_pareto.png", dpi=220)
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
