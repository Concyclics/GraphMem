#!/usr/bin/env python3
"""Render the V5.54 construction-side candidate and Token Pareto evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"all_pairs": "#D84A4A", "flat_sparse": "#2378D7", "cir": "#18A999"}
LABELS = {"all_pairs": "All-pairs", "flat_sparse": "Flat sparse", "cir": "Coarsen-refine"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_font() -> None:
    path = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if path.exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = ["FandolHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def relation_nodes(database: Path) -> dict[str, int]:
    connection = sqlite3.connect(database)
    try:
        return {str(memory_id): int(count) for memory_id, count in connection.execute(
            "SELECT memory_id, COUNT(*) FROM graph_nodes "
            "WHERE node_type IN ('canonical_fact','event_skeleton') "
            "GROUP BY memory_id")}
    finally:
        connection.close()


def sum_diag(report: dict[str, Any], key: str) -> int:
    return sum(int(row.get("relation_candidate_diagnostics", {}).get(key, 0))
               for row in report["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    args = parser.parse_args()

    synthetic = json.loads(args.synthetic.read_text(encoding="utf-8"))
    build = json.loads(args.build_report.read_text(encoding="utf-8"))
    atomic = relation_nodes(args.graph_db)
    all_pairs = sum(value * (value - 1) // 2 for value in atomic.values())
    funnel = {
        "all_pairs": all_pairs,
        "coarse_candidate_pairs": sum_diag(build, "coarse_candidate_pairs"),
        "gated_child_pairs": sum_diag(build, "gated_child_pairs"),
        "atomic_relation_pairs_proposed": sum_diag(
            build, "atomic_relation_pairs_proposed"),
        "relation_mask_pairs": sum_diag(build, "relation_mask_pairs"),
        "accepted_pairs": sum_diag(build, "accepted_pairs"),
    }
    real_rows = []
    for row in build["rows"]:
        memory_id = str(row["memory_id"])
        n = atomic.get(memory_id, 0)
        diag = row.get("relation_candidate_diagnostics", {})
        if n and int(diag.get("coarse_candidate_pairs", 0)) > 0:
            real_rows.append((n, int(diag["coarse_candidate_pairs"])))
    real_exponent = float(np.polyfit(
        np.log([row[0] for row in real_rows]),
        np.log([row[1] for row in real_rows]), 1)[0])
    payload = {
        "schema_version": "graphmem-v5.54-build-index-pareto-v1",
        "sources": {
            "synthetic": str(args.synthetic),
            "synthetic_sha256": sha256(args.synthetic),
            "build_report": str(args.build_report),
            "build_report_sha256": sha256(args.build_report),
            "graph_db": str(args.graph_db),
        },
        "synthetic_candidate_scaling_exponent": synthetic[
            "candidate_scaling_exponent"],
        "real_510_memory_coarse_candidate_exponent_diagnostic": real_exponent,
        "real_510_memory_funnel": funnel,
        "relation_bearing_nodes": sum(atomic.values()),
        "memories": len(atomic),
        "all_pairs_to_atomic_proposed_reduction": (
            1.0 - funnel["atomic_relation_pairs_proposed"] / max(1, all_pairs)),
        "all_pairs_to_accepted_reduction": (
            1.0 - funnel["accepted_pairs"] / max(1, all_pairs)),
        "token_accounting": {
            "synthetic_relation_tokens": "fixed per-candidate envelope, not API usage",
            "frozen_rebuild_generated_tokens": build["summary"].get("tokens_total", 0),
            "frozen_rebuild_cached_input_tokens": build["summary"].get(
                "cached_input_tokens", 0),
            "extraction_and_relation_stages_must_not_be_conflated": True,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    setup_font()
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 3.75))
    data = synthetic["rows"]
    for method in ("all_pairs", "flat_sparse", "cir"):
        selected = sorted((row for row in data if row["method"] == method),
                          key=lambda row: row["n"])
        axes[0].plot([row["n"] for row in selected],
                     [row["candidate_relations"] for row in selected],
                     marker="o", color=COLORS[method], label=LABELS[method])
    axes[0].set(xscale="log", yscale="log", xlabel="关系单元数 $N$",
                ylabel="候选关系数", title="(a) 候选规模")
    for method in ("flat_sparse", "cir"):
        selected = sorted((row for row in data if row["method"] == method),
                          key=lambda row: row["n"])
        axes[1].plot([row["n"] for row in selected],
                     [row["relation_decision_tokens"] for row in selected],
                     marker="o", color=COLORS[method], label=LABELS[method])
    axes[1].set(xscale="log", yscale="log", xlabel="关系单元数 $N$",
                ylabel="关系判定 Token 上界", title="(b) Token envelope")
    labels = ["All-pairs", "Atomic\nproposed", "Typed mask", "Accepted"]
    values = [funnel["all_pairs"], funnel["atomic_relation_pairs_proposed"],
              funnel["relation_mask_pairs"], funnel["accepted_pairs"]]
    bars = axes[2].bar(labels, values,
                       color=[COLORS["all_pairs"], "#4E90D9", "#46B7A5", COLORS["cir"]])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("510 Memories 候选/边数")
    axes[2].set_title("(c) 真实构建候选漏斗")
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value * 1.12,
                     f"{value/1e6:.2f}M" if value >= 1e6 else f"{value/1e3:.1f}K",
                     ha="center", va="bottom", fontsize=8)
    for axis in axes:
        axis.grid(True, which="both", axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="lower center", frameon=False,
                  bbox_to_anchor=(0.5, -0.04))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(args.output_figure.with_suffix(f".{suffix}"), dpi=220,
                       bbox_inches="tight")
    plt.close(figure)

    reduction = 100 * payload["all_pairs_to_atomic_proposed_reduction"]
    largest_n = max(int(row["n"]) for row in data)
    largest = {str(row["method"]): row for row in data
               if int(row["n"]) == largest_n}
    token_reduction_vs_flat = 100 * (
        1.0 - largest["cir"]["relation_decision_tokens"]
        / max(1, largest["flat_sparse"]["relation_decision_tokens"]))
    table = (
        f"All-pairs 候选上界 & {funnel['all_pairs']:,} & -- & 精确计数 \\\\\n"
        f"Coarse candidates & {funnel['coarse_candidate_pairs']:,} & "
        f"{100*funnel['coarse_candidate_pairs']/all_pairs:.2f}\\% & 实测 \\\\\n"
        f"Atomic proposed & {funnel['atomic_relation_pairs_proposed']:,} & "
        f"{100*funnel['atomic_relation_pairs_proposed']/all_pairs:.2f}\\% & 实测 \\\\\n"
        f"Typed mask & {funnel['relation_mask_pairs']:,} & "
        f"{100*funnel['relation_mask_pairs']/all_pairs:.2f}\\% & 实测 \\\\\n"
        f"最终关系边 & {funnel['accepted_pairs']:,} & "
        f"{100*funnel['accepted_pairs']/all_pairs:.3f}\\% & 实测 \\\\\n"
        "\\bottomrule\n")
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.write_text(table, encoding="utf-8")
    analysis = (
        r"\paragraph{构建侧候选开支。}在当前 510 个 Memory 的 "
        f"{sum(atomic.values()):,} 个关系承载节点上，逐 Memory All-pairs 候选上界为 "
        f"{all_pairs:,}；完整父边门控流程只提出 "
        f"{funnel['atomic_relation_pairs_proposed']:,} 个原子候选，减少 {reduction:.2f}\\%。"
        f"在受控的 $N={largest_n:,}$ 工作点，CIR 的关系判定 Token envelope 为 "
        f"{largest['cir']['relation_decision_tokens']/1e6:.2f}M，较 Flat sparse 的 "
        f"{largest['flat_sparse']['relation_decision_tokens']/1e6:.2f}M 降低 "
        f"{token_reduction_vs_flat:.1f}\\%，同时受控多跳路径保留率为 "
        f"{100*largest['cir']['multi_hop_path_retention']:.1f}\\%（Flat sparse 为 "
        f"{100*largest['flat_sparse']['multi_hop_path_retention']:.1f}\\%）。"
        f"真实 Memory 间的 coarse-candidate 诊断拟合指数为 {real_exponent:.2f}，"
        r"但该拟合同时包含内容与规模差异，只作为真实工作负载佐证；复杂度指数以受控规模实验为准。"
        r"图中的关系判定 Token 是固定请求契约下的候选上界，不是 API usage；"
        r"当前 Safe-Witness 冻结重建复用了语义抽取且没有新增生成式关系调用，因此不能把抽取账本误写为关系建边 Token。"
        "\n")
    args.output_analysis.write_text(analysis, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
