#!/usr/bin/env python3
"""Render the V5.54 factorial index ablation into report-native assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ARMS = ("seed_only", "hierarchy_only", "flat_graph", "full")
LABELS = {
    "seed_only": "Seed-only", "hierarchy_only": "Hierarchy-only",
    "flat_graph": "Flat Graph", "full": "Full GraphMem",
}
COLORS = {
    "seed_only": "#A7B6C2", "hierarchy_only": "#4E90D9",
    "flat_graph": "#E6A64C", "full": "#18A999",
}
GROUPS = (
    ("lme_multi_session", "LME Multi-session"),
    ("lme_temporal", "LME Temporal"),
    ("locomo_multihop", "LoCoMo Multi-hop"),
    ("locomo_temporal", "LoCoMo Temporal"),
)


def setup_font() -> None:
    path = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if path.exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = ["FandolHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def pct(value: float | None) -> str:
    return "待补" if value is None else f"{100 * value:.1f}"


def pp(value: float | None) -> str:
    return "待补" if value is None else f"{100 * value:+.1f}"


def number(value: float | None, digits: int = 1) -> str:
    return "待补" if value is None else f"{value:.{digits}f}"


def accuracy(arm: dict, group: str) -> float | None:
    row = arm.get("accuracy", {}).get(group, {})
    return float(row["accuracy"]) if row.get("accuracy") is not None else None


def metric(arm: dict, group: str, key: str) -> float | None:
    value = arm.get("retrieval", {}).get(group, {}).get(key)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    generated = args.report / "generated"
    figures = args.report / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for budget in (32, 64):
        arms = payload["budgets"][str(budget)]["arms"]
        for arm in ARMS:
            row = arms[arm]
            cells = [str(budget), LABELS[arm]]
            cells += [pct(accuracy(row, group)) for group, _label in GROUPS]
            cells += [pct(accuracy(row, "hard869")), pct(accuracy(row, "overall"))]
            cells += [number(metric(row, "overall", "api_prompt_tokens"), 0),
                      pct(metric(row, "hard869", "turn_all_hit")),
                      pct(metric(row, "hard869", "turn_precision")),
                      number(metric(row, "overall", "visited_edges")),
                      number(metric(row, "overall", "packed_turns"))]
            table_rows.append(" & ".join(cells) + r" \\")
    (generated / "v5_54_index_ablation_table.tex").write_text(
        "\n".join(table_rows) + "\n\\bottomrule\n", encoding="utf-8")

    setup_font()
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.9))
    width = 0.18
    for budget_index, budget in enumerate((32, 64)):
        arms = payload["budgets"][str(budget)]["arms"]
        offset = -1.5 * width if budget == 32 else 0.5 * width
        for arm_index, arm in enumerate(ARMS):
            value = 100 * float(accuracy(arms[arm], "hard869") or 0)
            x = arm_index + (budget_index - 0.5) * 0.34
            axes[0].bar(x, value, width=0.30, color=COLORS[arm],
                        alpha=0.68 if budget == 32 else 1.0,
                        hatch="//" if budget == 32 else None)
            axes[0].text(x, value + 0.35, f"{value:.1f}", ha="center", fontsize=7)
    axes[0].set_xticks(range(len(ARMS)), [LABELS[row] for row in ARMS], rotation=15)
    axes[0].set_ylabel("Hard-869 Accuracy (%)")
    axes[0].set_title("(a) 2×2 结构消融")

    xpositions = range(len(GROUPS))
    for budget, marker, linestyle in ((32, "o", "--"), (64, "s", "-")):
        arms = payload["budgets"][str(budget)]["arms"]
        deltas = [100 * ((accuracy(arms["full"], group) or 0)
                         - (accuracy(arms["seed_only"], group) or 0))
                  for group, _label in GROUPS]
        axes[1].plot(list(xpositions), deltas, marker=marker, linestyle=linestyle,
                     color=COLORS["full"], label=f"{budget}-turn")
    axes[1].axhline(0, color="#59636E", linewidth=0.8)
    axes[1].set_xticks(list(xpositions), [label for _group, label in GROUPS], rotation=18)
    axes[1].set_ylabel("Full − Seed-only (pp)")
    axes[1].set_title("(b) 分题型净增益")
    axes[1].legend(frameon=False)

    for budget, marker in ((32, "o"), (64, "s")):
        arms = payload["budgets"][str(budget)]["arms"]
        for arm in ARMS:
            x = metric(arms[arm], "overall", "api_prompt_tokens") or 0
            y = 100 * (accuracy(arms[arm], "overall") or 0)
            axes[2].scatter(x, y, marker=marker, s=66, color=COLORS[arm],
                            edgecolor="white", linewidth=0.8, zorder=3)
    arm_legend = axes[2].legend(
        handles=[Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                        markerfacecolor=COLORS[arm], markeredgecolor="white",
                        label=LABELS[arm]) for arm in ARMS],
        loc="upper left", ncols=2, frameon=False, fontsize=6.6,
        columnspacing=0.8, handletextpad=0.35)
    axes[2].add_artist(arm_legend)
    axes[2].legend(
        handles=[Line2D([0], [0], marker=marker, linestyle="none", markersize=6,
                        color="#59636E", label=f"{budget}-turn")
                 for budget, marker in ((32, "o"), (64, "s"))],
        loc="lower right", frameon=False, fontsize=6.6, handletextpad=0.35)
    axes[2].set_xlabel("Mean Answer Prompt Tokens")
    axes[2].set_ylabel("全量 Accuracy (%)")
    axes[2].set_title("(c) Token–Accuracy Pareto")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22)
    figure.tight_layout()
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(figures / f"v5_54_index_ablation.{suffix}", dpi=220,
                       bbox_inches="tight")
    plt.close(figure)

    analysis_parts = []
    for budget in (32, 64):
        block = payload["budgets"][str(budget)]
        arms = block["arms"]
        comparisons = block.get("comparisons", {})
        comparison = comparisons.get("seed_only->full", {}).get("hard869", {})
        hierarchy_effect = comparisons.get(
            "seed_only->hierarchy_only", {}).get("hard869", {})
        flat_relation_effect = comparisons.get(
            "seed_only->flat_graph", {}).get("hard869", {})
        conditional_relation_effect = comparisons.get(
            "hierarchy_only->full", {}).get("hard869", {})
        interaction = block.get("interaction", {}).get("hard869", {})
        ci = comparison.get("memory_cluster_bootstrap_95ci") or (None, None)
        type_deltas = {
            label: (accuracy(arms["full"], group) or 0)
            - (accuracy(arms["seed_only"], group) or 0)
            for group, label in GROUPS}
        best_type = max(type_deltas, key=type_deltas.get)
        worst_type = min(type_deltas, key=type_deltas.get)
        analysis_parts.append(
            f"{budget}-turn 下，Full GraphMem 相对 Seed-only 在 Hard-869 上由 "
            f"{pct(accuracy(arms['seed_only'], 'hard869'))}\\% 变为 "
            f"{pct(accuracy(arms['full'], 'hard869'))}\\%，变化 "
            f"{pp(comparison.get('delta'))} pp（修复/退化 "
            f"{comparison.get('gains', '待补')}/{comparison.get('losses', '待补')}，"
            f"McNemar $p={number(comparison.get('mcnemar_exact_p'), 3)}$，"
            f"Memory-cluster 95\\% CI {pp(ci[0])} 至 {pp(ci[1])} pp）。"
            f"关闭 relation 时的 hierarchy 主效应为 "
            f"{pp(hierarchy_effect.get('delta'))} pp，关闭 hierarchy 时的 flat relation "
            f"效应为 {pp(flat_relation_effect.get('delta'))} pp，层级条件下 relation 的"
            f"边际效应为 {pp(conditional_relation_effect.get('delta'))} pp，二者的 "
            f"difference-in-differences 为 "
            f"{pp(interaction.get('difference_in_differences'))} pp。"
            f"Full 相对 Seed 的最大/最小分题型变化分别是 {best_type} "
            f"{pp(type_deltas[best_type])} pp 与 {worst_type} "
            f"{pp(type_deltas[worst_type])} pp。")
    low = payload["budgets"]["32"]
    high = payload["budgets"]["64"]
    low_arms = low["arms"]
    high_arms = high["arms"]
    low_interaction = low.get("interaction", {}).get("hard869", {})
    high_interaction = high.get("interaction", {}).get("hard869", {})
    high_interaction_ci = high_interaction.get(
        "memory_cluster_bootstrap_95ci") or (None, None)
    low_temporal_delta = (
        (accuracy(low_arms["full"], "lme_temporal") or 0)
        - (accuracy(low_arms["seed_only"], "lme_temporal") or 0))
    high_flat_relation = high.get("comparisons", {}).get(
        "seed_only->flat_graph", {}).get("hard869", {}).get("delta")
    high_conditional_relation = high.get("comparisons", {}).get(
        "hierarchy_only->full", {}).get("hard869", {}).get("delta")
    interpretation = (
        "核心结果是层次路由与关系扩展的协同，而不是“增加关系边必然单调增益”："
        f"64-turn 下，flat relation 效应为 {pp(high_flat_relation)} pp，"
        f"在 hierarchy 条件下反转为 {pp(high_conditional_relation)} pp；"
        f"交互项为 {pp(high_interaction.get('difference_in_differences'))} pp"
        f"（Memory-cluster 95\\% CI {pp(high_interaction_ci[0])} 至 "
        f"{pp(high_interaction_ci[1])} pp），Full 同时取得四臂最高的 Hard-869 "
        f"{pct(accuracy(high_arms['full'], 'hard869'))}\\% 与全量 "
        f"{pct(accuracy(high_arms['full'], 'overall'))}\\%。"
        f"相反，32-turn 的交互项为 "
        f"{pp(low_interaction.get('difference_in_differences'))} pp，且 LME Temporal "
        f"变化为 {pp(low_temporal_delta)} pp，说明紧预算下未经充分容量承接的关系扩展"
        "会与关键证据竞争；因此索引优势依赖 coarse-to-fine 门控与足够的打包预算，"
        "不能归因于无约束扩边。")
    analysis = (
        r"\paragraph{查询侧 2$\times$2 消融。}"
        + " ".join(analysis_parts)
        + " " + interpretation
        + r" 所有臂共享同一 Safe-Witness 图和 V5.54 label-free readout，"
          r"因此索引路径变化与回答提示策略不再混合；32/64-turn 同时报告，"
          r"避免宽上下文饱和掩盖图索引的 Token--Accuracy 收益。"
        + "\n")
    (generated / "v5_54_index_ablation_analysis.tex").write_text(
        analysis, encoding="utf-8")
    sources = {
        "schema_version": "graphmem-v5.54-index-report-sources-v1",
        "summary": str(args.summary),
        "prepare_audit": payload.get("prepare_audit", {}),
        "final_audit": payload.get("final_audit", {}),
    }
    (generated / "v5_54_index_report_sources.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"table": str(generated / "v5_54_index_ablation_table.tex"),
                      "figure": str(figures / "v5_54_index_ablation.pdf")}, indent=2))


if __name__ == "__main__":
    main()
