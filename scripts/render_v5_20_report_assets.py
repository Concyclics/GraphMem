#!/usr/bin/env python3
"""Render V5.20 graph-mechanism and 32/64-turn report assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ARMS = (
    ("seed_only", "无关系扩展"),
    ("flat_graph", "平面关系图"),
    ("hierarchical", "分层自顶向下"),
    ("graph_rerank_layout", r"图重排 Layout"),
    ("topology_layout", r"完整图式 Prompt"),
)
GROUPS = ("lme_multi_session", "lme_temporal", "locomo_multihop",
          "locomo_temporal", "structural", "temporal", "overall")


def read(path: Path | None) -> dict:
    return (json.loads(path.read_text(encoding="utf-8"))
            if path is not None and path.exists() else {})


def tex_escape(value: str) -> str:
    return (value.replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def pct(value) -> str:
    return "运行中" if value is None else f"{100 * float(value):.1f}"


def delta(value) -> str:
    return "运行中" if value is None else f"{100 * float(value):+.1f}"


def number(value) -> str:
    if value is None:
        return "运行中"
    value = float(value)
    return f"{value:,.1f}" if value % 1 else f"{int(value):,}"


def token_quad(stats: dict | None) -> str:
    if not stats:
        return "运行中"
    return "/".join(number(stats.get(key)) for key in
                    ("mean", "p95", "p99", "max"))


def graph_table(payload: dict) -> str:
    arms = payload.get("arms", {})
    comparisons = payload.get("comparisons", {})
    previous = None
    output = []
    for key, label in ARMS:
        accuracy = arms.get(key, {}).get("accuracy", {})
        values = [pct(accuracy.get(group, {}).get("accuracy")) for group in GROUPS]
        if previous is None:
            change = "+0.0" if arms else "运行中"
            p_value = "1.000" if arms else "运行中"
        else:
            paired = comparisons.get(f"{previous}->{key}", {}).get("overall", {})
            change = delta(paired.get("delta"))
            raw_p = paired.get("mcnemar_exact_p")
            p_value = "运行中" if raw_p is None else f"{float(raw_p):.3f}"
        output.append(" & ".join((label, *values, change, p_value)) + r" \\")
        previous = key
    return "\n".join(output) + "\n\\bottomrule\n"


def graph_analysis(payload: dict) -> str:
    if not payload.get("arms"):
        return (r"\paragraph{结果状态。}四臂回答与 Luna judge 正在并行执行；"
                r"占位单元格不参与结论。" "\n")
    comparisons = payload.get("comparisons", {})
    final = comparisons.get("seed_only->topology_layout", {}).get("overall", {})
    ci = final.get("paired_bootstrap_95ci") or (None, None)
    p_value = final.get("mcnemar_exact_p")
    final_accuracy = payload["arms"]["topology_layout"]["accuracy"]["overall"]
    final_metrics = payload["arms"]["topology_layout"]["retrieval"]["overall"]
    prompt_control = payload.get("audit", {}).get("topology_prompt_control", {})
    strict = prompt_control.get("strict_same_set_accuracy_comparison", {})
    significant = p_value is not None and float(p_value) < 0.05
    wording = "达到统计显著" if significant else "尚未达到 $p<0.05$"
    strict_text = ""
    if prompt_control:
        strict_p = strict.get("mcnemar_exact_p")
        strict_text = (
            f"在严格 Prompt 组织对照中，{prompt_control.get('same_evidence_set', 0)}/"
            f"{prompt_control.get('questions_compared', 0)} 题的 evidence ID 集合完全相同，"
            f"其中 {prompt_control.get('reordered_same_set', 0)} 题只改变顺序与图标签；"
            f"该子集的准确率变化为 {delta(strict.get('delta'))} pp"
            + (f"（$p={float(strict_p):.3f}$）。" if strict_p is not None else "。")
            + f"其余 {prompt_control.get('changed_set_due_to_prompt_budget', 0)} 题因拓扑标签占用"
              "12K Prompt 预算而发生尾部证据替换，只计入完整机制结果，不计入纯布局结论。")
    return (
        r"\paragraph{结果解读。}相对无关系扩展，完整的分层图与拓扑证据编排在 200 题上变化 "
        f"{delta(final.get('delta'))} pp（95\\% CI {delta(ci[0])} 至 "
        f"{delta(ci[1])}，$p={float(p_value):.3f}$），{wording}；最终准确率为 "
        f"{pct(final_accuracy.get('accuracy'))}\\%。最终 evidence recall/precision/all-hit 为 "
        f"{pct(final_metrics.get('turn_recall'))}\\%/"
        f"{pct(final_metrics.get('turn_precision'))}\\%/"
        f"{pct(final_metrics.get('turn_all_hit'))}\\%。"
        r"我们仅把逐题显著性检验通过的变化称为准确率提升；访问范围与证据连通性的变化单独报告，"
        r"不以 recall 增大替代最终回答准确率。" + strict_text + "\n")


def budget_rows(payload: dict) -> str:
    rows = []
    graphmem = payload.get("graphmem", ())
    mem0 = payload.get("mem0", ())
    for row in graphmem:
        benchmark = ("LongMemEval" if row.get("benchmark") == "longmemeval"
                     else "LoCoMo Cat. 1--4")
        rows.append(" & ".join((
            "GraphMem", str(row.get("retrieval_setting")), benchmark,
            number(row.get("questions")), pct(row.get("accuracy")),
            token_quad(row.get("answer_tokens")),
            number(row.get("mean_packed_turns")),
        )) + r" \\")
    for row in mem0:
        benchmark = ("LongMemEval" if row.get("benchmark") == "longmemeval"
                     else "LoCoMo Cat. 1--4")
        rows.append(" & ".join((
            "Mem0", str(row.get("retrieval_setting")), benchmark,
            number(row.get("questions")), pct(row.get("accuracy")),
            token_quad(row.get("answer_tokens")), "--",
        )) + r" \\")
    if not rows:
        for method, setting in (("GraphMem", "32-turn"), ("GraphMem", "64-turn"),
                                ("Mem0", "top-50"), ("Mem0", "top-200")):
            for benchmark in ("LongMemEval", "LoCoMo Cat. 1--4"):
                rows.append(" & ".join((method, setting, benchmark,
                                        "运行中", "运行中", "运行中", "运行中"))
                            + r" \\")
    return "\n".join(rows) + "\n\\bottomrule\n"


def budget_analysis(payload: dict) -> str:
    graphmem = list(payload.get("graphmem", ()))
    if not graphmem or any(row.get("questions") not in (500, 1540)
                           for row in graphmem):
        return (r"\paragraph{结果状态。}GraphMem 32/64-turn 全量回答与 Luna judge 正在运行；"
                r"Mem0 top-50/top-200 仅在 GraphMem 全量题数审计通过后进行同图比较。" "\n")
    chunks = []
    for benchmark, label in (("longmemeval", "LongMemEval"),
                             ("locomo", "LoCoMo")):
        selected = {int(row["turn_budget"]): row for row in graphmem
                    if row.get("benchmark") == benchmark}
        if 32 not in selected or 64 not in selected:
            continue
        left, right = selected[32], selected[64]
        token_ratio = (float(right["answer_tokens"]["mean"])
                       / float(left["answer_tokens"]["mean"]))
        gain = 100 * (float(right["accuracy"]) - float(left["accuracy"]))
        chunks.append(
            f"{label} 的 32/64-turn 准确率为 {pct(left['accuracy'])}\\%/"
            f"{pct(right['accuracy'])}\\%，Answer Token mean 为 "
            f"{number(left['answer_tokens']['mean'])}/{number(right['answer_tokens']['mean'])}；"
            f"64-turn 以 {token_ratio:.2f}$\\times$ Token 换取 {gain:+.1f} pp")
    return (r"\paragraph{预算曲线。}" + "；".join(chunks) + "。"
            r"跨方法比较使用实测 Answer Token，而不是把 GraphMem turn 数与 Mem0 top-$k$"
            r"视为相同语义；构建 Token 仍在独立成本表中按 Memory 报告。" "\n")


def plot_graph(payload: dict, report: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphmem-v520-matplotlib")
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import font_manager  # noqa: PLC0415
    font = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.2,
                         "axes.axisbelow": True, "axes.unicode_minus": False})
    colors = {"navy": "#12304A", "blue": "#2378D7", "teal": "#18A999",
              "amber": "#F2A93B", "red": "#D84A4A", "gray": "#5F6B76"}
    arms = payload.get("arms", {})
    labels = [label for _, label in ARMS]
    x = list(range(len(labels)))
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    if arms:
        for group, label, color, marker in (
                ("overall", "总体", colors["navy"], "o"),
                ("structural", "结构题", colors["blue"], "s"),
                ("temporal", "Temporal", colors["amber"], "^")):
            values = [100 * arms[key]["accuracy"][group]["accuracy"]
                      for key, _ in ARMS]
            axes[0].plot(x, values, marker=marker, color=color, label=label,
                         linewidth=1.8)
        metric_specs = (
            ("turn_recall", "Recall", colors["blue"]),
            ("turn_all_hit", "All-hit", colors["teal"]),
            ("turn_precision", "Precision", colors["red"]),
        )
        for metric, label, color in metric_specs:
            values = [100 * arms[key]["retrieval"]["overall"][metric]
                      for key, _ in ARMS]
            axes[1].plot(x, values, marker="o", color=color, label=label,
                         linewidth=1.8)
        visited = [arms[key]["retrieval"]["overall"].get("visited_edges") or 0
                   for key, _ in ARMS]
        arranged = [
            (arms[key]["retrieval"]["overall"].get("evidence_chain_turns") or 0)
            + (arms[key]["retrieval"]["overall"].get("evidence_graph_turns") or 0)
            for key, _ in ARMS]
        axes[2].bar([value - 0.18 for value in x], visited, 0.36,
                    color=colors["blue"], label="访问关系边")
        axes[2].bar([value + 0.18 for value in x], arranged, 0.36,
                    color=colors["amber"], label="拓扑成组 turns")
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "实验运行中", transform=axis.transAxes,
                      ha="center", va="center", color=colors["gray"], fontsize=11)
    axes[0].set_title("(a) 最终回答准确率", fontweight="bold")
    axes[0].set_ylabel("Accuracy（%）")
    axes[1].set_title("(b) 证据质量", fontweight="bold")
    axes[1].set_ylabel("证据指标（%）")
    axes[2].set_title("(c) 图路径使用与证据成组", fontweight="bold")
    axes[2].set_ylabel("每题平均数量")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, frameon=False, fontsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("图结构如何转化为最终回答准确率", color=colors["navy"],
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(figures / f"v5_20_graph_ablation.{suffix}",
                       dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-ablation", type=Path)
    parser.add_argument("--budget-benchmark", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    graph = read(args.graph_ablation)
    budget = read(args.budget_benchmark)
    generated = args.report / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    outputs = {
        "v5_20_graph_ablation_table.tex": graph_table(graph),
        "v5_20_graph_ablation_analysis.tex": graph_analysis(graph),
        "v5_20_budget_table.tex": budget_rows(budget),
        "v5_20_budget_analysis.tex": budget_analysis(budget),
    }
    for name, content in outputs.items():
        (generated / name).write_text(content, encoding="utf-8")
    plot_graph(graph, args.report)
    sources = {
        "schema_version": "graphmem-v5.20-report-assets-v1",
        "graph_ablation": str(args.graph_ablation) if args.graph_ablation else None,
        "budget_benchmark": str(args.budget_benchmark) if args.budget_benchmark else None,
        "graph_loaded": bool(graph), "budget_loaded": bool(budget),
        "missing_values_render_as_zero": False,
    }
    (generated / "v5_20_report_sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sources, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
