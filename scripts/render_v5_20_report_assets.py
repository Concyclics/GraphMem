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


def compact_number(value) -> str:
    if value is None:
        return "运行中"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return number(value)


def compact_token_quad(stats: dict | None) -> str:
    if not stats:
        return "运行中"
    return "/".join(compact_number(stats.get(key)) for key in
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
    questions = int(payload.get("protocol", {}).get("questions") or 0)
    hierarchy = comparisons.get("flat_graph->hierarchical", {}).get("overall", {})
    hierarchy_ci = hierarchy.get("paired_bootstrap_95ci") or (None, None)
    hierarchy_p = hierarchy.get("mcnemar_exact_p")
    hierarchy_wording = (
        "达到统计显著" if hierarchy_p is not None and float(hierarchy_p) < 0.05
        else "尚未达到 $p<0.05$")
    final = comparisons.get("seed_only->topology_layout", {}).get("overall", {})
    final_p = final.get("mcnemar_exact_p")
    best_accuracy = payload["arms"]["hierarchical"]["accuracy"]["overall"]
    best_metrics = payload["arms"]["hierarchical"]["retrieval"]["overall"]
    rerank_groups = comparisons.get(
        "hierarchical->graph_rerank_layout", {}).get("by_group", {})
    prompt_control = payload.get("audit", {}).get("topology_prompt_control", {})
    strict = prompt_control.get("strict_same_set_accuracy_comparison", {})
    strict_text = ""
    if prompt_control:
        strict_p = strict.get("mcnemar_exact_p")
        strict_text = (
            f"在严格 Prompt 组织对照中，{prompt_control.get('same_evidence_set', 0)}/"
            f"{prompt_control.get('questions_compared', 0)} 题的 evidence ID 集合完全相同，"
            "这些题保持同一拓扑顺序，只增加图路径标签与读取指令；"
            f"该子集的准确率变化为 {delta(strict.get('delta'))} pp"
            + (f"（$p={float(strict_p):.3f}$）。" if strict_p is not None else "。")
            + f"其余 {prompt_control.get('changed_set_due_to_prompt_budget', 0)} 题因拓扑标签占用"
              "12K Prompt 预算而发生尾部证据替换，只计入完整机制结果，不计入纯布局结论。")
    group_text = []
    for group, label in (("lme_multi_session", "LME Multi-session"),
                         ("lme_temporal", "LME Temporal"),
                         ("locomo_multihop", "LoCoMo Multi-hop"),
                         ("locomo_temporal", "LoCoMo Temporal")):
        group_text.append(
            f"{label} {delta(rerank_groups.get(group, {}).get('delta'))} pp")
    return (
        r"\paragraph{结果解读。}在 " + str(questions)
        + r" 题完整分层上，平面关系图加入分层路由后变化 "
        f"{delta(hierarchy.get('delta'))} pp（95\\% CI "
        f"{delta(hierarchy_ci[0])} 至 {delta(hierarchy_ci[1])}，"
        f"$p={float(hierarchy_p):.3f}$），{hierarchy_wording}；分层臂准确率为 "
        f"{pct(best_accuracy.get('accuracy'))}\\%。其 evidence recall/precision/all-hit 为 "
        f"{pct(best_metrics.get('turn_recall'))}\\%/"
        f"{pct(best_metrics.get('turn_precision'))}\\%/"
        f"{pct(best_metrics.get('turn_all_hit'))}\\%。图重排相对分层路由的分题型变化为 "
        + "、".join(group_text) + "，说明统一重排会把 LME 收益与 LoCoMo 损失相互抵消。"
        f"完整图式 Prompt 相对 Seed-only 总体为 {delta(final.get('delta'))} pp"
        f"（$p={float(final_p):.3f}$）。"
        r"我们仅把逐题显著性检验通过的变化称为准确率提升；访问范围与证据连通性的变化单独报告，"
        r"不以 recall 增大替代最终回答准确率。" + strict_text + "\n")


def indexed_results(payload: dict) -> tuple[dict, dict]:
    graph = {
        (row.get("benchmark"), row.get("retrieval_setting")): row
        for row in payload.get("graphmem", ())
    }
    mem0 = {
        (row.get("benchmark"), row.get("retrieval_setting")): row
        for row in payload.get("mem0", ())
        if row.get("status") == "complete" and row.get("questions")
    }
    return graph, mem0


def build_rows(payload: dict) -> str:
    graph, mem0 = indexed_results(payload)
    rows = []
    for benchmark, label, unit in (
            ("longmemeval", "LongMemEval", "Memory"),
            ("locomo", "LoCoMo Cat. 1--4", "Conversation")):
        left = graph.get((benchmark, "32-turn"))
        right = mem0.get((benchmark, "top-50"))
        if (not left or not right or not left.get("build_tokens")
                or not right.get("build_tokens")):
            rows.append(" & ".join((label, unit, "运行中", "运行中",
                                    "运行中", "运行中")) + r" \\")
            continue
        graph_build = left["build_tokens"]
        mem0_build = right["build_tokens"]
        ratio = float(mem0_build["mean"]) / float(graph_build["mean"])
        reduction = 100 * (1 - float(graph_build["mean"])
                           / float(mem0_build["mean"]))
        rows.append(" & ".join((
            label, f"{unit}（{number(graph_build.get('count'))}）",
            compact_token_quad(graph_build), compact_token_quad(mem0_build),
            f"{ratio:.1f}$\\times$", f"{reduction:.1f}\\%",
        )) + r" \\")
    return "\n".join(rows) + "\n\\bottomrule\n"


def budget_rows(payload: dict) -> str:
    graph, mem0 = indexed_results(payload)
    pairs = (
        ("longmemeval", "LongMemEval", "低预算", "32-turn", "top-50",
         "近似等预算"),
        ("longmemeval", "LongMemEval", "高预算", "64-turn", "top-200",
         "近似等预算"),
        ("locomo", "LoCoMo Cat. 1--4", "等预算", "64-turn", "top-50",
         "近似等预算"),
        ("locomo", "LoCoMo Cat. 1--4", "Pareto 支配", "64-turn", "top-200",
         "更低预算"),
    )
    rows = []
    for benchmark, label, tier, graph_setting, mem0_setting, scope in pairs:
        left = graph.get((benchmark, graph_setting))
        right = mem0.get((benchmark, mem0_setting))
        if not left or not right:
            rows.append(" & ".join((label, tier, "运行中", "运行中",
                                    "运行中", "运行中", "运行中")) + r" \\")
            continue
        graph_tokens = float(left["answer_tokens"]["mean"])
        mem0_tokens = float(right["answer_tokens"]["mean"])
        gain = 100 * (float(left["accuracy"]) - float(right["accuracy"]))
        rows.append(" & ".join((
            label, tier, f"{graph_setting} / {mem0_setting}",
            f"{number(graph_tokens)} / {number(mem0_tokens)}",
            f"{graph_tokens / mem0_tokens:.2f}$\\times$（{scope}）",
            f"{pct(left['accuracy'])} / {pct(right['accuracy'])}",
            f"{gain:+.1f} pp",
        )) + r" \\")
    return "\n".join(rows) + "\n\\bottomrule\n"


def type_accuracy_rows(payload: dict) -> str:
    graph, mem0 = indexed_results(payload)
    specifications = (
        ("longmemeval", "LongMemEval", "single-session-user", "Single-session User"),
        ("longmemeval", "", "single-session-assistant", "Single-session Assistant"),
        ("longmemeval", "", "single-session-preference", "Single-session Preference"),
        ("longmemeval", "", "multi-session", "Multi-session"),
        ("longmemeval", "", "temporal-reasoning", "Temporal-reasoning"),
        ("longmemeval", "", "knowledge-update", "Knowledge-update"),
        ("locomo", "LoCoMo", "category_1", "Category 1 Multi-hop"),
        ("locomo", "", "category_2", "Category 2 Temporal"),
        ("locomo", "", "category_3", "Category 3 Open-domain"),
        ("locomo", "", "category_4", "Category 4 Single-hop"),
    )
    rows = []
    previous_benchmark = None
    for benchmark, benchmark_label, key, type_label in specifications:
        if previous_benchmark is not None and benchmark != previous_benchmark:
            rows.append(r"\midrule")
        points = (
            graph.get((benchmark, "32-turn"), {}),
            graph.get((benchmark, "64-turn"), {}),
            mem0.get((benchmark, "top-50"), {}),
            mem0.get((benchmark, "top-200"), {}),
        )
        values = [(point.get("accuracy_by_type") or {}).get(key, {})
                  for point in points]
        sample_sizes = {int(value.get("questions")) for value in values
                        if value.get("questions") is not None}
        sample = next(iter(sample_sizes)) if len(sample_sizes) == 1 else None
        accuracies = [value.get("accuracy") for value in values]
        if benchmark == "longmemeval" and all(value is not None for value in accuracies):
            matched_delta = (
                f"{100 * (accuracies[0] - accuracies[2]):+.1f} / "
                f"{100 * (accuracies[1] - accuracies[3]):+.1f} pp")
        elif benchmark == "locomo" and all(value is not None for value in accuracies):
            matched_delta = f"{100 * (accuracies[1] - accuracies[2]):+.1f} pp"
        else:
            matched_delta = "运行中"
        rows.append(" & ".join((
            benchmark_label, type_label, number(sample),
            *(pct(value) for value in accuracies), matched_delta,
        )) + r" \\")
        previous_benchmark = benchmark
    return "\n".join(rows) + "\n\\bottomrule\n"


def type_accuracy_analysis(payload: dict) -> str:
    graph, mem0 = indexed_results(payload)
    required = (
        graph.get(("longmemeval", "32-turn")),
        graph.get(("longmemeval", "64-turn")),
        mem0.get(("longmemeval", "top-50")),
        mem0.get(("longmemeval", "top-200")),
        graph.get(("locomo", "64-turn")),
        mem0.get(("locomo", "top-50")),
    )
    if not all(required):
        return (r"\paragraph{题型结果状态。}逐题类型账本仍在审计；"
                r"缺失单元格不参与题型结论。" "\n")

    def typed(row: dict, key: str) -> float:
        return float(row["accuracy_by_type"][key]["accuracy"])

    lme32, lme64, mem50, mem200, locomo64, locomo50 = required
    lme_temporal_low = 100 * (
        typed(lme32, "temporal-reasoning")
        - typed(mem50, "temporal-reasoning"))
    lme_temporal_high = 100 * (
        typed(lme64, "temporal-reasoning")
        - typed(mem200, "temporal-reasoning"))
    lme_multi_low = 100 * (
        typed(lme32, "multi-session") - typed(mem50, "multi-session"))
    lme_multi_high = 100 * (
        typed(lme64, "multi-session") - typed(mem200, "multi-session"))
    locomo_deltas = {
        category: 100 * (typed(locomo64, category) - typed(locomo50, category))
        for category in ("category_1", "category_2", "category_3", "category_4")
    }
    return (
        r"\paragraph{题型分解。}"
        f"LongMemEval 的近似等预算增益主要来自 Temporal-reasoning："
        f"低/高预算分别为 {lme_temporal_low:+.1f}/{lme_temporal_high:+.1f} pp；"
        f"Multi-session 仅为 {lme_multi_low:+.1f}/{lme_multi_high:+.1f} pp，"
        r"说明跨会话集合闭包仍是当前弱项。"
        f"LoCoMo 64-turn 对 top-50 的总体优势并非各题型均匀提升："
        f"Category 2 Temporal 为 {locomo_deltas['category_2']:+.1f} pp，"
        f"Category 3 Open-domain 为 {locomo_deltas['category_3']:+.1f} pp，"
        f"但 Category 1 Multi-hop 与 Category 4 Single-hop 分别为 "
        f"{locomo_deltas['category_1']:+.1f}/{locomo_deltas['category_4']:+.1f} pp。"
        r"因此总体提升应解释为时间关系建模的集中收益，而不是所有查询类型上的全面支配；"
        r"后两类仍是下一轮索引与回答路径优化的直接目标。" "\n")


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
    mem0 = {
        (row.get("benchmark"), row.get("retrieval_setting")): row
        for row in payload.get("mem0", ())
        if row.get("status") == "complete" and row.get("questions")
    }
    matched = []
    build_matched = []
    graph = {(row.get("benchmark"), row.get("retrieval_setting")): row
             for row in graphmem}
    pairs = (
        (("longmemeval", "32-turn"), ("longmemeval", "top-50"),
         "LongMemEval 低预算"),
        (("longmemeval", "64-turn"), ("longmemeval", "top-200"),
         "LongMemEval 高预算"),
        (("locomo", "64-turn"), ("locomo", "top-50"),
         "LoCoMo 近似等预算"),
        (("locomo", "64-turn"), ("locomo", "top-200"),
         "LoCoMo Pareto 点"),
    )
    for graph_key, mem0_key, label in pairs:
        if graph_key not in graph or mem0_key not in mem0:
            continue
        left, right = graph[graph_key], mem0[mem0_key]
        accuracy_gain = 100 * (float(left["accuracy"]) - float(right["accuracy"]))
        token_change = 100 * (float(left["answer_tokens"]["mean"])
                              / float(right["answer_tokens"]["mean"]) - 1)
        matched.append(
            f"{label}相对 Mem0 为 {accuracy_gain:+.1f} pp Accuracy、"
            f"{token_change:+.1f}\\% Answer Token")
    for benchmark, label in (("longmemeval", "LongMemEval"),
                             ("locomo", "LoCoMo")):
        graph_row = graph.get((benchmark, "32-turn"))
        mem0_row = mem0.get((benchmark, "top-50"))
        if not graph_row or not mem0_row:
            continue
        graph_build = graph_row.get("build_tokens") or {}
        mem0_build = mem0_row.get("build_tokens") or {}
        if graph_build.get("mean") is None or mem0_build.get("mean") is None:
            continue
        ratio = float(mem0_build["mean"]) / float(graph_build["mean"])
        reduction = 100 * (1 - float(graph_build["mean"])
                           / float(mem0_build["mean"]))
        build_matched.append(
            f"{label} 的 GraphMem/Mem0 Build mean 为 "
            f"{compact_number(graph_build['mean'])}/"
            f"{compact_number(mem0_build['mean'])}，即 {ratio:.1f}$\\times$ "
            f"差距（GraphMem 减少 {reduction:.1f}\\%）")
    comparison = ("；" + "；".join(matched) if matched else "")
    build_comparison = ("；" + "；".join(build_matched)
                        if build_matched else "")
    return (r"\paragraph{预算曲线。}" + "；".join(chunks) + comparison
            + build_comparison + "。"
            r"跨方法比较使用实测 Answer Token，而不是把 GraphMem turn 数与 Mem0 top-$k$"
            r"视为相同语义；Build Token 在 32/64-turn 或 top-50/top-200 间共享，"
            r"表中重复展示仅为便于逐行对照，不重复计费。" "\n")


def plot_style():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphmem-v520-matplotlib")
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import font_manager  # noqa: PLC0415
    font = Path(
        "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/"
        "FandolHei-Regular.otf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams.update({
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.2,
        "axes.axisbelow": True, "axes.unicode_minus": False,
    })
    colors = {
        "navy": "#12304A", "blue": "#2378D7", "blue_light": "#68A9E6",
        "teal": "#18A999", "amber": "#F2A93B", "amber_dark": "#D98218",
        "red": "#D84A4A", "gray": "#5F6B76",
    }
    return plt, colors


def save_budget_figure(figure, report: Path, stem: str) -> None:
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        path = figures / f"{stem}.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        if suffix == "svg":
            # Matplotlib leaves spaces at SVG path line breaks.  Normalize the
            # generated source so repository whitespace checks stay clean.
            text = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
                encoding="utf-8")


def finish_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_build_tokens(payload: dict, report: Path) -> None:
    plt, colors = plot_style()
    from matplotlib.patches import Patch  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter  # noqa: PLC0415

    graph, mem0 = indexed_results(payload)
    statistics = ("mean", "p95", "p99", "max")
    labels = ("Mean", "p95", "p99", "Max")
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 3.8))
    for axis, (benchmark, title) in zip(
            axes, (("longmemeval", "(a) LongMemEval"),
                   ("locomo", "(b) LoCoMo Cat. 1--4")), strict=True):
        left = graph.get((benchmark, "32-turn"), {}).get("build_tokens")
        right = mem0.get((benchmark, "top-50"), {}).get("build_tokens")
        if not left or not right:
            axis.text(0.5, 0.5, "实验运行中", transform=axis.transAxes,
                      ha="center", va="center", color=colors["gray"])
            axis.set_title(title, fontweight="bold")
            continue
        x = list(range(len(statistics)))
        width = 0.34
        graph_values = [float(left[key]) for key in statistics]
        mem0_values = [float(right[key]) for key in statistics]
        graph_bars = axis.bar(
            [value - width / 2 for value in x], graph_values, width,
            color=colors["blue"], edgecolor="white", label="GraphMem")
        mem0_bars = axis.bar(
            [value + width / 2 for value in x], mem0_values, width,
            color=colors["amber"], edgecolor="white", label="Mem0")
        axis.set_yscale("log")
        axis.set_xticks(x, labels)
        axis.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: compact_number(value)))
        axis.set_xlabel("构建开支统计量")
        axis.set_ylabel("Build Token / owner（log scale）")
        ratio = mem0_values[0] / graph_values[0]
        reduction = 100 * (1 - graph_values[0] / mem0_values[0])
        axis.set_title(
            f"{title}\nMean：Mem0/GraphMem = {ratio:.1f}×；"
            f"GraphMem -{reduction:.1f}%",
            fontweight="bold", fontsize=10.5, pad=7)
        for bars, values in ((graph_bars, graph_values),
                             (mem0_bars, mem0_values)):
            for bar, value in zip(bars, values, strict=True):
                axis.annotate(
                    compact_number(value),
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
        finish_axes(axis)
    figure.suptitle("GraphMem 与 Mem0 的构建 Token 开支", color=colors["navy"],
                    fontsize=14, fontweight="bold")
    figure.legend(
        handles=(Patch(facecolor=colors["blue"], label="GraphMem"),
                 Patch(facecolor=colors["amber"], label="Mem0")),
        loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2,
        frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    save_budget_figure(figure, report, "v5_20_build_tokens")
    plt.close(figure)


def plot_budget_accuracy(payload: dict, report: Path) -> None:
    plt, colors = plot_style()
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter  # noqa: PLC0415

    graph, mem0 = indexed_results(payload)
    statistics = ("mean", "p95", "p99", "max")
    markers = ("o", "s", "^", "D")
    series = (
        ("GraphMem 32-turn", graph, "32-turn", colors["blue"], "--"),
        ("GraphMem 64-turn", graph, "64-turn", colors["blue"], "-"),
        ("Mem0 top-50", mem0, "top-50", colors["amber_dark"], "--"),
        ("Mem0 top-200", mem0, "top-200", colors["amber_dark"], "-"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.15))
    for axis, (benchmark, title) in zip(
            axes, (("longmemeval", "(a) LongMemEval"),
                   ("locomo", "(b) LoCoMo Cat. 1--4")), strict=True):
        observed = []
        for label, source, setting, color, linestyle in series:
            row = source.get((benchmark, setting))
            if not row or not row.get("answer_tokens") or row.get("accuracy") is None:
                continue
            x = [float(row["answer_tokens"][key]) for key in statistics]
            y = [100 * float(row["accuracy"])] * len(statistics)
            observed.extend(y)
            axis.plot(x, y, color=color, linestyle=linestyle, linewidth=1.8,
                      alpha=0.94, label=label)
            for token, accuracy, marker in zip(x, y, markers, strict=True):
                axis.scatter(token, accuracy, marker=marker, s=42,
                             color=color, edgecolor="white", linewidth=0.7,
                             zorder=3)
        if not observed:
            axis.text(0.5, 0.5, "实验运行中", transform=axis.transAxes,
                      ha="center", va="center", color=colors["gray"])
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Answer total Token")
        axis.set_ylabel("Accuracy（%）")
        axis.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: compact_number(value)))
        axis.set_ylim(50, 80)
        finish_axes(axis)
    series_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.8,
               label=label)
        for label, _, _, color, linestyle in series
    ]
    marker_handles = [
        Line2D([0], [0], color=colors["gray"], marker=marker,
               linestyle="None", markersize=6, label=label)
        for marker, label in zip(markers, ("Mean", "p95", "p99", "Max"),
                                 strict=True)
    ]
    figure.suptitle("Answer Token 预算--准确率工作点", color=colors["navy"],
                    fontsize=14, fontweight="bold")
    figure.legend(series_handles, [item.get_label() for item in series_handles],
                  loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4,
                  frameon=False, fontsize=8)
    figure.legend(marker_handles, [item.get_label() for item in marker_handles],
                  loc="lower center", bbox_to_anchor=(0.5, 0.00), ncol=4,
                  frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.09, 1, 0.84))
    save_budget_figure(figure, report, "v5_20_budget_accuracy")
    plt.close(figure)


def plot_type_accuracy(payload: dict, report: Path) -> None:
    plt, colors = plot_style()
    from matplotlib.patches import Patch  # noqa: PLC0415

    graph, mem0 = indexed_results(payload)
    specifications = (
        ("longmemeval", "single-session-user", "LME · User"),
        ("longmemeval", "single-session-assistant", "LME · Assistant"),
        ("longmemeval", "single-session-preference", "LME · Preference"),
        ("longmemeval", "multi-session", "LME · Multi-session"),
        ("longmemeval", "temporal-reasoning", "LME · Temporal"),
        ("longmemeval", "knowledge-update", "LME · Knowledge-update"),
        ("locomo", "category_1", "LoCoMo · Cat.1 Multi-hop"),
        ("locomo", "category_2", "LoCoMo · Cat.2 Temporal"),
        ("locomo", "category_3", "LoCoMo · Cat.3 Open-domain"),
        ("locomo", "category_4", "LoCoMo · Cat.4 Single-hop"),
    )
    configurations = (
        (graph, "32-turn", "G32", colors["blue_light"]),
        (graph, "64-turn", "G64", colors["blue"]),
        (mem0, "top-50", "M50", colors["amber"]),
        (mem0, "top-200", "M200", colors["amber_dark"]),
    )
    figure, axes = plt.subplots(2, 5, figsize=(13.1, 6.4), sharey=True)
    for index, (axis, (benchmark, type_key, title)) in enumerate(
            zip(axes.flat, specifications, strict=True)):
        rows = [source.get((benchmark, setting), {})
                for source, setting, _, _ in configurations]
        accuracies = [
            (row.get("accuracy_by_type") or {}).get(type_key, {}).get("accuracy")
            for row in rows
        ]
        low_token_configurations = {0, 2}
        for bar_index, (accuracy, (_, _, label, color)) in enumerate(
                zip(accuracies, configurations, strict=True)):
            if accuracy is None:
                continue
            value = 100 * float(accuracy)
            hatch = "////" if bar_index in low_token_configurations else None
            edgecolor = colors["navy"] if hatch else "white"
            axis.bar(bar_index, value, width=0.78, color=color,
                     hatch=hatch, edgecolor=edgecolor, linewidth=0.65)
            axis.text(bar_index, min(98, value + 2.0), f"{value:.1f}",
                      ha="center", va="bottom", fontsize=8)
        axis.set_title(f"({chr(97 + index)}) {title}", fontsize=11,
                       fontweight="bold", pad=5)
        axis.set_xticks(range(4), [item[2] for item in configurations],
                        fontsize=8.5)
        axis.set_ylim(0, 105)
        axis.set_yticks((0, 25, 50, 75, 100))
        axis.tick_params(axis="y", labelsize=8.5)
        if index % 5 == 0:
            axis.set_ylabel("Accuracy（%）", fontsize=10)
        finish_axes(axis)
    handles = [
        Patch(facecolor=color, edgecolor="white", label=label)
        for _, _, label, color in configurations
    ] + [
        Patch(facecolor="white", edgecolor=colors["navy"], hatch="////",
              label="低 Token 配置（32-turn / top-50）")
    ]
    figure.suptitle("按题型细分的端到端准确率", color=colors["navy"],
                    fontsize=16, fontweight="bold")
    figure.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.5, 0.005), ncol=5, frameon=False,
                  fontsize=9)
    figure.tight_layout(rect=(0, 0.08, 1, 0.92), h_pad=1.7, w_pad=0.7)
    save_budget_figure(figure, report, "v5_20_type_accuracy")
    plt.close(figure)


def plot_graph(payload: dict, report: Path) -> None:
    plt, colors = plot_style()
    arms = payload.get("arms", {})
    labels = [label for _, label in ARMS]
    x = list(range(len(labels)))
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    if arms:
        for group, label, color, marker in (
                ("overall", "总体", colors["navy"], "o"),
                ("lme_multi_session", "LME Multi-session", colors["blue"], "s"),
                ("lme_temporal", "LME Temporal", colors["teal"], "^"),
                ("locomo_multihop", "LoCoMo Multi-hop", colors["amber"], "D"),
                ("locomo_temporal", "LoCoMo Temporal", colors["red"], "v")):
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
    axes[0].set_title("(a) 四题型与总体准确率", fontweight="bold")
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
    save_budget_figure(figure, report, "v5_20_graph_ablation")
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
        "v5_20_build_table.tex": build_rows(budget),
        "v5_20_budget_table.tex": budget_rows(budget),
        "v5_20_type_accuracy_table.tex": type_accuracy_rows(budget),
        "v5_20_type_accuracy_analysis.tex": type_accuracy_analysis(budget),
        "v5_20_budget_analysis.tex": budget_analysis(budget),
    }
    for name, content in outputs.items():
        (generated / name).write_text(content, encoding="utf-8")
    plot_graph(graph, args.report)
    plot_build_tokens(budget, args.report)
    plot_budget_accuracy(budget, args.report)
    plot_type_accuracy(budget, args.report)
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
