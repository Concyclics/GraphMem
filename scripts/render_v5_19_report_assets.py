#!/usr/bin/env python3
"""Render V5.19 manifest-backed LaTeX tables and the attribute-edge figure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ARMS = (
    ("full", "Full"), ("no_scene", "No-Scene"),
    ("no_entity_family", "No-Entity-family"),
    ("no_temporal", "No-Temporal"), ("no_lexical", "No-Lexical"),
    ("semantic_only", "Semantic-only"),
)
GROUPS = ("lme_multi_session", "lme_temporal", "locomo_multihop",
          "locomo_temporal", "structural", "temporal")
MODEL_LABELS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": "Qwen3-30B",
    "Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B (BF16)",
    "gpt-5.4-mini": "GPT-5.4-mini",
}
DISPLAYABLE_STATUSES = {"complete", "supplied"}


def read(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value) -> str:
    return "待补" if value is None else f"{100 * float(value):.1f}"


def delta(value) -> str:
    return "待补" if value is None else f"{100 * float(value):+.1f}"


def number(value) -> str:
    if value is None:
        return "待补"
    value = float(value)
    return f"{value:,.1f}" if value % 1 else f"{int(value):,}"


def tex_escape(value: str) -> str:
    return (value.replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def render_ablation_table(payload: dict) -> str:
    rows = []
    arms = payload.get("arms", {})
    for key, label in ARMS:
        arm = arms.get(key, {})
        groups = arm.get("accuracy", {}).get("by_group", {})
        values = [pct(groups.get(group, {}).get("accuracy")) for group in GROUPS]
        paired = arm.get("paired_vs_full", {})
        p_value = paired.get("mcnemar_exact_p")
        rows.append(" & ".join((label, *values,
                                delta(paired.get("delta")),
                                "待补" if p_value is None else f"{p_value:.3f}"))
                    + r" \\")
    # Keep the trailing booktabs rule inside the input file.  A \bottomrule
    # immediately following \input can be seen before TeX has closed the last
    # alignment row and produces a misleading ``Misplaced \noalign`` error.
    return "\n".join(rows) + "\n\\bottomrule\n"


def token_quad(stats: dict | None) -> str:
    if not stats:
        return "待补"
    return "/".join(number(stats.get(key)) for key in
                    ("mean", "p95", "p99", "max"))


def render_main_table(payload: dict) -> str:
    rows = []
    for row in payload.get("rows", ()): 
        method = str(row.get("method") or "")
        model = MODEL_LABELS.get(str(row.get("answer_model")),
                                 str(row.get("answer_model") or ""))
        benchmark = ("LongMemEval" if row.get("benchmark") == "longmemeval"
                     else "LoCoMo Cat. 1--4")
        retrieval_setting = str(row.get("retrieval_setting") or "待补")
        status = str(row.get("status") or "pending")
        supplied = status == "supplied"
        if status not in DISPLAYABLE_STATUSES:
            missing = "运行中" if method == "GraphMem" and model == "GPT-5.4-mini" \
                else "待补"
            question = accuracy = answer = missing
        else:
            question = number(row.get("questions"))
            accuracy = pct(row.get("accuracy"))
            answer = token_quad(row.get("answer_tokens"))
        if supplied:
            method += "*"
        build = token_quad(row.get("build_tokens"))
        if method == "Mem0" and row.get("build_tokens") is None:
            build = "待补"
        rows.append(" & ".join(map(tex_escape, (
            method, model, benchmark, retrieval_setting, question, accuracy,
            build, answer)))
                    + r" \\")
    if not rows:
        for method, model in (("GraphMem", "Qwen3-30B"),
                              ("GraphMem", "GPT-5.4-mini"),
                              ("Mem0", "Qwen3-30B"),
                              ("Mem0", "GPT-5.4-mini")):
            for benchmark in ("LongMemEval", "LoCoMo Cat. 1--4"):
                state = "运行中" if method == "GraphMem" and model == "GPT-5.4-mini" else "待补"
                rows.append(" & ".join((method, model, benchmark, "待补", state,
                                        state, "待补", state)) + r" \\")
    return "\n".join(rows) + "\n\\bottomrule\n"


def render_ablation_analysis(payload: dict) -> str:
    arms = payload.get("arms", {})
    if not arms:
        return (r"\paragraph{结果状态。}六臂协议、统计单位和图表接口已冻结；"
                r"当前未导入通过契约审计的 Luna verdict，故不对占位单元格作数值解释。" "\n")
    full_groups = arms.get("full", {}).get("accuracy", {}).get("by_group", {})
    full_retrieval = arms.get("full", {}).get("retrieval", {})
    text = [
        r"\paragraph{结果解读。}Full 在结构题、Temporal 与全部 200 题上的准确率分别为 "
        f"{pct(full_groups.get('structural', {}).get('accuracy'))}\\%、"
        f"{pct(full_groups.get('temporal', {}).get('accuracy'))}\\% 和 "
        f"{pct(full_groups.get('overall', {}).get('accuracy'))}\\%。"
        "其最终 evidence recall/precision 为 "
        f"{pct(full_retrieval.get('turn_recall'))}\\%/"
        f"{pct(full_retrieval.get('turn_precision'))}\\%。"
    ]
    comparisons = []
    for key, label in ARMS[1:]:
        arm = arms.get(key, {})
        paired = arm.get("paired_vs_full", {})
        ci = paired.get("paired_bootstrap_95ci")
        comparisons.append(
            f"{label}: {delta(paired.get('delta'))} pp"
            + (f"（95\\% CI {delta(ci[0])} 至 {delta(ci[1])}，"
               f"$p={float(paired.get('mcnemar_exact_p')):.3f}$）"
               if ci and paired.get("mcnemar_exact_p") is not None else "（待补）"))
    text.append("相对 Full 的逐题转换为：" + "；".join(comparisons) + "。"
                "准确率变化必须与候选/最终 precision、物化边和实际 traversal 联合解释，"
                "不能把全量召回带来的 recall 增益直接归因于关系边质量。")
    return "".join(text) + "\n"


def render_main_analysis(payload: dict) -> str:
    rows = list(payload.get("rows", ()))
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return (r"\paragraph{结果状态。}全量表仅接受同时带配置哈希、artifact 路径、"
                r"完整题数与 Luna verdict 的 manifest；当前数值尚未满足该契约。" "\n")
    graphmem = {(str(row.get("answer_model")), str(row.get("benchmark"))): row
                for row in complete if row.get("method") == "GraphMem"}
    clauses = []
    for model, label in (("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "Qwen3-30B"),
                         ("gpt-5.4-mini", "GPT-5.4-mini")):
        lme = graphmem.get((model, "longmemeval"))
        locomo = graphmem.get((model, "locomo"))
        if lme and locomo:
            clauses.append(
                f"{label} 在 LongMemEval/LoCoMo 上分别达到 "
                f"{pct(lme.get('accuracy'))}\\%/{pct(locomo.get('accuracy'))}\\%，"
                "回答 total Token mean/p95/p99/max 分别为 "
                f"{token_quad(lme.get('answer_tokens'))} 与 "
                f"{token_quad(locomo.get('answer_tokens'))}")
    mem0_qwen = {(str(row.get("benchmark")), int(row.get("cutoff"))): row
                 for row in complete if row.get("method") == "Mem0"
                 and row.get("answer_model") == "Qwen3-30B-A3B-Instruct-2507"
                 and row.get("cutoff") in (50, 200)}
    for benchmark, label in (("longmemeval", "LongMemEval"),
                             ("locomo", "LoCoMo Category 1--4")):
        top50 = mem0_qwen.get((benchmark, 50))
        top200 = mem0_qwen.get((benchmark, 200))
        if top50 and top200:
            ratio = (float(top200["answer_tokens"]["mean"])
                     / float(top50["answer_tokens"]["mean"]))
            gain = 100 * (float(top200["accuracy"]) - float(top50["accuracy"]))
            clauses.append(
                f"Mem0 Qwen3-30B 在 {label} 的 top-50/top-200 准确率为 "
                f"{pct(top50.get('accuracy'))}\\%/{pct(top200.get('accuracy'))}\\%，"
                f"回答 total Token mean 为 {number(top50['answer_tokens']['mean'])}/"
                f"{number(top200['answer_tokens']['mean'])}；top-200 以 {ratio:.2f}$\\times$ "
                f"Token 换取 {gain:+.2f} pp")
    pending_gpt = any(row.get("method") == "GraphMem"
                      and row.get("answer_model") == "gpt-5.4-mini"
                      and row.get("status") != "complete" for row in rows)
    pending_mem0 = any(row.get("method") == "Mem0"
                       and row.get("status") not in DISPLAYABLE_STATUSES
                       for row in rows)
    suffix = []
    if pending_gpt:
        suffix.append("GPT-5.4-mini 仍在运行")
    if pending_mem0:
        suffix.append("Mem0 GPT-5.4-mini 的 top-50/top-200 数据待补")
    body = "；".join(clauses)
    if body:
        body += "。"
    if suffix:
        body += "；".join(suffix) + "，暂不进行跨方法优势声明。"
    baseline = payload.get("mem0_baseline_contract") or {}
    audit = baseline.get("audit") or {}
    if mem0_qwen:
        lme_failed = (audit.get("longmemeval") or {}).get("final_failed_pairs", 0)
        locomo_rows = [row for (bench, _), row in mem0_qwen.items()
                       if bench == "locomo"]
        cap_by_cutoff = "/".join(
            str(row.get("completion_cap_hits", 0))
            for row in sorted(locomo_rows, key=lambda item: item.get("cutoff", 0)))
        body += (f"该 Mem0 baseline 为 BF16 无量化运行；LongMemEval 有 {lme_failed} 个"
                 "最终失败的 ingestion pair；LoCoMo top-50/top-200 分别有 "
                 f"{cap_by_cutoff} 题命中 4096 completion Token 上限。"
                 "LoCoMo 构建成本按 10 个 conversation owner 去重统计。")
    return r"\paragraph{结果解读。}" + body + "\n"


def plot_ablation(payload: dict, report: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphmem-v519-matplotlib")
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import font_manager  # noqa: PLC0415
    font = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.2, "axes.axisbelow": True,
                         "axes.unicode_minus": False})
    colors = {"navy": "#12304A", "blue": "#2378D7", "teal": "#18A999",
              "amber": "#F2A93B", "red": "#D84A4A", "gray": "#5F6B76"}
    arms = payload.get("arms", {})
    labels = [label for _, label in ARMS]
    has_data = bool(arms)
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.65),
                                gridspec_kw={"width_ratios": [1.2, 1, 1]})

    signals = ("scene_similar", "shared_entity", "state_compatible",
               "collection_related", "temporal_near", "lexical_rare")
    signal_labels = ("scene", "entity", "state", "collection", "temporal", "lexical")
    signal_colors = (colors["blue"], colors["teal"], colors["amber"],
                     colors["red"], colors["navy"], colors["gray"])
    structural = []; temporal = []; multi_counts = []
    edge_by_signal = {signal: [] for signal in signals}
    traversal_by_signal = {signal: [] for signal in signals}
    for key, _ in ARMS:
        group = arms.get(key, {}).get("accuracy", {}).get("by_group", {})
        full_group = arms.get("full", {}).get("accuracy", {}).get("by_group", {})
        structural.append(100 * ((group.get("structural", {}).get("accuracy") or 0)
                                 - (full_group.get("structural", {}).get("accuracy") or 0)))
        temporal.append(100 * ((group.get("temporal", {}).get("accuracy") or 0)
                               - (full_group.get("temporal", {}).get("accuracy") or 0)))
        graph = arms.get(key, {}).get("graph") or {}
        multi_counts.append(int(graph.get("multi_attribute_edges", 0)))
        traversed = arms.get(key, {}).get("retrieval", {}).get(
            "traversed_relation_signals", {})
        for signal in signals:
            edge_by_signal[signal].append(int(
                graph.get("edge_signal_counts", {}).get(signal, 0)))
            traversal_by_signal[signal].append(int(traversed.get(signal, 0)))
    x = range(len(labels)); width = 0.36
    axes[0].bar([i - width / 2 for i in x], structural, width,
                label="结构题", color=colors["blue"])
    axes[0].bar([i + width / 2 for i in x], temporal, width,
                label="Temporal", color=colors["amber"])
    axes[0].axhline(0, color=colors["navy"], linewidth=0.8)
    axes[0].set_ylabel("相对 Full 的准确率变化（pp）")
    axes[0].set_title("(a) 端到端准确率增量", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="best")

    edge_bottom = [0] * len(labels)
    traversal_bottom = [0] * len(labels)
    for signal, signal_label, color in zip(signals, signal_labels, signal_colors):
        edge_values = edge_by_signal[signal]
        traversal_values = traversal_by_signal[signal]
        axes[1].bar(x, edge_values, bottom=edge_bottom, color=color,
                    label=signal_label)
        axes[2].bar(x, traversal_values, bottom=traversal_bottom, color=color,
                    label=signal_label)
        edge_bottom = [left + right for left, right in zip(edge_bottom, edge_values)]
        traversal_bottom = [left + right for left, right in zip(
            traversal_bottom, traversal_values)]
    axes[1].plot(x, multi_counts, color="#111827", marker="D", markersize=3,
                 linewidth=1.1, label="multi-attribute edges")
    axes[1].set_ylabel("物化 signal incidence / edges")
    axes[1].set_title("(b) 各 signal 物化与多属性边", fontweight="bold")

    axes[2].set_ylabel("查询实际 traversal 次数")
    axes[2].set_title("(c) 属性边实际使用量", fontweight="bold")
    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=28, ha="right")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    if not has_data:
        for axis in axes:
            axis.text(0.5, 0.52, "实验运行中：不渲染占位数值",
                      transform=axis.transAxes, ha="center", va="center",
                      color=colors["gray"], fontsize=10)
    figure.suptitle("多属性关系信号的质量—结构—使用闭环消融",
                    color=colors["navy"], fontsize=14, fontweight="bold")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    if has_data:
        figure.legend(handles, legend_labels, frameon=False, ncol=4,
                      loc="lower center", bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.08 if has_data else 0, 1, 0.92))
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(figures / f"v5_19_attribute_ablation.{suffix}",
                       dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    ablation = read(args.ablation)
    benchmark = read(args.benchmark)
    generated = args.report / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "v5_19_attribute_ablation_table.tex").write_text(
        render_ablation_table(ablation), encoding="utf-8")
    (generated / "v5_19_dual_model_table.tex").write_text(
        render_main_table(benchmark), encoding="utf-8")
    (generated / "v5_19_attribute_ablation_analysis.tex").write_text(
        render_ablation_analysis(ablation), encoding="utf-8")
    (generated / "v5_19_dual_model_analysis.tex").write_text(
        render_main_analysis(benchmark), encoding="utf-8")
    plot_ablation(ablation, args.report)
    sources = {
        "schema_version": "graphmem-v5.19-report-assets-v1",
        "ablation_manifest": str(args.ablation) if args.ablation else None,
        "benchmark_manifest": str(args.benchmark) if args.benchmark else None,
        "ablation_loaded": bool(ablation), "benchmark_loaded": bool(benchmark),
        "missing_values_render_as_zero": False,
    }
    (generated / "v5_19_report_sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sources, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
