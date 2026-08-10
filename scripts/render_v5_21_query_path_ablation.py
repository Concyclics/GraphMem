#!/usr/bin/env python3
"""Render the full-corpus V5.21 query-path ablation and its report assets."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def by_id(data: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["question_id"]): row for row in data}
    if len(result) != len(data):
        raise ValueError("duplicate question_id")
    return result


def verdicts(*paths: Path) -> dict[str, bool]:
    data: dict[str, bool] = {}
    for path in paths:
        for row in rows(path):
            question_id = str(row["question_id"])
            if question_id in data:
                raise ValueError(f"duplicate judge verdict: {question_id}")
            data[question_id] = bool(row["correct"])
    return data


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = min(gains, losses)
    return min(1.0, 2.0 * sum(
        math.comb(discordant, value) for value in range(tail + 1))
        / (2 ** discordant))


def paired(left: Mapping[str, bool], right: Mapping[str, bool],
           question_ids: Sequence[str]) -> dict[str, Any]:
    gains = sum(not left[item] and right[item] for item in question_ids)
    losses = sum(left[item] and not right[item] for item in question_ids)
    return {
        "questions": len(question_ids), "gains": gains, "losses": losses,
        "delta": (gains - losses) / max(1, len(question_ids)),
        "mcnemar_exact_p": exact_mcnemar(gains, losses),
    }


def mean(data: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in data if row.get(key) is not None]
    return sum(values) / max(1, len(values))


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def delta(value: float) -> str:
    return f"{100 * value:+.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True,
                        help="PDF output; matching PNG and SVG are also written")
    args = parser.parse_args()

    root = args.experiment_root
    definitions = (
        ("baseline", "V5.20 fixed", args.baseline_root,
         args.baseline_root / "judge_lme" / "auto_eval.jsonl",
         args.baseline_root / "judge_locomo" / "auto_eval.jsonl"),
        ("safe_witness", "+ Safe Witness", root / "answer_baseline",
         root / "answer_baseline" / "paired_vs_v520_judge_lme" / "auto_eval.jsonl",
         root / "answer_baseline" / "paired_vs_v520_judge_locomo" / "auto_eval.jsonl"),
        ("aggregation", "+ Operand Ledger", root / "answer_m4_aggregation",
         root / "answer_m4_aggregation" / "paired_vs_m2_judge_lme" / "auto_eval.jsonl",
         root / "answer_m4_aggregation" / "paired_vs_m2_judge_locomo" / "auto_eval.jsonl"),
        ("preference", "+ Preference Synthesis", root / "answer_m5_preference",
         root / "answer_m5_preference" / "paired_vs_m4_judge_lme" / "auto_eval.jsonl",
         root / "answer_m5_preference" / "paired_vs_m4_judge_locomo" / "auto_eval.jsonl"),
    )
    stages: dict[str, dict[str, Any]] = {}
    for key, label, answer_root, lme_judge, locomo_judge in definitions:
        answers = by_id(rows(answer_root / "answers.jsonl"))
        judged = verdicts(lme_judge, locomo_judge)
        if set(answers) != set(judged):
            raise ValueError(f"{key}: answer/judge IDs differ")
        accuracy: dict[str, Any] = {}
        by_type: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"questions": 0, "correct": 0})
        for benchmark in ("longmemeval", "locomo"):
            ids = [item for item, row in answers.items()
                   if row.get("benchmark") == benchmark]
            correct = sum(judged[item] for item in ids)
            accuracy[benchmark] = {
                "questions": len(ids), "correct": correct,
                "accuracy": correct / max(1, len(ids)),
            }
        for question_id, row in answers.items():
            label_key = str(row.get("question_type") or row.get("stratum"))
            by_type[label_key]["questions"] += 1
            by_type[label_key]["correct"] += int(judged[question_id])
        for value in by_type.values():
            value["accuracy"] = value["correct"] / max(1, value["questions"])
        stages[key] = {
            "label": label, "answer_root": str(answer_root),
            "accuracy": accuracy, "by_type": dict(by_type),
            "answers": answers, "verdicts": judged,
        }

    comparisons: dict[str, Any] = {}
    stage_keys = [row[0] for row in definitions]
    for left_key, right_key in zip(stage_keys, stage_keys[1:]):
        left = stages[left_key]; right = stages[right_key]
        entry = {}
        for benchmark in ("longmemeval", "locomo"):
            ids = [item for item, row in right["answers"].items()
                   if row.get("benchmark") == benchmark]
            entry[benchmark] = paired(
                left["verdicts"], right["verdicts"], ids)
        comparisons[f"{left_key}->{right_key}"] = entry

    final_comparison = {}
    for benchmark in ("longmemeval", "locomo"):
        ids = [item for item, row in stages["preference"]["answers"].items()
               if row.get("benchmark") == benchmark]
        final_comparison[benchmark] = paired(
            stages["baseline"]["verdicts"],
            stages["preference"]["verdicts"], ids)

    old_retrieval = rows(args.baseline_root / "retrieval.jsonl")
    new_retrieval = rows(root / "answer_baseline" / "retrieval.jsonl")
    retrieval_effect = {}
    for benchmark in ("longmemeval", "locomo"):
        old_rows = [row for row in old_retrieval
                    if row.get("benchmark") == benchmark]
        new_rows = [row for row in new_retrieval
                    if row.get("benchmark") == benchmark]
        retrieval_effect[benchmark] = {
            key: {"before": mean(old_rows, key), "after": mean(new_rows, key)}
            for key in ("candidate_turn_recall", "candidate_turn_precision",
                        "turn_recall", "turn_precision", "turn_all_hit",
                        "visited_nodes", "visited_edges", "latency_total_ms")
        }

    prepared_m4 = rows(root / "answer_m4_aggregation" / "prepared_answers.jsonl")
    prepared_m5 = rows(root / "answer_m5_preference" / "prepared_answers.jsonl")
    route_audit = {
        "aggregation_routed": sum(
            row.get("trace", {}).get("aggregation_ledger") is not None
            for row in prepared_m4),
        "aggregation_certified": sum(
            bool(row.get("deterministic_prediction")) for row in prepared_m4),
        "preference_routed": sum(
            bool(row.get("trace", {}).get("preference_synthesis"))
            for row in prepared_m5),
        "preference_routed_locomo": sum(
            bool(row.get("trace", {}).get("preference_synthesis"))
            and stages["preference"]["answers"][str(row["question_id"])].get(
                "benchmark") == "locomo" for row in prepared_m5),
    }

    serializable_stages = {}
    for key, value in stages.items():
        serializable_stages[key] = {
            item: payload for item, payload in value.items()
            if item not in {"answers", "verdicts"}
        }
    payload = {
        "schema_version": "graphmem-v5.21-full-query-path-ablation-v1",
        "protocol": {"longmemeval_questions": 500,
                     "locomo_questions": 1540,
                     "answer_model": "Qwen3-30B",
                     "judge_model": "gpt-5.6-luna",
                     "turn_budget": 64, "evidence_token_budget": 12000},
        "stages": serializable_stages, "comparisons": comparisons,
        "final_vs_baseline": final_comparison,
        "retrieval_effect": retrieval_effect, "route_audit": route_audit,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n",
                                encoding="utf-8")

    table_rows = []
    for index, key in enumerate(stage_keys):
        stage = stages[key]
        if index:
            comparison = comparisons[f"{stage_keys[index - 1]}->{key}"]
        else:
            comparison = None
        cells = [stage["label"]]
        for benchmark in ("longmemeval", "locomo"):
            result = stage["accuracy"][benchmark]
            cells += [f"{result['correct']}/{result['questions']}",
                      pct(result["accuracy"])]
            if comparison is None:
                cells += ["--", "--", "--"]
            else:
                paired_result = comparison[benchmark]
                cells += [delta(paired_result["delta"]),
                          f"{paired_result['gains']}/{paired_result['losses']}",
                          f"{paired_result['mcnemar_exact_p']:.3f}"]
        table_rows.append(" & ".join(cells) + r" \\")
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.write_text("\n".join(table_rows) + "\n\\bottomrule\n",
                                 encoding="utf-8")

    witness_lme = comparisons["baseline->safe_witness"]["longmemeval"]
    witness_locomo = comparisons["baseline->safe_witness"]["locomo"]
    ledger_lme = comparisons["safe_witness->aggregation"]["longmemeval"]
    preference_lme = comparisons["aggregation->preference"]["longmemeval"]
    final_lme = stages["preference"]["accuracy"]["longmemeval"]
    final_locomo = stages["preference"]["accuracy"]["locomo"]
    multi_before = stages["safe_witness"]["by_type"]["multi-session"]
    multi_after = stages["aggregation"]["by_type"]["multi-session"]
    pref_before = stages["aggregation"]["by_type"]["single-session-preference"]
    pref_after = stages["preference"]["by_type"]["single-session-preference"]
    loc_retrieval = retrieval_effect["locomo"]
    analysis = (
        r"\paragraph{全量查询路径消融。}所有候选均在 LongMemEval 500 题与 "
        r"LoCoMo Category 1--4 的 1,540 题上运行，不再以 hard 子集决定版本。"
        f"Safe Witness 使 LME 变化 {delta(witness_lme['delta'])} pp、LoCoMo "
        f"变化 {delta(witness_locomo['delta'])} pp，但 LoCoMo 的 candidate recall "
        f"{100 * loc_retrieval['candidate_turn_recall']['before']:.2f}\\%$\\rightarrow$"
        f"{100 * loc_retrieval['candidate_turn_recall']['after']:.2f}\\%、packed recall "
        f"{100 * loc_retrieval['turn_recall']['before']:.2f}\\%$\\rightarrow$"
        f"{100 * loc_retrieval['turn_recall']['after']:.2f}\\% 几乎不变；"
        f"visited edges 从 {loc_retrieval['visited_edges']['before']:.1f} 降到 "
        f"{loc_retrieval['visited_edges']['after']:.1f}。这说明该改动主要收紧访问范围，"
        r"尚未形成新的 gold 证据闭包，不能把局部图结构变化表述为显著准确率提升。"
        f"Operand Ledger 随后把 LME 提高 {delta(ledger_lme['delta'])} pp"
        f"（{ledger_lme['gains']}/{ledger_lme['losses']} 修复/退化，"
        f"$p={ledger_lme['mcnemar_exact_p']:.3f}$），其中 Multi-session 从 "
        f"{pct(multi_before['accuracy'])}\\% 提高到 {pct(multi_after['accuracy'])}\\%；"
        f"{route_audit['aggregation_routed']} 题进入 ledger，只有 "
        f"{route_audit['aggregation_certified']} 题满足严格闭包并由确定性代码直答。"
        f"Preference Synthesis 只路由 {route_audit['preference_routed']} 题且 LoCoMo "
        f"误路由为 {route_audit['preference_routed_locomo']}，使 Preference 从 "
        f"{pref_before['correct']}/{pref_before['questions']}（{pct(pref_before['accuracy'])}\\%）"
        f"提高到 {pref_after['correct']}/{pref_after['questions']}"
        f"（{pct(pref_after['accuracy'])}\\%），{preference_lme['gains']} 修复、"
        f"{preference_lme['losses']} 退化。最终 LME/LoCoMo 分别为 "
        f"{final_lme['correct']}/{final_lme['questions']}（{pct(final_lme['accuracy'])}\\%）"
        f"与 {final_locomo['correct']}/{final_locomo['questions']}"
        f"（{pct(final_locomo['accuracy'])}\\%）。逐项 $p$ 值如表所示；"
        "未达到 $p<0.05$ 的小幅变化只作为方向性结果，不作显著提升声明。\n"
    )
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(analysis, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [stages[key]["label"] for key in stage_keys]
    x = range(len(labels))
    colors = {"longmemeval": "#2378D7", "locomo": "#18A999"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), constrained_layout=True)
    for benchmark, label in (("longmemeval", "LongMemEval"),
                             ("locomo", "LoCoMo Cat. 1-4")):
        values = [100 * stages[key]["accuracy"][benchmark]["accuracy"]
                  for key in stage_keys]
        axes[0].plot(x, values, marker="o", linewidth=2.2, markersize=6,
                     color=colors[benchmark], label=label)
        for index, value in enumerate(values):
            axes[0].annotate(f"{value:.1f}", (index, value),
                             xytext=(0, 7), textcoords="offset points",
                             ha="center", fontsize=8, color=colors[benchmark])
    axes[0].set_xticks(list(x), ["Base", "+Witness", "+Ledger", "+Preference"])
    axes[0].set_ylabel("Judge accuracy (%)")
    axes[0].set_title("Full-benchmark accuracy")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, loc="lower right")

    transition_keys = list(comparisons)
    positions = list(range(len(transition_keys)))
    width = .34
    for offset, benchmark, label in ((-.17, "longmemeval", "LongMemEval"),
                                     (.17, "locomo", "LoCoMo Cat. 1-4")):
        values = [100 * comparisons[key][benchmark]["delta"]
                  for key in transition_keys]
        bars = axes[1].bar([item + offset for item in positions], values,
                           width=width, color=colors[benchmark], label=label)
        axes[1].bar_label(bars, labels=[f"{value:+.1f}" for value in values],
                          padding=2, fontsize=8)
    axes[1].axhline(0, color="#5F6B76", linewidth=.8)
    axes[1].set_xticks(positions, ["Witness", "Ledger", "Preference"])
    axes[1].set_ylabel("Paired delta (pp)")
    axes[1].set_title("Incremental effect vs. previous stage")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, bbox_inches="tight")
    fig.savefig(args.output_figure.with_suffix(".png"), dpi=220,
                bbox_inches="tight")
    fig.savefig(args.output_figure.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
