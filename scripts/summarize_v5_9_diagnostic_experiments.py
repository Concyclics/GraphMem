#!/usr/bin/env python3
"""Summarize V5.9 diagnostic interventions with paired statistics."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN = WORKSPACE / "artifacts/v5_9/full_benchmark_20260809/answers/merged"
ABL = WORKSPACE / "artifacts/v5_9/diagnostic_ablations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=RUN)
    parser.add_argument(
        "--candidate-off", type=Path,
        default=ABL / "candidate_off/v5_6_answer_candidate_off_dev200_20260808T190434Z",
    )
    parser.add_argument(
        "--span128", type=Path,
        default=ABL / "span128/v5_6_answer_span128_dev200_20260808T191111Z",
    )
    parser.add_argument(
        "--turn64", type=Path,
        default=ABL / "turn64/v5_6_answer_turn64_dev200_20260808T191807Z",
    )
    parser.add_argument(
        "--dense", type=Path,
        default=ABL / "dense_dev200/results/c23_results.json",
    )
    parser.add_argument(
        "--sparse", type=Path,
        default=ABL / "dense_dev200/results_sparse/c23_results.json",
    )
    parser.add_argument(
        "--path-retention", type=Path,
        default=ROOT / "artifacts/report/v5_9/path_retention/path_retention.json",
    )
    parser.add_argument(
        "--extraction", type=Path,
        default=ROOT / "artifacts/report/v5_9/extraction_rescue/extraction_rescue.json",
    )
    parser.add_argument(
        "--error-chain", type=Path,
        default=ROOT / "artifacts/report/v5_9/error_chain/error_chain.json",
    )
    parser.add_argument(
        "--system", type=Path,
        default=ROOT / "artifacts/report/v5_9/system/system_results.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/report/v5_9/diagnostic_summary",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def judge_map(root: Path, benchmark: str) -> dict[str, bool]:
    directory = "judge_lme" if benchmark == "longmemeval" else "judge_locomo"
    return {str(row["question_id"]): bool(row["correct"])
            for row in read_jsonl(root / directory / "auto_eval.jsonl")}


def exact_mcnemar(old: dict[str, bool], new: dict[str, bool], ids: list[str]) -> dict[str, Any]:
    ids = [qid for qid in ids if qid in old and qid in new]
    new_only = sum(new[qid] and not old[qid] for qid in ids)
    old_only = sum(old[qid] and not new[qid] for qid in ids)
    discordant = new_only + old_only
    p = 1.0
    if discordant:
        tail = sum(math.comb(discordant, index)
                   for index in range(min(new_only, old_only) + 1)) / (2 ** discordant)
        p = min(1.0, 2 * tail)
    return {
        "questions": len(ids),
        "baseline_correct": sum(old[qid] for qid in ids),
        "arm_correct": sum(new[qid] for qid in ids),
        "baseline_accuracy": sum(old[qid] for qid in ids) / max(1, len(ids)),
        "arm_accuracy": sum(new[qid] for qid in ids) / max(1, len(ids)),
        "delta": (sum(new[qid] for qid in ids) - sum(old[qid] for qid in ids)) / max(1, len(ids)),
        "new_only": new_only, "baseline_only": old_only,
        "mcnemar_exact_p": p,
    }


def paired_bootstrap(values: list[float], resamples: int = 5000) -> list[float] | None:
    if len(values) < 5:
        return None
    rng = random.Random(42)
    samples = [fmean(values[rng.randrange(len(values))] for _ in values)
               for _ in range(resamples)]
    samples.sort()
    return [samples[int(.025 * len(samples))], samples[int(.975 * len(samples))]]


def answer_arm(name: str, root: Path, baseline: Path,
               base_answers: dict[str, dict[str, Any]],
               base_retrieval: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answers = read_jsonl(root / "answers.jsonl")
    retrieval = {str(row["dev_question_id"]): row
                 for row in read_jsonl(root / "retrieval.jsonl")}
    result: dict[str, Any] = {"name": name, "root": str(root), "benchmarks": {}}
    for benchmark in ("longmemeval", "locomo"):
        ids = [str(row["question_id"]) for row in answers if row["benchmark"] == benchmark]
        old, new = judge_map(baseline, benchmark), judge_map(root, benchmark)
        paired = exact_mcnemar(old, new, ids)
        prompt_delta = [float(retrieval[qid]["prompt_tokens"])
                        - float(base_retrieval[qid]["prompt_tokens"]) for qid in ids]
        evidence_delta = [float(retrieval[qid]["evidence_tokens"])
                          - float(base_retrieval[qid]["evidence_tokens"]) for qid in ids]
        all_hit_old = {qid: bool(base_retrieval[qid]["turn_all_hit"]) for qid in ids}
        all_hit_new = {qid: bool(retrieval[qid]["turn_all_hit"]) for qid in ids}
        closed = [qid for qid in ids if bool(base_retrieval[qid].get("closed_form"))]
        not_closed = [qid for qid in ids if qid not in set(closed)]
        arm_answers = {str(row["question_id"]): row for row in answers}
        result["benchmarks"][benchmark] = {
            **paired,
            "prediction_identical_rate": sum(
                str(arm_answers[qid]["prediction"]) == str(base_answers[qid]["prediction"])
                for qid in ids) / len(ids),
            "prompt_tokens_mean": fmean(float(retrieval[qid]["prompt_tokens"]) for qid in ids),
            "prompt_token_delta_mean": fmean(prompt_delta),
            "prompt_token_delta_nonzero": sum(value != 0 for value in prompt_delta),
            "evidence_token_delta_mean": fmean(evidence_delta),
            "all_hit": exact_mcnemar(all_hit_old, all_hit_new, ids),
            "baseline_closed_form_subset": exact_mcnemar(old, new, closed),
            "baseline_non_closed_subset": exact_mcnemar(old, new, not_closed),
            "by_stratum": {},
        }
        for stratum in sorted({str(retrieval[qid]["stratum"]) for qid in ids}):
            subset = [qid for qid in ids if str(retrieval[qid]["stratum"]) == stratum]
            result["benchmarks"][benchmark]["by_stratum"][stratum] = {
                **exact_mcnemar(old, new, subset),
                "all_hit": sum(bool(retrieval[qid]["turn_all_hit"]) for qid in subset) / len(subset),
                "turn_recall": fmean(float(retrieval[qid]["turn_recall"]) for qid in subset),
                "evidence_tokens": fmean(float(retrieval[qid]["evidence_tokens"]) for qid in subset),
            }
    return result


def dense_summary(sparse_path: Path, dense_path: Path) -> dict[str, Any]:
    sparse = read_json(sparse_path)["arms"]["adaptive@32"]
    dense = read_json(dense_path)["arms"]["adaptive@32"]
    old_rows = {str(row["question_id"]): row for row in sparse["per_question"]}
    new_rows = {str(row["question_id"]): row for row in dense["per_question"]}
    ids = sorted(set(old_rows) & set(new_rows))
    old_hit = {qid: bool(old_rows[qid]["all_hit"]) for qid in ids}
    new_hit = {qid: bool(new_rows[qid]["all_hit"]) for qid in ids}
    recall_deltas = [float(new_rows[qid]["turn_recall"])
                     - float(old_rows[qid]["turn_recall"]) for qid in ids]
    return {
        "questions": len(ids),
        "all_hit": exact_mcnemar(old_hit, new_hit, ids),
        "turn_recall_sparse": fmean(float(old_rows[qid]["turn_recall"]) for qid in ids),
        "turn_recall_dense": fmean(float(new_rows[qid]["turn_recall"]) for qid in ids),
        "turn_recall_delta": fmean(recall_deltas),
        "turn_recall_delta_bootstrap_ci95": paired_bootstrap(recall_deltas),
        "route_recall_sparse": sparse["route_gold_session_recall"],
        "route_recall_dense": dense["route_gold_session_recall"],
        "latency_p95_ms_sparse": sparse["latency_ms"]["p95"],
        "latency_p95_ms_dense": dense["latency_ms"]["p95"],
        "latency_p95_ratio": dense["latency_ms"]["p95"] / sparse["latency_ms"]["p95"],
        "false_complete_given_certified_sparse": sparse["false_complete_given_certified"],
        "false_complete_given_certified_dense": dense["false_complete_given_certified"],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# V5.9 诊断实验方法与结论",
        "",
        "## 实验设计",
        "",
        "所有答案实验固定同一 V5.9 SQLite 图、H10 QueryIR、Qwen-30B、temperature=0 和 pinned judge。"
        "改动项之外保持一致，逐题以 McNemar exact test 比较。span128 是空操作/运行间噪声对照；"
        "dense 只比较检索，不重新回答。",
        "",
        "## 答案与证据消融（困难集 100+100）",
        "",
        "| Arm | Benchmark | Baseline | Arm | Delta | New/Old only | p | Prompt Δ | All-hit Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["answer_arms"]:
        for benchmark, label in (("longmemeval", "LME"), ("locomo", "LoCoMo")):
            row = arm["benchmarks"][benchmark]
            hit = row["all_hit"]
            lines.append(
                f"| {arm['name']} | {label} | {row['baseline_accuracy']:.1%} | "
                f"{row['arm_accuracy']:.1%} | {row['delta']:+.1%} | "
                f"{row['new_only']}/{row['baseline_only']} | {row['mcnemar_exact_p']:.4f} | "
                f"{row['prompt_token_delta_mean']:+.0f} | {hit['delta']:+.1%} |"
            )
    dense = payload["dense"]
    extraction = payload["extraction"]["summary"]["overall"]
    path20 = {row["method"]: row for row in payload["path_retention"]["rows"]
              if row["n"] == 20000}
    lines.extend([
        "",
        "## Dense 检索",
        "",
        f"同一 200 题中 all-hit {dense['all_hit']['baseline_accuracy']:.1%}→"
        f"{dense['all_hit']['arm_accuracy']:.1%}（{dense['all_hit']['delta']:+.1%}, "
        f"p={dense['all_hit']['mcnemar_exact_p']:.4f}），p95 "
        f"{dense['latency_p95_ms_sparse']:.0f}→{dense['latency_p95_ms_dense']:.0f} ms。"
        f"Gold-session route recall 不变，说明 dense 只补 seed，不改善层级路由。",
        "",
        "## 原文到 Fact 补抽",
        "",
        f"在 {extraction['questions']} 个缺失 Fact 的问题中，current/augmented/raw sufficient "
        f"分别为 {extraction['current_sufficiency']:.1%}/"
        f"{extraction['augmented_sufficiency']:.1%}/"
        f"{extraction['raw_oracle_sufficiency']:.1%}；对 raw-supported current failure 的救回率为 "
        f"{extraction['rescue_rate_given_raw_supported_current_failure']:.1%}。",
        "",
        "## 修正后的 C1 路径指标（N=20K）",
        "",
        "| Method | Edge recall | ≤2-hop reachability | Internal component connected |",
        "|---|---:|---:|---:|",
    ])
    for method in ("ann_only", "flat_sparse", "cir"):
        row = path20[method]
        lines.append(
            f"| {method} | {row['gold_edge_recall']:.2%} | "
            f"{row['gold_pair_reachable_within_2_hops']:.2%} | "
            f"{row['component_connected_internal_edges_rate']:.2%} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- span128 没有改变任何一题的 prompt token，却改变大量答案，是运行间非确定性估计，不能解释成方法效果。",
        "- extraction rescue 的 extractor 与 judge 使用同一 backbone，需异构 judge/人工复核。",
        "- C1 是合成结构 workload，不是 typed-relation precision 或端到端 QA。",
        "- 所有 accuracy intervention 在困难开发集完成，进入主报告前须在冻结 holdout 复验。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    base_answers = {str(row["question_id"]): row
                    for row in read_jsonl(args.baseline / "answers.jsonl")}
    base_retrieval = {str(row["dev_question_id"]): row
                      for row in read_jsonl(args.baseline / "retrieval.jsonl")}
    arms = [
        answer_arm("candidate_off", args.candidate_off, args.baseline,
                   base_answers, base_retrieval),
        answer_arm("span128", args.span128, args.baseline,
                   base_answers, base_retrieval),
        answer_arm("turn64", args.turn64, args.baseline,
                   base_answers, base_retrieval),
    ]
    payload = {
        "schema_version": "graphmem-v5.9-diagnostic-summary-v1",
        "answer_arms": arms,
        "dense": dense_summary(args.sparse, args.dense),
        "path_retention": read_json(args.path_retention),
        "extraction": read_json(args.extraction),
        "error_chain": read_json(args.error_chain),
        "system": read_json(args.system),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render(payload), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
