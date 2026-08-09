#!/usr/bin/env python3
"""Summarize the frozen-fact V5.10 LongMemEval + LoCoMo run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from summarize_v5_9_full_benchmark import (
    closed_form_audit, paired_stats, read_json, read_jsonl, retrieval_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", type=Path,
        default=Path("../artifacts/v5_10/full_benchmark_20260809/answers/merged"))
    parser.add_argument(
        "--graph-manifest", type=Path,
        default=Path("../artifacts/v5_10/full_benchmark_20260809/graph/recoarsen_manifest.json"))
    parser.add_argument(
        "--v5-9-root", type=Path,
        default=Path("../artifacts/v5_9/full_benchmark_20260809/answers/merged"))
    parser.add_argument(
        "--v5-8-root", type=Path,
        default=Path("../artifacts/v5_8/answers_rank/merged"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("../artifacts/report/v5_10/full_benchmark"))
    args = parser.parse_args()
    retrieval = read_jsonl(args.run_root / "retrieval.jsonl")
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for row in retrieval:
        by_benchmark[str(row["benchmark"])].append(row)
    payload = {
        "schema_version": "graphmem-v5.10-full-benchmark-v1",
        "scope": (
            "Frozen P8 fact projection; V5.10 recoarsening, H11 unified IR, "
            "native seed fusion and obligation-aware span packing. No extraction rebuild."
        ),
        "run_root": str(args.run_root),
        "run_manifest": read_json(args.run_root / "run_manifest.json"),
        "graph": read_json(args.graph_manifest),
        "benchmarks": {},
    }
    for key, judge_dir in (("longmemeval", "judge_lme"),
                           ("locomo", "judge_locomo")):
        current = read_jsonl(args.run_root / judge_dir / "auto_eval.jsonl")
        previous = read_jsonl(args.v5_9_root / judge_dir / "auto_eval.jsonl")
        v58 = read_jsonl(args.v5_8_root / judge_dir / "auto_eval.jsonl")
        payload["benchmarks"][key] = {
            "accuracy": read_json(args.run_root / judge_dir / "judge_token_stats.json"),
            "retrieval": retrieval_summary(by_benchmark[key]),
            "paired_vs_v5_9": paired_stats(previous, current),
            "paired_vs_v5_8": paired_stats(v58, current),
            "closed_form_audit_vs_v5_9": closed_form_audit(
                by_benchmark[key], previous, current),
        }
    payload["benchmarks"]["locomo"]["official_token_f1"] = read_json(
        args.run_root / "locomo_official_f1" / "official_eval.json")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V5.10 full benchmark audit", "",
        "固定 P8 fact projection；本轮只改变层级关系、H11、native seed 与 span pack。", "",
        "| Benchmark | Accuracy | all-hit | Prompt mean/p95 | Retrieval p95 | vs V5.9 | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("longmemeval", "LongMemEval"), ("locomo", "LoCoMo Cat1--4")):
        row = payload["benchmarks"][key]
        accuracy = row["accuracy"]["accuracy"]
        r = row["retrieval"]
        paired = row["paired_vs_v5_9"]
        lines.append(
            f"| {label} | {accuracy:.2%} | {r['turn_all_hit']:.2%} | "
            f"{r['prompt_tokens']['mean']:.0f}/{r['prompt_tokens']['p95']:.0f} | "
            f"{r['retrieval_latency_ms']['p95']:.1f} ms | "
            f"{paired['delta'] * 100:+.2f} pp | {paired['mcnemar_exact_p']:.4f} |")
    lines.extend([
        "", "注意：accuracy 对比受本地 FP8 answer/judge 运行波动影响；确定性 retrieval 指标应单独解释。",
        "本轮没有重建 extraction，因此不能用它验证 V5.10 atomic extractor 的全量 QA 收益。", "",
    ])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        key: {
            "accuracy": payload["benchmarks"][key]["accuracy"]["accuracy"],
            "retrieval": payload["benchmarks"][key]["retrieval"],
            "paired_vs_v5_9": payload["benchmarks"][key]["paired_vs_v5_9"],
        } for key in ("longmemeval", "locomo")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
