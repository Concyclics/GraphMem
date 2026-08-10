#!/usr/bin/env python3
"""Create the report contract for GraphMem/Mem0 dual-backbone results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def stats(root: Path, benchmark: str) -> dict | None:
    judge = read(root / ("judge_lme" if benchmark == "longmemeval"
                         else "judge_locomo") / "judge_token_stats.json")
    manifest = read(root / "run_manifest.json")
    if judge is None or manifest is None:
        return None
    return {
        "questions": judge.get("question_count"),
        "accuracy": judge.get("accuracy"),
        "answer_tokens": manifest.get(
            "answer_api_tokens_by_benchmark", {}).get(benchmark, {}).get("total")
            or manifest.get(
                "api_tokens_by_benchmark", {}).get(benchmark, {}).get("total"),
        "judge_model": judge.get("model"),
        "config_hash": manifest.get("config_hash"),
        "answer_prompt_hash": manifest.get("answer_prompt_hash"),
        "artifacts": str(root),
    }


def placeholder(model: str, benchmark: str) -> dict:
    return {
        "answer_model": model, "benchmark": benchmark, "status": "pending",
        "retrieval_setting": "H11/64",
        "questions": None, "accuracy": None, "build_tokens": None,
        "answer_tokens": None, "artifacts": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--gpt-root", type=Path, required=True)
    parser.add_argument("--mem0", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    build = read(args.build_report) or {}
    build_stats = build.get("summary", {}).get("build_token_stats", {}).get("total")
    build_config_hash = build.get("summary", {}).get("config_hash")
    methods = []
    for method, model, root in (
        ("GraphMem", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", args.qwen_root),
        ("GraphMem", "gpt-5.4-mini", args.gpt_root),
    ):
        for benchmark in ("longmemeval", "locomo"):
            measured = stats(root, benchmark)
            row = placeholder(model, benchmark)
            row.update({"method": method, "build_tokens": build_stats,
                        "build_config_hash": build_config_hash})
            if measured:
                row.update(measured); row["status"] = "complete"
            methods.append(row)

    supplied = read(args.mem0) if args.mem0 else None
    if supplied:
        methods.extend(list(supplied.get("rows", ())))
    else:
        for model in ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "gpt-5.4-mini"):
            for benchmark in ("longmemeval", "locomo"):
                row = placeholder(model, benchmark)
                row["method"] = "Mem0"
                methods.append(row)
    payload = {
        "schema_version": "graphmem-v5.19-dual-backbone-benchmark-v1",
        "percentile_method": "nearest_rank",
        "judge_model": "gpt-5.6-luna",
        "judge_tokens_excluded": True,
        "build_tokens_exclude_embedding": True,
        "rows": methods,
        "mem0_baseline_contract": ({key: supplied.get(key) for key in (
            "schema_version", "archive", "model", "percentile_method",
            "judge_model", "locomo_categories", "cutoffs", "audit", "warnings")}
            if supplied else None),
        "sources": {"build_report": str(args.build_report),
                    "qwen_root": str(args.qwen_root),
                    "gpt_root": str(args.gpt_root),
                    "mem0": str(args.mem0) if args.mem0 else None},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"rows": len(methods), "complete": sum(
        row.get("status") == "complete" for row in methods)}, indent=2))


if __name__ == "__main__":
    main()
