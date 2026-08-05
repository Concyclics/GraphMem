#!/usr/bin/env python3
"""Run persisted V4.1 replay in CPU-parallel stable shards and merge JSONL."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers-per-shard", type=int, default=4)
    parser.add_argument("--context-token-budget", type=int, default=9200)
    parser.add_argument("--query-target-tokens", type=int, default=10000)
    parser.add_argument("--query-hard-limit-tokens", type=int, default=13000)
    parser.add_argument("--planner-output-tokens", type=int, default=192)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--llm-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--llm-request-profile", default="deepseek")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("data must be a non-empty JSON list")
    shard_count = max(1, min(args.shards, len(rows)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    buckets = [[] for _ in range(shard_count)]
    for position, row in enumerate(rows):
        buckets[position % shard_count].append(row)

    processes: list[tuple[int, subprocess.Popen, object]] = []
    for shard_id, bucket in enumerate(buckets):
        shard_dir = shard_root / f"{shard_id:03d}"
        shard_dir.mkdir(exist_ok=True)
        shard_data = shard_dir / "cases.json"
        shard_data.write_text(
            json.dumps(bucket, ensure_ascii=False), encoding="utf-8",
        )
        log_handle = (shard_dir / "replay.log").open(
            "a" if args.resume else "w", encoding="utf-8",
        )
        command = [
            sys.executable, str(ROOT / "scripts" / "replay_v41_persisted.py"),
            "--data", str(shard_data),
            "--index-run-root", str(args.index_run_root),
            "--output-dir", str(shard_dir),
            "--workers", str(args.workers_per_shard),
            "--context-token-budget", str(args.context_token_budget),
            "--query-target-tokens", str(args.query_target_tokens),
            "--query-hard-limit-tokens", str(args.query_hard_limit_tokens),
            "--planner-output-tokens", str(args.planner_output_tokens),
            "--max-answer-tokens", str(args.max_answer_tokens),
            "--llm-model", args.llm_model,
            "--llm-base-url", args.llm_base_url,
            "--llm-api-key-env", args.llm_api_key_env,
            "--llm-request-profile", args.llm_request_profile,
        ]
        if args.resume:
            command.append("--resume")
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_id, process, log_handle))
        print(
            f"started shard {shard_id:03d}: {len(bucket)} cases "
            f"pid={process.pid}", flush=True,
        )

    failures = []
    for shard_id, process, log_handle in processes:
        code = process.wait()
        log_handle.close()
        print(f"finished shard {shard_id:03d}: exit={code}", flush=True)
        if code:
            failures.append((shard_id, code))
    if failures:
        raise RuntimeError(f"shard failures: {failures}")

    order = {str(row["question_id"]): i for i, row in enumerate(rows)}
    for filename in ("answers.jsonl", "retrieval_results.jsonl", "llm_calls.jsonl"):
        merged: list[dict] = []
        for shard_id in range(shard_count):
            merged.extend(read_jsonl(shard_root / f"{shard_id:03d}" / filename))
        if filename == "llm_calls.jsonl":
            merged.sort(key=lambda row: (
                order.get(str(row.get("question_id")), len(order)),
                str(row.get("stage") or ""),
            ))
        else:
            unique = {str(row["question_id"]): row for row in merged}
            missing = [qid for qid in order if qid not in unique]
            if missing:
                raise RuntimeError(
                    f"{filename} missing {len(missing)} questions: {missing[:8]}"
                )
            merged = sorted(unique.values(), key=lambda row: order[str(row["question_id"])])
        target = args.output_dir / filename
        with target.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged {len(rows)} cases into {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
