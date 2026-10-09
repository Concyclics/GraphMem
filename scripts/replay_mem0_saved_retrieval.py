#!/usr/bin/env python3
"""Replay frozen Mem0 retrieval results with an OpenAI-compatible answerer.

The runner deliberately reuses the answer prompts and post-processing from the
pinned ``mem0/evaluation`` checkout.  It never rebuilds memories or reruns
retrieval, so an answer-model comparison changes only the answer backbone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem.judging import OpenAICompatibleClient  # noqa: E402


LME_PROMPT_SHA256 = "ba8cf60d26f1390ecbef0f07b3e950556fe3bc5a37ba4b5343f28217f18c144f"
LOCOMO_PROMPT_SHA256 = "8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2"
MEMORY_BENCHMARKS_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
EXPECTED_COUNTS = {"longmemeval": 500, "locomo": 1540}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longmemeval-dir", type=Path, required=True)
    parser.add_argument("--locomo-dir", type=Path, required=True)
    parser.add_argument("--memory-benchmarks-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("SGAO_MODEL_GPT56_LUNA", "gpt-5.6-luna"))
    parser.add_argument("--base-url", default=os.environ.get("SGAO_BASE_URL"))
    parser.add_argument("--api-key-env", default="SGAO_API_KEY")
    parser.add_argument("--request-profile", choices=("openai", "qwen", "omit"), default="openai")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="max",
    )
    parser.add_argument("--cutoff", type=int, default=200)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--benchmark", choices=("both", "longmemeval", "locomo"), default="both")
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        default=0,
        help="deterministic smoke-test limit; zero means the complete selected benchmark(s)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prompt_payload_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_prompt_module(path: Path, expected_sha256: str, module_name: str) -> Any:
    digest = sha256_bytes(path.read_bytes())
    if digest != expected_sha256:
        raise RuntimeError(
            f"prompt source changed for {path}: {digest} != {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_longmemeval_date_human(date_str: str) -> str:
    """Exact copy of the pinned benchmark's date normalization."""
    try:
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
        value = datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
        return value.strftime("%A, %B %d, %Y")
    except (ValueError, TypeError):
        return date_str


def load_raw_json(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if "question_id" not in value or "retrieval" not in value:
            continue
        rows.append((path, value))
    return rows


def make_cases(args: argparse.Namespace, lme_prompts: Any, locomo_prompts: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if args.benchmark in {"both", "longmemeval"}:
        raw = load_raw_json(args.longmemeval_dir)
        if len(raw) != EXPECTED_COUNTS["longmemeval"]:
            raise RuntimeError(f"expected 500 LongMemEval rows, found {len(raw)}")
        for path, row in raw:
            retrieval = list(row["retrieval"]["search_results"])
            if len(retrieval) < args.cutoff:
                raise RuntimeError(f"{row['question_id']} has only {len(retrieval)} memories")
            # This matches process_question_answerer: slice by retrieval rank,
            # then present that fixed set chronologically to the answerer.
            evidence = sorted(
                retrieval[: args.cutoff], key=lambda item: item.get("created_at") or ""
            )
            question_date = str(row.get("question_date") or "")
            prompt = lme_prompts.get_answer_generation_prompt(
                question=str(row["question"]),
                search_results=evidence,
                question_date=(
                    parse_longmemeval_date_human(question_date)
                    if question_date
                    else ""
                ),
                user_profile=row.get("user_profile"),
            )
            cases.append(
                {
                    "question_id": str(row["question_id"]),
                    "benchmark": "longmemeval",
                    "stratum": str(row.get("question_type") or "unknown"),
                    "question_type": row.get("question_type"),
                    "question": row.get("question"),
                    "gold_answer": row.get("ground_truth_answer"),
                    "question_date": row.get("question_date"),
                    "source_file": str(path),
                    "memory_ids": [item.get("id") for item in evidence],
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
    if args.benchmark in {"both", "locomo"}:
        raw = [
            (path, row)
            for path, row in load_raw_json(args.locomo_dir)
            if int(row.get("category") or 0) in locomo_prompts.CATEGORIES_TO_EVALUATE
        ]
        if len(raw) != EXPECTED_COUNTS["locomo"]:
            raise RuntimeError(f"expected 1540 LoCoMo category 1-4 rows, found {len(raw)}")
        for path, row in raw:
            retrieval = list(row["retrieval"]["search_results"])
            if len(retrieval) < args.cutoff:
                raise RuntimeError(f"{row['question_id']} has only {len(retrieval)} memories")
            evidence = retrieval[: args.cutoff]
            prompt = locomo_prompts.get_answer_generation_prompt(
                str(row["question"]),
                evidence,
                reference_date=row.get("reference_date"),
                user_profile=row.get("user_profile"),
            )
            # get_answer_generation_prompt performs its own chronological sort.
            presented = sorted(evidence, key=lambda item: item.get("created_at", ""))
            cases.append(
                {
                    "question_id": str(row["question_id"]),
                    "benchmark": "locomo",
                    "stratum": f"category_{int(row['category'])}",
                    "category": int(row["category"]),
                    "question": row.get("question"),
                    "gold_answer": row.get("ground_truth_answer"),
                    "source_file": str(path),
                    "memory_ids": [item.get("id") for item in presented],
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
    cases.sort(key=lambda row: (str(row["benchmark"]), str(row["question_id"])))
    if args.limit_per_benchmark:
        selected: list[dict[str, Any]] = []
        for benchmark in ("longmemeval", "locomo"):
            selected.extend(
                row
                for row in cases
                if row["benchmark"] == benchmark
            )
            selected = (
                [row for row in selected if row["benchmark"] != benchmark]
                + [row for row in selected if row["benchmark"] == benchmark][
                    : args.limit_per_benchmark
                ]
            )
        cases = sorted(
            selected, key=lambda row: (str(row["benchmark"]), str(row["question_id"]))
        )
    for row in cases:
        row["prompt_payload_hash"] = prompt_payload_hash(row["messages"])
    ids = [str(row["question_id"]) for row in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate question IDs in replay workload")
    return cases


def clean_prediction(benchmark: str, response: str) -> str:
    value = response.strip()
    if benchmark == "longmemeval":
        value = re.sub(
            r"[<\[]mem_thinking[>\]].*?[<\[]/mem_thinking[>\]]",
            "",
            value,
            flags=re.DOTALL,
        ).strip()
    if "ANSWER:" in value:
        value = value.rsplit("ANSWER:", 1)[-1].strip()
    return value


def nearest_rank_stats(values: list[int | float], unit: str) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(p: float) -> int | float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else 0,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(ordered, default=0),
        "unit": unit,
        "percentile_method": "nearest_rank",
    }


def main() -> None:
    args = parse_args()
    prompt_root = args.memory_benchmarks_repo / "benchmarks"
    lme_path = prompt_root / "longmemeval" / "prompts.py"
    locomo_path = prompt_root / "locomo" / "prompts.py"
    lme_prompts = load_prompt_module(
        lme_path, LME_PROMPT_SHA256, "mem0_replay_longmemeval_prompts"
    )
    locomo_prompts = load_prompt_module(
        locomo_path, LOCOMO_PROMPT_SHA256, "mem0_replay_locomo_prompts"
    )
    cases = make_cases(args, lme_prompts, locomo_prompts)

    if args.output_root.exists() and not args.resume:
        if any(args.output_root.iterdir()):
            raise RuntimeError(
                f"output directory is not empty; pass --resume or choose another path: {args.output_root}"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output_root / "prepared_requests.jsonl"
    prepared_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
        encoding="utf-8",
    )

    answers_path = args.output_root / "answers.jsonl"
    usage_path = args.output_root / "answer_usage.jsonl"
    completed_answers = {
        str(row["question_id"]): row for row in read_jsonl(answers_path)
    }
    completed_usage = {
        str(row["question_id"]): row for row in read_jsonl(usage_path)
    }
    completed = set(completed_answers) & set(completed_usage)
    pending = [row for row in cases if str(row["question_id"]) not in completed]
    print(
        json.dumps(
            {
                "prepared": len(cases),
                "resumed": len(completed),
                "pending": len(pending),
                "by_benchmark": {
                    name: sum(row["benchmark"] == name for row in cases)
                    for name in ("longmemeval", "locomo")
                },
            }
        ),
        flush=True,
    )

    client = OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        request_profile=args.request_profile,
        max_retries=args.max_retries,
        timeout_sec=args.timeout_sec,
    )

    def answer(case: dict[str, Any]):
        result = client.chat(
            question_id=str(case["question_id"]),
            variant=f"mem0_top_{args.cutoff}_answer",
            stage="answer",
            messages=case["messages"],
            thinking_mode=args.reasoning_effort,
            max_tokens=args.max_output_tokens or None,
            temperature=0.0,
            seed=0,
        )
        return case, result

    failures: list[dict[str, str]] = []
    finished_now = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(answer, case): case for case in pending}
        for future in as_completed(futures):
            source = futures[future]
            try:
                case, result = future.result()
            except Exception as error:
                failure = {
                    "question_id": str(source["question_id"]),
                    "benchmark": str(source["benchmark"]),
                    "error": repr(error),
                }
                failures.append(failure)
                append_jsonl(args.output_root / "answer_failures.jsonl", failure)
                print(f"[answer error] {source['question_id']}: {error}", flush=True)
                continue
            response = result.text
            answer_row = {
                key: value
                for key, value in case.items()
                if key not in {"messages", "source_file", "memory_ids"}
            }
            answer_row.update(
                {
                    "prediction": clean_prediction(str(case["benchmark"]), response),
                    "raw_response_sha256": sha256_bytes(response.encode("utf-8")),
                    "answer_model": result.record.model,
                    "answer_reasoning_effort": args.reasoning_effort,
                    "retrieval_setting": f"top-{args.cutoff}",
                }
            )
            usage_row = asdict(result.record)
            usage_row.update(
                {
                    "question_id": str(case["question_id"]),
                    "benchmark": case["benchmark"],
                    "stratum": case["stratum"],
                    "prompt_payload_hash": case["prompt_payload_hash"],
                    "answer_model": result.record.model,
                    "answer_reasoning_effort": args.reasoning_effort,
                }
            )
            append_jsonl(answers_path, answer_row)
            append_jsonl(usage_path, usage_row)
            finished_now += 1
            if finished_now % 25 == 0 or finished_now == len(pending):
                print(f"checkpointed {len(completed) + finished_now}/{len(cases)}", flush=True)

    answers = {str(row["question_id"]): row for row in read_jsonl(answers_path)}
    usage = {str(row["question_id"]): row for row in read_jsonl(usage_path)}
    expected_ids = [str(row["question_id"]) for row in cases]
    complete_ids = set(answers) & set(usage)
    if failures or complete_ids != set(expected_ids):
        raise RuntimeError(
            f"replay incomplete: expected={len(expected_ids)} complete={len(complete_ids)} failures={len(failures)}; rerun with --resume"
        )

    # Normalize checkpoints into deterministic workload order after completion.
    ordered_answers = [answers[question_id] for question_id in expected_ids]
    ordered_usage = [usage[question_id] for question_id in expected_ids]
    answers_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_answers),
        encoding="utf-8",
    )
    usage_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_usage),
        encoding="utf-8",
    )
    for benchmark in ("longmemeval", "locomo"):
        selected = [row for row in ordered_answers if row["benchmark"] == benchmark]
        (args.output_root / f"answers_{benchmark}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )

    token_fields = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
    manifest = {
        "schema_version": "mem0-frozen-retrieval-answer-replay-v1",
        "memory_system": "Mem0",
        "memory_build_model": "Qwen3-30B",
        "retrieval_setting": f"top-{args.cutoff}",
        "answer_model": args.model,
        "answer_reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens or None,
        "temperature": 0.0,
        "seed": 0,
        "questions": len(cases),
        "questions_by_benchmark": {
            name: sum(row["benchmark"] == name for row in cases)
            for name in ("longmemeval", "locomo")
        },
        "memory_benchmarks_commit": MEMORY_BENCHMARKS_COMMIT,
        "prompt_sources": {
            "longmemeval": {"path": str(lme_path), "sha256": LME_PROMPT_SHA256},
            "locomo": {"path": str(locomo_path), "sha256": LOCOMO_PROMPT_SHA256},
        },
        "source_retrieval": {
            "longmemeval": str(args.longmemeval_dir),
            "locomo": str(args.locomo_dir),
        },
        "prepared_requests": str(prepared_path),
        "prepared_sha256": sha256_bytes(prepared_path.read_bytes()),
        "unique_prompt_hashes": len({row["prompt_payload_hash"] for row in cases}),
        "output_truncated": sum(row.get("finish_reason") == "length" for row in ordered_usage),
        "request_retry_count": sum(int(row.get("retry_count") or 0) for row in ordered_usage),
        "api_usage_additivity_ok": all(
            int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
            == int(row.get("total_tokens") or 0)
            for row in ordered_usage
        ),
        "api_tokens_by_benchmark": {
            benchmark: {
                field: nearest_rank_stats(
                    [
                        int(row.get(field) or 0)
                        for row in ordered_usage
                        if row["benchmark"] == benchmark
                    ],
                    "tokens_per_question",
                )
                for field in token_fields
            }
            for benchmark in ("longmemeval", "locomo")
            if any(row["benchmark"] == benchmark for row in ordered_usage)
        },
        "latency_by_benchmark": {
            benchmark: nearest_rank_stats(
                [
                    float(row.get("latency_sec") or 0) * 1000
                    for row in ordered_usage
                    if row["benchmark"] == benchmark
                ],
                "milliseconds_per_question",
            )
            for benchmark in ("longmemeval", "locomo")
            if any(row["benchmark"] == benchmark for row in ordered_usage)
        },
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
