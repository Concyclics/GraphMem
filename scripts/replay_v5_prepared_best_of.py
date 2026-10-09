#!/usr/bin/env python3
"""Generate N sampled answers per frozen prompt in one local API request."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


def stats(values: list[int]) -> dict[str, Any]:
    rows = sorted(values)

    def nearest(p: float) -> int:
        return rows[max(0, math.ceil(p * len(rows)) - 1)] if rows else 0

    return {
        "count": len(rows),
        "mean": sum(rows) / max(1, len(rows)),
        "p50": nearest(0.50),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": max(rows, default=0),
        "sum": sum(rows),
        "percentile_method": "nearest_rank",
    }


def normalized(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7208)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--max-inflight-choices", type=int, default=256,
        help=("cap workers*n so multi-choice decoding cannot create an "
              "unbounded number of concurrent sequences; zero disables"))
    parser.add_argument(
        "--max-recoverable-retries", type=int, default=0,
        help=("maximum connection/timeout/429/5xx retries per question; "
              "zero waits indefinitely for an auto-restarting service"))
    parser.add_argument(
        "--expected-questions", type=int, default=1540,
        help="Expected prepared rows; permits audited subset continuations.")
    parser.add_argument(
        "--reuse-root", type=Path, action="append", default=[],
        help=("reuse candidate/usage rows from a prior compatible run when "
              "the exact prompt payload hash matches; may be repeated"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n <= 0 or args.max_output_tokens <= 0 or args.workers <= 0:
        raise ValueError("n, max-output-tokens and workers must be positive")
    if args.max_inflight_choices < 0:
        raise ValueError("max-inflight-choices must be non-negative")
    if args.max_recoverable_retries < 0:
        raise ValueError("max-recoverable-retries must be non-negative")
    if not 0.0 < args.temperature <= 2.0:
        raise ValueError("Best-of sampling temperature must be in (0, 2]")

    prepared_rows = read_jsonl(args.prepared)
    cases = {
        str(row["question_id"]): row
        for row in json.loads(args.locomo_data.read_text(encoding="utf-8"))
        if int(row["locomo_category"]) in {1, 2, 3, 4}
    }
    aligned_rows = [
        row for row in prepared_rows if str(row["question_id"]) in cases]
    prepared = {str(row["question_id"]): row for row in aligned_rows}
    if len(prepared) != len(aligned_rows):
        raise RuntimeError("prepared prompts contain duplicate question IDs")
    if (args.expected_questions and
            len(prepared) != args.expected_questions):
        raise RuntimeError(
            f"expected {args.expected_questions} aligned LoCoMo prompts, "
            f"got {len(prepared)}")
    if not set(prepared).issubset(cases):
        raise RuntimeError("prepared prompt IDs are not a LoCoMo subset")

    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    candidate_path = args.output_root / "candidates.jsonl"
    usage_path = args.output_root / "usage.jsonl"
    candidates = read_jsonl(candidate_path) if args.resume else []
    usage = read_jsonl(usage_path) if args.resume else []
    if not args.resume:
        candidate_path.write_text("", encoding="utf-8")
        usage_path.write_text("", encoding="utf-8")
    reused_sources: list[str] = []
    reused_count = 0
    candidate_by_id = {str(row["question_id"]): row for row in candidates}
    usage_by_id = {str(row["question_id"]): row for row in usage}
    for reuse_root in args.reuse_root:
        manifest_path = reuse_root / "run_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"missing reuse manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_contract = {
            "model": args.model,
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "max_output_tokens_per_choice": args.max_output_tokens,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected_contract.items()
            if manifest.get(key) != value}
        if mismatches:
            raise RuntimeError(
                f"incompatible reuse contract at {reuse_root}: {mismatches}")
        reusable_candidates = {
            str(row["question_id"]): row
            for row in read_jsonl(reuse_root / "candidates.jsonl")}
        reusable_usage = {
            str(row["question_id"]): row
            for row in read_jsonl(reuse_root / "usage.jsonl")}
        for question_id in set(reusable_candidates) & set(reusable_usage):
            if question_id not in prepared or question_id in candidate_by_id:
                continue
            prompt_hash = str(
                prepared[question_id].get("prompt_payload_hash") or "")
            candidate_row = reusable_candidates[question_id]
            usage_row = reusable_usage[question_id]
            if (str(candidate_row.get("prompt_payload_hash") or "") != prompt_hash
                    or str(usage_row.get("prompt_payload_hash") or "") != prompt_hash):
                continue
            candidate_by_id[question_id] = candidate_row
            usage_by_id[question_id] = usage_row
            reused_count += 1
        reused_sources.append(str(reuse_root))
    candidates = list(candidate_by_id.values())
    usage = list(usage_by_id.values())
    if args.reuse_root:
        write_jsonl(candidate_path, candidates)
        write_jsonl(usage_path, usage)
    completed = ({str(row["question_id"]) for row in candidates}
                 & {str(row["question_id"]) for row in usage})
    preexisting_completed_count = len(completed)
    order = [str(row["question_id"]) for row in prepared_rows
             if str(row["question_id"]) in prepared]
    pending = [prepared[question_id] for question_id in order
               if question_id not in completed]

    client = OpenAI(
        base_url=args.base_url, api_key="local", max_retries=0, timeout=1800.0)

    def generate(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        question_id = str(row["question_id"])
        request = {
            "model": args.model,
            "messages": row["messages"],
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_tokens": args.max_output_tokens,
            "extra_body": {
                "top_k": args.top_k,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        started = time.perf_counter()
        retries = 0
        while True:
            try:
                response = client.chat.completions.create(**request)
                break
            except Exception as error:
                status = getattr(error, "status_code", None)
                recoverable = (
                    error.__class__.__name__ in {
                        "APIConnectionError", "APITimeoutError",
                        "InternalServerError", "RateLimitError"}
                    or (isinstance(status, int)
                        and (status in {408, 409, 429} or status >= 500)))
                if not recoverable:
                    raise
                retries += 1
                if (args.max_recoverable_retries
                        and retries > args.max_recoverable_retries):
                    raise
                # A small stable per-question offset avoids a synchronized
                # retry wave when a local vLLM service comes back online.
                jitter = int(hashlib.sha256(
                    question_id.encode()).hexdigest()[:2], 16) / 255.0
                time.sleep(min(15.0, 2.0 ** min(retries - 1, 4)) + jitter)
        choices = sorted(response.choices, key=lambda choice: int(choice.index))
        if len(choices) != args.n:
            raise RuntimeError(
                f"{question_id}: expected {args.n} choices, got {len(choices)}")
        candidate_rows = []
        for rank, choice in enumerate(choices, start=1):
            prediction = normalized(choice.message.content or "")
            candidate_rows.append({
                "rank": rank,
                "prediction": prediction,
                "prediction_sha256": hashlib.sha256(
                    prediction.encode("utf-8")).hexdigest(),
                "finish_reason": str(choice.finish_reason or ""),
            })
        raw_usage = response.usage
        prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(raw_usage, "total_tokens", 0)
                           or prompt_tokens + completion_tokens)
        case = cases[question_id]
        return ({
            "question_id": question_id,
            "conversation_id": case["locomo_sample_id"],
            "category": int(case["locomo_category"]),
            "memory_id": row.get("memory_id"),
            "prompt_payload_hash": row.get("prompt_payload_hash"),
            "candidates": candidate_rows,
        }, {
            "question_id": question_id,
            "prompt_payload_hash": row.get("prompt_payload_hash"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "retry_count": retries,
        })

    effective_workers = args.workers
    if args.max_inflight_choices:
        effective_workers = min(
            effective_workers,
            max(1, args.max_inflight_choices // args.n))
    if effective_workers != args.workers:
        print(
            f"capped workers {args.workers}->{effective_workers} for "
            f"n={args.n} and max_inflight_choices="
            f"{args.max_inflight_choices}", flush=True)

    new_candidates: list[dict[str, Any]] = []
    new_usage: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(generate, row): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            candidate_row, usage_row = future.result()
            new_candidates.append(candidate_row)
            new_usage.append(usage_row)
            with candidate_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(candidate_row, ensure_ascii=False) + "\n")
            with usage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(usage_row, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(pending):
                print(f"generated {index}/{len(pending)} pending prompts", flush=True)

    by_candidate = {str(row["question_id"]): row
                    for row in [*candidates, *new_candidates]}
    by_usage = {str(row["question_id"]): row for row in [*usage, *new_usage]}
    if set(by_candidate) != set(prepared) or set(by_usage) != set(prepared):
        raise RuntimeError("Best-of generation is incomplete; rerun with --resume")
    candidates = [by_candidate[question_id] for question_id in order]
    usage = [by_usage[question_id] for question_id in order]
    write_jsonl(candidate_path, candidates)
    write_jsonl(usage_path, usage)

    unique_counts = [len({candidate["prediction_sha256"]
                          for candidate in row["candidates"]})
                     for row in candidates]
    manifest = {
        "schema_version": "graphmem-local-locomo-best-of-v1",
        "interpretation": (
            "One local request per question with n choices; API usage is the "
            "actual shared-prefill n-choice charge."),
        "model": args.model,
        "base_url": args.base_url,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_output_tokens_per_choice": args.max_output_tokens,
        "requested_workers": args.workers,
        "effective_workers": effective_workers,
        "max_inflight_choices": args.max_inflight_choices or None,
        "max_recoverable_retries": (
            args.max_recoverable_retries or None),
        "questions": len(candidates),
        "expected_questions": args.expected_questions,
        "prepared": str(args.prepared),
        "prepared_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "reused_prompt_rows": preexisting_completed_count,
        "reuse_rows_imported_this_invocation": reused_count,
        "generated_prompt_rows": len(new_candidates),
        "reuse_roots": reused_sources,
        "prompt_hash_mismatches": sum(
            str(row.get("prompt_payload_hash") or "") !=
            str(prepared[str(row["question_id"])].get("prompt_payload_hash") or "")
            for row in candidates),
        "unique_candidates_per_question": stats(unique_counts),
        "all_eight_identical": sum(value == 1 for value in unique_counts),
        "all_choices_identical": sum(value == 1 for value in unique_counts),
        "api_tokens_per_n8_request": {
            "prompt": stats([int(row["prompt_tokens"]) for row in usage]),
            "completion_all_choices": stats([
                int(row["completion_tokens"]) for row in usage]),
            "total": stats([int(row["total_tokens"]) for row in usage]),
        },
        "api_tokens_per_request": {
            "prompt": stats([int(row["prompt_tokens"]) for row in usage]),
            "completion_all_choices": stats([
                int(row["completion_tokens"]) for row in usage]),
            "total": stats([int(row["total_tokens"]) for row in usage]),
        },
        "api_usage_sums": {
            key: sum(int(row[f"{key}_tokens"]) for row in usage)
            for key in ("prompt", "completion", "total")
        },
        "retry_count": sum(int(row["retry_count"]) for row in usage),
        "output_truncated_choices": sum(
            candidate["finish_reason"] == "length" for row in candidates
            for candidate in row["candidates"]),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
