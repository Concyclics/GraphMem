#!/usr/bin/env python3
"""Replay frozen V5.54 failure cases across model/reasoning settings.

This diagnostic deliberately changes only ``model`` and ``reasoning_effort``.
The prepared message payload, evidence order, temperature, seed, and output cap
remain identical across arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


DEFAULT_CASES = (
    "6cb6f249",          # multi-session aggregation
    "a11281a2",          # delta versus terminal value
    "e3038f8c",          # operand closure
    "51c32626",          # event time versus observation time
    "gpt4_7bc6cf22",     # date difference
    "3ba21379",          # latest-state selection
    "f685340e_abs",      # exact entity / near match
    "75832dbd",          # preference synthesis
    "gpt4_5501fe77",     # cross-session maximum
    "85fa3a3f",          # four-operand sum
    "gpt4_2c50253f",     # directional time arithmetic
    "gpt4_93159ced_abs", # false-premise rejection
    "gpt4_fe651585_abs", # open-world comparison
    "6456829e_abs",      # missing operand
)

ARMS = (
    ("gpt54_none", "gpt-5.4-mini", "none"),
    ("gpt54_medium", "gpt-5.4-mini", "medium"),
    ("luna_none", "gpt-5.6-luna", "none"),
    ("luna_medium", "gpt-5.6-luna", "medium"),
    ("sol_none", "gpt-5.6-sol", "none"),
    ("sol_medium", "gpt-5.6-sol", "medium"),
    ("sol_high", "gpt-5.6-sol", "high"),
    ("sol_max", "gpt-5.6-sol", "max"),
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


def nearest(values: list[int | float], percentile: float) -> int | float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize(values: list[int | float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": nearest(values, 0.50),
        "p95": nearest(values, 0.95),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--metadata-answers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get(
        "SGAO_BASE_URL", "https://sub2api.sgao.me/v1"))
    parser.add_argument("--api-key-env", default="SGAO_API_KEY")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--arm", action="append", dest="arms",
        choices=tuple(row[0] for row in ARMS),
        help="run only the named arm; repeat to select multiple arms")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required")
    selected_ids = tuple(args.cases or DEFAULT_CASES)
    selected_arm_names = set(args.arms or (row[0] for row in ARMS))
    selected_arms = tuple(row for row in ARMS if row[0] in selected_arm_names)
    prepared = {str(row["question_id"]): row
                for row in read_jsonl(args.prepared)}
    metadata = {str(row["question_id"]): row
                for row in read_jsonl(args.metadata_answers)}
    missing = set(selected_ids) - prepared.keys()
    if missing:
        raise ValueError(f"prepared artifact is missing: {sorted(missing)}")
    missing = set(selected_ids) - metadata.keys()
    if missing:
        raise ValueError(f"metadata artifact is missing: {sorted(missing)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def client() -> OpenAI:
        if not hasattr(local, "client"):
            local.client = OpenAI(base_url=args.base_url, api_key=api_key)
        return local.client

    def run_one(arm: str, model: str, effort: str, question_id: str) -> dict:
        frozen = prepared[question_id]
        request = {
            "model": model,
            "messages": frozen["messages"],
            "temperature": 0,
            "seed": 0,
            "reasoning_effort": effort,
            "max_completion_tokens": args.max_output_tokens,
        }
        started = time.perf_counter()
        retries = 0
        for attempt in range(12):
            try:
                response = client().chat.completions.create(**request)
                break
            except Exception as error:
                status = getattr(error, "status_code", None)
                recoverable = (
                    error.__class__.__name__ in {
                        "APIConnectionError", "APITimeoutError",
                        "InternalServerError", "RateLimitError",
                    }
                    or (isinstance(status, int)
                        and (status in {408, 409, 429} or status >= 500)))
                if not recoverable or attempt == 11:
                    raise
                retries += 1
                time.sleep(min(8.0, float(2 ** attempt)))
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        prediction = " ".join((choice.message.content or "").split())
        return {
            "arm": arm,
            "question_id": question_id,
            "model": model,
            "reasoning_effort": effort,
            "prediction": prediction,
            "prompt_payload_hash": frozen["prompt_payload_hash"],
            "messages_sha256": hashlib.sha256(json.dumps(
                frozen["messages"], sort_keys=True, ensure_ascii=False,
                separators=(",", ":")).encode()).hexdigest(),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "finish_reason": str(choice.finish_reason or ""),
            "retry_count": retries,
        }

    jobs = [(arm, model, effort, question_id)
            for arm, model, effort in selected_arms
            for question_id in selected_ids]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"completed {row['arm']} {row['question_id']}", flush=True)

    order = {question_id: index for index, question_id in enumerate(selected_ids)}
    for arm, model, effort in selected_arms:
        arm_rows = sorted(
            (row for row in results if row["arm"] == arm),
            key=lambda row: order[row["question_id"]])
        answer_rows = []
        for row in arm_rows:
            base = dict(metadata[row["question_id"]])
            base.update({
                "prediction": row["prediction"],
                "answer_model": model,
                "answer_reasoning_effort": effort,
                "prompt_payload_hash": row["prompt_payload_hash"],
            })
            answer_rows.append(base)
        arm_root = args.output_root / arm
        write_jsonl(arm_root / "calls.jsonl", arm_rows)
        write_jsonl(arm_root / "answers_longmemeval.jsonl", answer_rows)

    hashes = {row["question_id"]: set() for row in results}
    for row in results:
        hashes[row["question_id"]].add(row["messages_sha256"])
    manifest = {
        "schema_version": "graphmem-v5.54-reasoning-case-study-v1",
        "prepared": str(args.prepared),
        "selected_question_ids": list(selected_ids),
        "questions": len(selected_ids),
        "temperature": 0,
        "seed": 0,
        "max_output_tokens": args.max_output_tokens,
        "arms": {},
        "prompt_identity_audit": {
            "all_arms_exactly_match": all(len(value) == 1 for value in hashes.values()),
            "mismatched_question_ids": [key for key, value in hashes.items()
                                        if len(value) != 1],
        },
    }
    for arm, model, effort in selected_arms:
        rows = [row for row in results if row["arm"] == arm]
        manifest["arms"][arm] = {
            "model": model,
            "reasoning_effort": effort,
            "prompt_tokens": summarize([row["prompt_tokens"] for row in rows]),
            "completion_tokens": summarize(
                [row["completion_tokens"] for row in rows]),
            "reasoning_tokens": summarize(
                [row["reasoning_tokens"] for row in rows]),
            "total_tokens": summarize([row["total_tokens"] for row in rows]),
            "latency_ms": summarize([row["latency_ms"] for row in rows]),
            "retries": sum(row["retry_count"] for row in rows),
            "truncated": sum(row["finish_reason"] == "length" for row in rows),
        }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
