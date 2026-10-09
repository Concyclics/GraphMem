#!/usr/bin/env python3
"""Replay frozen prompts as two n=4 families and select locally from candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.ensemble import (  # noqa: E402
    ENSEMBLE_SCHEMA_VERSION, FAMILIES, build_candidate_verifier_messages,
    build_ensemble_family_messages, deterministic_candidate_fallback,
    normalize_answer, parse_verifier_choice, prompt_payload_hash,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def nearest_stats(values: list[int | float]) -> dict[str, Any]:
    rows = sorted(values)

    def percentile(value: float) -> int | float:
        return rows[max(0, math.ceil(value * len(rows)) - 1)] if rows else 0

    return {
        "count": len(rows), "mean": sum(rows) / max(1, len(rows)),
        "p50": percentile(0.50), "p95": percentile(0.95),
        "p99": percentile(0.99), "max": max(rows, default=0),
        "sum": sum(rows), "percentile_method": "nearest_rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-input", type=Path,
        help="reuse a complete eight-candidate JSONL and run selection only")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--n-per-family", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--verifier-model")
    parser.add_argument("--verifier-base-url")
    parser.add_argument("--verifier-max-tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-retries", type=int, default=0,
                        help="0 waits through recoverable local-service restarts")
    parser.add_argument("--expected-questions", type=int, default=1540)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n_per_family != 4:
        raise ValueError("the V5.73 contract requires n-per-family=4")
    if min(args.max_output_tokens, args.verifier_max_tokens, args.workers) <= 0:
        raise ValueError("token limits and workers must be positive")
    if not 0.0 < args.temperature <= 2.0:
        raise ValueError("temperature must be in (0, 2]")

    prepared_rows = read_jsonl(args.prepared)
    cases = {
        str(row["question_id"]): row
        for row in json.loads(args.locomo_data.read_text(encoding="utf-8"))
        if int(row["locomo_category"]) in {1, 2, 3, 4}
    }
    prepared = {str(row["question_id"]): row for row in prepared_rows
                if str(row["question_id"]) in cases and row.get("messages")}
    if set(prepared) != set(cases):
        missing = sorted(set(cases) - set(prepared))
        raise RuntimeError(
            f"prepared/case mismatch: {len(prepared)} aligned, "
            f"{len(missing)} missing (first={missing[:5]})")
    if args.expected_questions and len(prepared) != args.expected_questions:
        raise RuntimeError(
            f"expected {args.expected_questions} questions, got {len(prepared)}")
    order = [str(row["question_id"]) for row in prepared_rows
             if str(row["question_id"]) in prepared]

    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    candidate_path = args.output_root / "candidates.jsonl"
    generation_usage_path = args.output_root / "generation_usage.jsonl"
    selection_path = args.output_root / "selections.jsonl"
    verifier_usage_path = args.output_root / "verifier_usage.jsonl"
    answer_path = args.output_root / "answers.jsonl"
    if not args.resume:
        for path in (candidate_path, generation_usage_path, selection_path,
                     verifier_usage_path, answer_path):
            path.write_text("", encoding="utf-8")

    client = OpenAI(
        base_url=args.base_url, api_key="local", max_retries=0, timeout=1800.0)
    verifier_client = OpenAI(
        base_url=args.verifier_base_url or args.base_url,
        api_key="local", max_retries=0, timeout=1800.0)

    def retry_call(call: Callable[[], Any]) -> tuple[Any, int]:
        retries = 0
        while True:
            try:
                return call(), retries
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
                if args.max_retries and retries > args.max_retries:
                    raise
                time.sleep(min(30.0, 1.5 * (2 ** min(retries - 1, 5))))

    def usage_row(response: Any) -> dict[str, int]:
        usage = response.usage
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": total}

    def generate(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        question_id = str(row["question_id"])
        all_candidates: list[dict[str, Any]] = []
        family_usage: list[dict[str, Any]] = []
        started = time.perf_counter()
        for family_index, family in enumerate(FAMILIES):
            messages = build_ensemble_family_messages(row["messages"], family)
            request = {
                "model": args.model, "messages": list(messages),
                "n": args.n_per_family, "temperature": args.temperature,
                "top_p": args.top_p, "seed": args.seed + 101 * family_index,
                "max_tokens": args.max_output_tokens,
                "extra_body": {
                    "top_k": args.top_k,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
            family_started = time.perf_counter()
            response, retries = retry_call(
                lambda request=request: client.chat.completions.create(**request))
            choices = sorted(response.choices, key=lambda choice: int(choice.index))
            if len(choices) != args.n_per_family:
                raise RuntimeError(
                    f"{question_id}/{family}: expected 4 choices, got {len(choices)}")
            for rank, choice in enumerate(choices, 1):
                prediction = normalize_answer(choice.message.content or "")
                all_candidates.append({
                    "rank": len(all_candidates) + 1, "family": family,
                    "family_rank": rank, "prediction": prediction,
                    "prediction_sha256": hashlib.sha256(
                        prediction.encode()).hexdigest(),
                    "finish_reason": str(choice.finish_reason or ""),
                })
            family_usage.append({
                "family": family,
                "prompt_payload_hash": prompt_payload_hash(messages),
                **usage_row(response), "retry_count": retries,
                "latency_ms": (time.perf_counter() - family_started) * 1000.0,
            })
        case = cases[question_id]
        return ({
            "question_id": question_id,
            "conversation_id": case.get("locomo_sample_id"),
            "category": int(case["locomo_category"]),
            "memory_id": row.get("memory_id"),
            "base_prompt_payload_hash": row.get("prompt_payload_hash"),
            "candidates": all_candidates,
        }, {
            "question_id": question_id, "families": family_usage,
            "prompt_tokens": sum(item["prompt_tokens"] for item in family_usage),
            "completion_tokens": sum(
                item["completion_tokens"] for item in family_usage),
            "total_tokens": sum(item["total_tokens"] for item in family_usage),
            "retry_count": sum(item["retry_count"] for item in family_usage),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        })

    if args.candidate_input is not None:
        supplied = {str(row["question_id"]): row
                    for row in read_jsonl(args.candidate_input)}
        if set(supplied) != set(prepared):
            raise RuntimeError(
                "candidate input must cover exactly the prepared questions")
        if any(len(row.get("candidates", ())) != 8
               for row in supplied.values()):
            raise RuntimeError("candidate input must contain eight choices per row")
        candidate_rows = [supplied[question_id] for question_id in order]
        generation_rows = [{
            "question_id": question_id, "families": [],
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "retry_count": 0, "latency_ms": 0.0, "reused": True,
        } for question_id in order]
    else:
        old_candidates = {str(row["question_id"]): row
                          for row in read_jsonl(candidate_path)}
        old_generation = {str(row["question_id"]): row
                          for row in read_jsonl(generation_usage_path)}
        generated = set(old_candidates) & set(old_generation)
        pending = [prepared[question_id] for question_id in order
                   if question_id not in generated]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate, row): str(row["question_id"])
                       for row in pending}
            for index, future in enumerate(as_completed(futures), 1):
                candidate, usage = future.result()
                old_candidates[str(candidate["question_id"])] = candidate
                old_generation[str(usage["question_id"])] = usage
                append_jsonl(candidate_path, candidate)
                append_jsonl(generation_usage_path, usage)
                if index % 25 == 0 or index == len(pending):
                    print(
                        f"generated {index}/{len(pending)} pending prompts",
                        flush=True)
        if (set(old_candidates) != set(prepared)
                or set(old_generation) != set(prepared)):
            raise RuntimeError("generation incomplete; rerun with --resume")
        candidate_rows = [old_candidates[question_id] for question_id in order]
        generation_rows = [old_generation[question_id] for question_id in order]
    rewrite_jsonl(candidate_path, candidate_rows)
    rewrite_jsonl(generation_usage_path, generation_rows)

    def select(candidate_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        question_id = str(candidate_row["question_id"])
        candidates = list(candidate_row["candidates"])
        unique: list[dict[str, Any]] = []
        unique_by_hash: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            key = str(candidate["prediction_sha256"])
            if key in unique_by_hash:
                prior = unique_by_hash[key]
                prior["support_count"] += 1
                prior["support_families"] = list(dict.fromkeys((
                    *prior["support_families"],
                    str(candidate.get("family") or "unknown"))))
                continue
            enriched = dict(candidate)
            enriched["support_count"] = 1
            enriched["support_families"] = [
                str(candidate.get("family") or "unknown")]
            unique_by_hash[key] = enriched
            unique.append(enriched)
        if len(unique) == 1:
            chosen = next(
                index for index, candidate in enumerate(candidates)
                if candidate["prediction_sha256"]
                == unique[0]["prediction_sha256"])
            return ({
                "question_id": question_id, "selected_index": chosen,
                "prediction": candidates[chosen]["prediction"],
                "selection_mode": "unanimous", "verifier_raw": "",
            }, {
                "question_id": question_id, "skipped": True,
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "retry_count": 0, "latency_ms": 0.0,
            })
        messages = build_candidate_verifier_messages(
            prepared[question_id]["messages"], unique)
        request = {
            "model": args.verifier_model or args.model,
            "messages": list(messages), "temperature": 0.0, "seed": args.seed,
            "max_tokens": args.verifier_max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        started = time.perf_counter()
        response, retries = retry_call(
            lambda: verifier_client.chat.completions.create(**request))
        raw = response.choices[0].message.content or ""
        unique_index = parse_verifier_choice(raw, len(unique))
        mode = "local_verifier"
        if unique_index is None:
            unique_index = deterministic_candidate_fallback(unique)
            mode = "verifier_parse_fallback_modal"
        selected_candidate = unique[unique_index]
        chosen = next(index for index, candidate in enumerate(candidates)
                      if candidate["prediction_sha256"] ==
                      selected_candidate["prediction_sha256"])
        return ({
            "question_id": question_id, "selected_index": chosen,
            "selected_unique_index": unique_index,
            "prediction": candidates[chosen]["prediction"],
            "selection_mode": mode, "verifier_raw": normalize_answer(raw),
            "verifier_prompt_payload_hash": prompt_payload_hash(messages),
        }, {
            "question_id": question_id, "skipped": False,
            **usage_row(response), "retry_count": retries,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        })

    old_selections = {str(row["question_id"]): row
                      for row in read_jsonl(selection_path)}
    old_verifier = {str(row["question_id"]): row
                    for row in read_jsonl(verifier_usage_path)}
    selected_done = set(old_selections) & set(old_verifier)
    select_pending = [row for row in candidate_rows
                      if str(row["question_id"]) not in selected_done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(select, row): str(row["question_id"])
                   for row in select_pending}
        for index, future in enumerate(as_completed(futures), 1):
            selection, usage = future.result()
            old_selections[str(selection["question_id"])] = selection
            old_verifier[str(usage["question_id"])] = usage
            append_jsonl(selection_path, selection)
            append_jsonl(verifier_usage_path, usage)
            if index % 25 == 0 or index == len(select_pending):
                print(f"selected {index}/{len(select_pending)} pending prompts", flush=True)
    if set(old_selections) != set(prepared) or set(old_verifier) != set(prepared):
        raise RuntimeError("selection incomplete; rerun with --resume")
    selections = [old_selections[question_id] for question_id in order]
    verifier_rows = [old_verifier[question_id] for question_id in order]
    rewrite_jsonl(selection_path, selections)
    rewrite_jsonl(verifier_usage_path, verifier_rows)
    answers = [{
        "question_id": row["question_id"],
        "prediction": row["prediction"],
        "answer_model": args.model,
        "selection_mode": row["selection_mode"],
    } for row in selections]
    rewrite_jsonl(answer_path, answers)

    unique_counts = [len({candidate["prediction_sha256"]
                          for candidate in row["candidates"]})
                     for row in candidate_rows]
    manifest = {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "model": args.model,
        "base_url": args.base_url,
        "verifier_model": args.verifier_model or args.model,
        "verifier_base_url": args.verifier_base_url or args.base_url,
        "families": list(dict.fromkeys(
            str(candidate.get("family") or "unknown")
            for candidate in candidate_rows[0]["candidates"])),
        "n_per_family": args.n_per_family,
        "total_candidates": len(candidate_rows[0]["candidates"]),
        "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "seed": args.seed,
        "max_output_tokens_per_candidate": args.max_output_tokens,
        "questions": len(answers),
        "prepared": str(args.prepared),
        "prepared_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "candidate_input": (str(args.candidate_input)
                            if args.candidate_input is not None else None),
        "candidate_input_sha256": (
            hashlib.sha256(args.candidate_input.read_bytes()).hexdigest()
            if args.candidate_input is not None else None),
        "generation_reused": args.candidate_input is not None,
        "unique_candidates_per_question": nearest_stats(unique_counts),
        "unanimous_questions": sum(value == 1 for value in unique_counts),
        "selection_modes": dict(sorted(__import__("collections").Counter(
            row["selection_mode"] for row in selections).items())),
        "generation_tokens": {
            key: nearest_stats([int(row[f"{key}_tokens"])
                                for row in generation_rows])
            for key in ("prompt", "completion", "total")},
        "verifier_tokens": {
            key: nearest_stats([int(row[f"{key}_tokens"])
                                for row in verifier_rows])
            for key in ("prompt", "completion", "total")},
        "generation_retries": sum(int(row["retry_count"])
                                  for row in generation_rows),
        "verifier_retries": sum(int(row["retry_count"])
                                for row in verifier_rows),
        "output_truncated_candidates": sum(
            candidate["finish_reason"] == "length" for row in candidate_rows
            for candidate in row["candidates"]),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
