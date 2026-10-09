#!/usr/bin/env python3
"""Select from a V5.76 adaptive candidate pool without benchmark labels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.budget_controller import (  # noqa: E402
    BUDGET_CONTROLLER_SCHEMA_VERSION, answer_signature,
    modal_candidate_choice, reliable_candidate_choice,
    stage_normalized_candidate_choice,
)
from graphmem.answer.ensemble import (  # noqa: E402
    build_candidate_verifier_messages, deterministic_candidate_fallback,
    normalize_answer, parse_verifier_choice, prompt_payload_hash,
)


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in rows(path)}


def rewrite(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in values), encoding="utf-8")


def append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def nearest(values: list[int | float], fraction: float) -> int | float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0


def stats(values: list[int | float]) -> dict[str, Any]:
    return {
        "count": len(values), "mean": sum(values) / max(1, len(values)),
        "p50": nearest(values, 0.50), "p95": nearest(values, 0.95),
        "p99": nearest(values, 0.99), "max": max(values, default=0),
        "sum": sum(values), "percentile_method": "nearest_rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7600)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument(
        "--unresolved-policy",
        choices=("verifier", "modal", "stage_normalized"),
        default="verifier")
    parser.add_argument("--expected", type=int, default=1540)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.max_tokens) <= 0:
        raise ValueError("workers and max-tokens must be positive")

    prepared_rows = rows(args.prepared)
    candidate_rows = rows(args.candidates)
    prepared = {str(row["question_id"]): row for row in prepared_rows}
    candidates = {str(row["question_id"]): row for row in candidate_rows}
    if set(prepared) != set(candidates):
        raise RuntimeError("prepared and candidate question sets differ")
    if args.expected and len(prepared) != args.expected:
        raise RuntimeError(f"expected {args.expected}, got {len(prepared)}")
    order = [str(row["question_id"]) for row in candidate_rows]

    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    selection_path = args.output_root / "selections.jsonl"
    usage_path = args.output_root / "verifier_usage.jsonl"
    answer_path = args.output_root / "answers.jsonl"
    if not args.resume:
        selection_path.write_text("", encoding="utf-8")
        usage_path.write_text("", encoding="utf-8")
    old_selection = keyed(selection_path)
    old_usage = keyed(usage_path)
    completed = set(old_selection) & set(old_usage)

    client = OpenAI(
        base_url=args.base_url, api_key="local", max_retries=0, timeout=1800.0)

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
                time.sleep(min(30.0, 1.5 * 2 ** min(retries - 1, 5)))

    def select(question_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        choices = list(candidates[question_id].get("candidates", ()))
        if not choices:
            raise RuntimeError(f"empty candidate pool for {question_id}")
        consensus = reliable_candidate_choice(choices)
        if consensus is not None:
            return ({
                "question_id": question_id,
                "selected_index": consensus,
                "prediction": normalize_answer(
                    str(choices[consensus].get("prediction") or "")),
                "selection_mode": "reliable_budget_consensus",
                "verifier_raw": "",
            }, {
                "question_id": question_id, "skipped": True,
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "retry_count": 0, "latency_ms": 0.0,
            })
        if args.unresolved_policy != "verifier":
            selected = (
                modal_candidate_choice(choices)
                if args.unresolved_policy == "modal"
                else stage_normalized_candidate_choice(choices))
            return ({
                "question_id": question_id,
                "selected_index": selected,
                "prediction": normalize_answer(
                    str(choices[selected].get("prediction") or "")),
                "selection_mode": f"{args.unresolved_policy}_fallback",
                "verifier_raw": "",
            }, {
                "question_id": question_id, "skipped": True,
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "retry_count": 0, "latency_ms": 0.0,
            })

        unique: list[dict[str, Any]] = []
        signature_to_unique: dict[str, dict[str, Any]] = {}
        for choice in choices:
            signature = answer_signature(str(choice.get("prediction") or ""))
            if not signature:
                continue
            if signature in signature_to_unique:
                prior = signature_to_unique[signature]
                prior["support_count"] += 1
                prior["support_families"] = list(dict.fromkeys((
                    *prior["support_families"],
                    str(choice.get("family") or "unknown"))))
                continue
            enriched = dict(choice)
            enriched["support_count"] = 1
            enriched["support_families"] = [
                str(choice.get("family") or "unknown")]
            signature_to_unique[signature] = enriched
            unique.append(enriched)
        if not unique:
            raise RuntimeError(f"all candidate predictions empty for {question_id}")
        messages = build_candidate_verifier_messages(
            prepared[question_id]["messages"], unique)
        request = {
            "model": args.model,
            "messages": list(messages),
            "temperature": 0.0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        started = time.perf_counter()
        response, retries = retry_call(
            lambda: client.chat.completions.create(**request))
        raw = response.choices[0].message.content or ""
        unique_index = parse_verifier_choice(raw, len(unique))
        mode = "evidence_verifier"
        if unique_index is None:
            unique_index = deterministic_candidate_fallback(unique)
            mode = "verifier_parse_fallback_modal"
        selected_signature = answer_signature(
            str(unique[unique_index].get("prediction") or ""))
        selected = next(
            index for index, choice in enumerate(choices)
            if answer_signature(str(choice.get("prediction") or ""))
            == selected_signature)
        usage = response.usage
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        return ({
            "question_id": question_id,
            "selected_index": selected,
            "selected_unique_index": unique_index,
            "prediction": normalize_answer(
                str(choices[selected].get("prediction") or "")),
            "selection_mode": mode,
            "verifier_raw": normalize_answer(raw),
            "verifier_prompt_payload_hash": prompt_payload_hash(messages),
        }, {
            "question_id": question_id, "skipped": False,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": int(getattr(
                usage, "total_tokens", 0) or prompt + completion),
            "retry_count": retries,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        })

    pending = [question_id for question_id in order
               if question_id not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(select, question_id): question_id
                   for question_id in pending}
        for index, future in enumerate(as_completed(futures), 1):
            selection, usage = future.result()
            question_id = str(selection["question_id"])
            old_selection[question_id] = selection
            old_usage[question_id] = usage
            append(selection_path, selection)
            append(usage_path, usage)
            if index % 25 == 0 or index == len(pending):
                print(f"selected {index}/{len(pending)}", flush=True)
    if set(old_selection) != set(prepared) or set(old_usage) != set(prepared):
        raise RuntimeError("selection incomplete; rerun with --resume")
    selections = [old_selection[question_id] for question_id in order]
    usage_rows = [old_usage[question_id] for question_id in order]
    rewrite(selection_path, selections)
    rewrite(usage_path, usage_rows)
    answers = [{
        "question_id": row["question_id"],
        "prediction": row["prediction"],
        "answer_model": args.model,
        "selection_mode": row["selection_mode"],
    } for row in selections]
    rewrite(answer_path, answers)
    manifest = {
        "schema_version": BUDGET_CONTROLLER_SCHEMA_VERSION,
        "questions": len(answers),
        "selection_uses_gold_or_judge": False,
        "model": args.model,
        "base_url": args.base_url,
        "unresolved_policy": args.unresolved_policy,
        "selection_modes": dict(sorted(Counter(
            row["selection_mode"] for row in selections).items())),
        "verifier_tokens": {
            key: stats([int(row[f"{key}_tokens"]) for row in usage_rows])
            for key in ("prompt", "completion", "total")
        },
        "verifier_retries": sum(int(row["retry_count"])
                                for row in usage_rows),
        "prepared": str(args.prepared),
        "prepared_sha256": hashlib.sha256(
            args.prepared.read_bytes()).hexdigest(),
        "candidates": str(args.candidates),
        "candidates_sha256": hashlib.sha256(
            args.candidates.read_bytes()).hexdigest(),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
