#!/usr/bin/env python3
"""Run a durable two-stage Luna typed-readout experiment on frozen evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dotenv import load_dotenv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.typed_readout import (  # noqa: E402
    extract_evidence_blocks,
    make_audit_messages,
    make_answer_messages,
    make_readout_messages,
    parse_json_object,
    route_question,
    select_compact_evidence,
)
from graphmem.judging.clients import OpenAICompatibleClient  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question IDs in {path}")
    return result


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest_messages(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def stats(values: Iterable[int | float]) -> dict[str, int | float]:
    rows = sorted(values)
    if not rows:
        return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "max": 0}
    nearest = lambda p: rows[max(0, math.ceil(p * len(rows)) - 1)]
    return {
        "count": len(rows), "mean": sum(rows) / len(rows),
        "p50": nearest(0.50), "p95": nearest(0.95), "max": max(rows),
    }


def select_questions(
    *, metadata: dict[str, dict[str, Any]], selection: str,
    judge_lme: Path | None, judge_locomo: Path | None,
) -> tuple[list[str], dict[str, str]]:
    canonical = list(metadata)
    if selection == "all":
        return canonical, {question_id: "full" for question_id in canonical}
    if judge_lme is None or judge_locomo is None:
        raise ValueError("pilot selections require both baseline judge files")
    verdicts = {**keyed(judge_lme), **keyed(judge_locomo)}
    if set(verdicts) != set(metadata):
        raise ValueError("baseline answers and judge IDs differ")
    wrong = [question_id for question_id in canonical
             if not bool(verdicts[question_id]["correct"])]
    if selection == "wrong-only":
        return wrong, {question_id: "baseline_wrong" for question_id in wrong}

    # One deterministic, same-stratum correct control for every wrong case.
    wrong_by_stratum: Counter[str] = Counter(
        str(metadata[question_id].get("stratum") or "unknown")
        for question_id in wrong
    )
    correct_by_stratum: dict[str, list[str]] = defaultdict(list)
    for question_id in canonical:
        if bool(verdicts[question_id]["correct"]):
            stratum = str(metadata[question_id].get("stratum") or "unknown")
            correct_by_stratum[stratum].append(question_id)
    controls: list[str] = []
    for stratum, count in sorted(wrong_by_stratum.items()):
        candidates = sorted(
            correct_by_stratum[stratum],
            key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
        )
        if len(candidates) < count:
            raise ValueError(f"not enough correct controls for {stratum}")
        controls.extend(candidates[:count])
    chosen = set(wrong) | set(controls)
    selected = [question_id for question_id in canonical if question_id in chosen]
    roles = {question_id: "baseline_wrong" for question_id in wrong}
    roles.update({question_id: "baseline_correct_control" for question_id in controls})
    return selected, roles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-answers", type=Path, required=True)
    parser.add_argument("--baseline-judge-lme", type=Path)
    parser.add_argument("--baseline-judge-locomo", type=Path)
    parser.add_argument(
        "--selection", choices=("all", "wrong-only", "wrong-plus-controls"),
        default="wrong-plus-controls",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default=os.environ.get(
        "SGAO_BASE_URL", "https://sub2api.sgao.me/v1/"))
    parser.add_argument("--api-key-env", default="SGAO_API_KEY")
    parser.add_argument("--readout-effort", default="medium",
                        choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--answer-effort", default="high",
                        choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--readout-max-tokens", type=int, default=8192)
    parser.add_argument("--answer-max-tokens", type=int, default=8192)
    parser.add_argument("--max-compact-blocks", type=int, default=20)
    parser.add_argument(
        "--baseline-mode", choices=("include", "omit"), default="include",
        help="Whether the final verifier sees the previous answer.",
    )
    parser.add_argument(
        "--answer-strategy", choices=("verify", "audit"), default="verify",
        help="Use a concise verifier or an exhaustive adversarial JSON audit.",
    )
    parser.add_argument(
        "--audit-evidence-order", choices=("source", "lexical"), default="source",
        help="For audit strategy, preserve source layout or group query-overlap evidence first.",
    )
    parser.add_argument(
        "--reuse-readouts", type=Path,
        help="Reuse a complete compatible readout_calls.jsonl instead of calling the API.",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.resume and any(args.output_root.iterdir()):
        raise FileExistsError(f"non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    prepared = keyed(args.prepared)
    metadata = keyed(args.baseline_answers)
    if set(prepared) != set(metadata):
        raise ValueError("prepared and baseline answer IDs differ")
    selected, roles = select_questions(
        metadata=metadata, selection=args.selection,
        judge_lme=args.baseline_judge_lme,
        judge_locomo=args.baseline_judge_locomo,
    )
    selection_rows = [{
        "question_id": question_id,
        "benchmark": metadata[question_id].get("benchmark"),
        "stratum": metadata[question_id].get("stratum"),
        "role": roles[question_id],
    } for question_id in selected]
    write_jsonl(args.output_root / "selection.jsonl", selection_rows)

    local = threading.local()

    def client() -> OpenAICompatibleClient:
        if not hasattr(local, "client"):
            local.client = OpenAICompatibleClient(
                model=args.model, base_url=args.base_url,
                api_key_env=args.api_key_env, request_profile="openai",
                max_retries=12, timeout_sec=600.0,
            )
        return local.client

    readout_path = args.output_root / "readout_calls.jsonl"
    if args.reuse_readouts:
        readouts = keyed(args.reuse_readouts)
        missing = set(selected) - set(readouts)
        extra = set(readouts) - set(selected)
        if missing or extra:
            raise ValueError(
                "reused readout IDs differ from selection: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        write_jsonl(readout_path, (readouts[question_id] for question_id in selected))
    else:
        readouts = keyed(readout_path) if args.resume and readout_path.exists() else {}

    def compile_one(question_id: str) -> dict[str, Any]:
        source = prepared[question_id]
        base = metadata[question_id]
        blocks = extract_evidence_blocks(source.get("messages") or [])
        if not blocks:
            # Certified deterministic PreparedAnswer rows intentionally carry
            # no model messages.  Preserve their already-audited prediction
            # rather than turning a prompt experiment into a regression.
            return {
                "question_id": question_id, "route": "deterministic",
                "pilot_role": roles[question_id], "readout": "{}",
                "readout_valid_json": True, "readout_passthrough": True,
                "readout_prompt_sha256": "",
                "source_prompt_payload_hash": source.get("prompt_payload_hash"),
                "evidence_blocks": 0, "model": args.model,
                "thinking_mode": args.readout_effort, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
                "reasoning_tokens": 0, "retry_count": 0,
                "finish_reason": "deterministic", "latency_sec": 0.0,
            }
        route = route_question(
            str(base.get("question") or ""),
            str(base.get("stratum") or base.get("question_type") or ""),
        )
        messages = make_readout_messages(
            question=str(base.get("question") or ""),
            question_date=str(base.get("question_date") or ""),
            route=route, evidence_blocks=blocks,
        )
        result = client().chat(
            question_id=question_id, variant="v5_67_typed_readout",
            stage="answer_note_extraction", messages=messages,
            thinking_mode=args.readout_effort,
            max_tokens=args.readout_max_tokens, json_mode=True,
            temperature=0.0, seed=0,
        )
        parsed = parse_json_object(result.text)
        return {
            "question_id": question_id, "route": route,
            "pilot_role": roles[question_id], "readout": result.text,
            "readout_valid_json": bool(parsed),
            "readout_prompt_sha256": digest_messages(messages),
            "source_prompt_payload_hash": source.get("prompt_payload_hash"),
            "evidence_blocks": len(blocks), **asdict(result.record),
        }

    pending = (
        [] if args.reuse_readouts else
        [question_id for question_id in selected if question_id not in readouts]
    )
    failures: list[dict[str, str]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(compile_one, question_id): question_id
                       for question_id in pending}
            completed = len(readouts)
            for future in as_completed(futures):
                question_id = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    failures.append({"question_id": question_id,
                                     "stage": "readout", "error": repr(error)})
                    continue
                readouts[question_id] = row
                append_jsonl(readout_path, row, lock)
                completed += 1
                if completed % 25 == 0 or completed == len(selected):
                    print(f"readout {completed}/{len(selected)}", flush=True)
    if failures:
        write_jsonl(args.output_root / "failures.jsonl", failures)
        raise RuntimeError(f"{len(failures)} readout calls failed; rerun --resume")

    answer_path = args.output_root / "answer_calls.jsonl"
    calls = keyed(answer_path) if args.resume and answer_path.exists() else {}

    def answer_one(question_id: str) -> dict[str, Any]:
        source = prepared[question_id]
        base = metadata[question_id]
        readout = readouts[question_id]
        if bool(readout.get("readout_passthrough")):
            return {
                "question_id": question_id, "route": "deterministic",
                "pilot_role": roles[question_id],
                "prediction": str(base.get("prediction") or ""),
                "changed_from_baseline": False, "answer_prompt_sha256": "",
                "source_prompt_payload_hash": source.get("prompt_payload_hash"),
                "compact_evidence_ids": [], "compact_evidence_blocks": 0,
                "model": args.model, "thinking_mode": args.answer_effort,
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "reasoning_tokens": 0,
                "retry_count": 0, "finish_reason": "deterministic",
                "latency_sec": 0.0,
            }
        parsed = parse_json_object(str(readout.get("readout") or ""))
        blocks = extract_evidence_blocks(source.get("messages") or [])
        if args.answer_strategy == "audit":
            compact = (
                select_compact_evidence(
                    question=str(base.get("question") or ""), readout={},
                    evidence_blocks=blocks, max_blocks=args.max_compact_blocks,
                )
                if args.audit_evidence_order == "lexical"
                else list(enumerate(blocks[:args.max_compact_blocks], 1))
            )
            messages = make_audit_messages(
                question=str(base.get("question") or ""),
                question_date=str(base.get("question_date") or ""),
                route=str(readout["route"]),
                baseline=(
                    str(base.get("prediction") or "")
                    if args.baseline_mode == "include" else None
                ),
                readout_text=str(readout.get("readout") or "{}"),
                evidence_blocks=compact,
            )
        else:
            compact = select_compact_evidence(
                question=str(base.get("question") or ""), readout=parsed,
                evidence_blocks=blocks, max_blocks=args.max_compact_blocks,
            )
            messages = make_answer_messages(
                question=str(base.get("question") or ""),
                question_date=str(base.get("question_date") or ""),
                route=str(readout["route"]),
                baseline=(
                    str(base.get("prediction") or "")
                    if args.baseline_mode == "include" else None
                ),
                readout_text=str(readout.get("readout") or "{}"),
                compact_evidence=compact,
            )
        result = client().chat(
            question_id=question_id,
            variant=("v5_67_adversarial_audit"
                     if args.answer_strategy == "audit"
                     else "v5_67_typed_verify_answer"),
            stage=("answer_audit" if args.answer_strategy == "audit" else "answer"),
            messages=messages,
            thinking_mode=args.answer_effort,
            max_tokens=args.answer_max_tokens,
            json_mode=args.answer_strategy == "audit",
            temperature=0.0, seed=0,
        )
        answer_audit = (
            parse_json_object(result.text) if args.answer_strategy == "audit" else {}
        )
        audit_answer = answer_audit.get("final_answer")
        if isinstance(audit_answer, (dict, list)):
            audit_answer = json.dumps(audit_answer, ensure_ascii=False)
        prediction = " ".join(
            str(audit_answer if audit_answer is not None else result.text).split()
        )
        if not prediction:
            raise RuntimeError(f"empty answer for {question_id}")
        return {
            "question_id": question_id, "route": readout["route"],
            "pilot_role": roles[question_id], "prediction": prediction,
            "changed_from_baseline": prediction != str(base.get("prediction") or ""),
            "answer_prompt_sha256": digest_messages(messages),
            "source_prompt_payload_hash": source.get("prompt_payload_hash"),
            "compact_evidence_ids": [f"E{index:02d}" for index, _ in compact],
            "compact_evidence_blocks": len(compact),
            "answer_strategy": args.answer_strategy,
            "answer_audit_valid_json": (
                bool(answer_audit) if args.answer_strategy == "audit" else None
            ),
            "answer_audit": result.text if args.answer_strategy == "audit" else None,
            **asdict(result.record),
        }

    pending = [question_id for question_id in selected if question_id not in calls]
    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(answer_one, question_id): question_id
                       for question_id in pending}
            completed = len(calls)
            for future in as_completed(futures):
                question_id = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    failures.append({"question_id": question_id,
                                     "stage": "answer", "error": repr(error)})
                    continue
                calls[question_id] = row
                append_jsonl(answer_path, row, lock)
                completed += 1
                if completed % 25 == 0 or completed == len(selected):
                    print(f"answer {completed}/{len(selected)}", flush=True)
    if failures:
        write_jsonl(args.output_root / "failures.jsonl", failures)
        raise RuntimeError(f"{len(failures)} answer calls failed; rerun --resume")

    answers: list[dict[str, Any]] = []
    for question_id in selected:
        output = dict(metadata[question_id])
        call = calls[question_id]
        output.update({
            "prediction": call["prediction"], "answer_model": args.model,
            "answer_reasoning_effort": args.answer_effort,
            "prompt_payload_hash": call["answer_prompt_sha256"],
            "source_prompt_payload_hash": call["source_prompt_payload_hash"],
            "typed_readout_route": call["route"],
            "pilot_role": roles[question_id],
        })
        answers.append(output)
    write_jsonl(args.output_root / "answers.jsonl", answers)
    write_jsonl(args.output_root / "answers_longmemeval.jsonl", (
        row for row in answers if row.get("benchmark") == "longmemeval"))
    write_jsonl(args.output_root / "answers_locomo.jsonl", (
        row for row in answers if row.get("benchmark") == "locomo"))

    all_records = list(readouts.values()) + list(calls.values())
    manifest = {
        "schema_version": "graphmem-v5.67-typed-readout-v1",
        "selection": args.selection, "questions": len(selected),
        "selection_roles": dict(Counter(roles.values())),
        "model": args.model, "readout_effort": args.readout_effort,
        "answer_effort": args.answer_effort,
        "readout_max_tokens": args.readout_max_tokens,
        "answer_max_tokens": args.answer_max_tokens,
        "max_compact_blocks": args.max_compact_blocks,
        "baseline_mode": args.baseline_mode,
        "answer_strategy": args.answer_strategy,
        "audit_evidence_order": args.audit_evidence_order,
        "reused_readouts": str(args.reuse_readouts) if args.reuse_readouts else None,
        "source_prepared": str(args.prepared),
        "source_prepared_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "uses_gold_in_prompts": False,
        "uses_judge_only_for_pilot_selection": args.selection != "all",
        "readout_valid_json": sum(bool(row["readout_valid_json"])
                                  for row in readouts.values()),
        "changed_from_baseline": sum(bool(row["changed_from_baseline"])
                                     for row in calls.values()),
        "routes": dict(Counter(str(row["route"]) for row in calls.values())),
        "request_retries": sum(int(row.get("retry_count") or 0)
                               for row in all_records),
        "finish_reason_length": sum(str(row.get("finish_reason")) == "length"
                                    for row in all_records),
        "readout_total_tokens": stats(
            int(row.get("total_tokens") or 0) for row in readouts.values()),
        "answer_total_tokens": stats(
            int(row.get("total_tokens") or 0) for row in calls.values()),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
