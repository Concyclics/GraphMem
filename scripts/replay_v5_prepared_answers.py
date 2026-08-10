#!/usr/bin/env python3
"""Replay frozen GraphMem answer prompts on another answer backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer import AnswerConfig, AnswerStage, PreparedAnswer  # noqa: E402
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


def stats(values) -> dict:
    rows = sorted(int(value) for value in values)
    def nearest(p: float) -> int:
        return rows[max(0, math.ceil(p * len(rows)) - 1)] if rows else 0
    return {
        "count": len(rows), "mean": sum(rows) / max(1, len(rows)),
        "p50": nearest(0.50), "p95": nearest(0.95),
        "p99": nearest(0.99), "max": max(rows, default=0),
        "unit": "tokens_per_question", "percentile_method": "nearest_rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--metadata-answers", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--answer-base-url", required=True)
    parser.add_argument("--answer-api-key-env", required=True)
    parser.add_argument("--answer-request-profile",
                        choices=("qwen", "openai", "omit"), default="openai")
    parser.add_argument("--packing-model")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    answer_path = args.output_root / "answers.jsonl"
    usage_path = args.output_root / "answer_usage.jsonl"
    answer_checkpoint = read_jsonl(answer_path) if args.resume else []
    usage_checkpoint = read_jsonl(usage_path) if args.resume else []
    completed = ({str(row["question_id"]) for row in answer_checkpoint}
                 & {str(row["question_id"]) for row in usage_checkpoint})
    prepared = [PreparedAnswer.from_record(row) for row in read_jsonl(args.prepared)]
    prepared_ids = [row.question_id for row in prepared]
    if len(prepared_ids) != len(set(prepared_ids)):
        raise ValueError("prepared artifact contains duplicate question IDs")
    metadata = {str(row["question_id"]): row for row in read_jsonl(args.metadata_answers)}
    missing = {row.question_id for row in prepared} - set(metadata)
    if missing:
        raise ValueError(f"metadata missing {len(missing)} prepared questions")
    mismatched_hashes = [
        row.question_id for row in prepared
        if str(metadata[row.question_id].get("prompt_payload_hash") or "")
        != row.prompt_payload_hash
    ]
    if mismatched_hashes:
        raise ValueError(
            "metadata/prepared prompt hash mismatch for "
            f"{len(mismatched_hashes)} questions; first={mismatched_hashes[0]}")
    if args.resume:
        order = [row.question_id for row in prepared if row.question_id in completed]
        for path, checkpoint in ((answer_path, answer_checkpoint),
                                 (usage_path, usage_checkpoint)):
            by_id = {str(row["question_id"]): row for row in checkpoint
                     if str(row["question_id"]) in completed}
            path.write_text("".join(
                json.dumps(by_id[question_id], ensure_ascii=True) + "\n"
                for question_id in order), encoding="utf-8")
    pending = [row for row in prepared if row.question_id not in completed]

    store = SQLiteGraphStore(args.source_db, read_only=True)
    cache = SQLiteGraphStore(args.output_root / "answer_cache.sqlite")
    config = load_config(args.config)
    stage = AnswerStage(
        store, config, "v5-prepared-replay",
        answer_config=AnswerConfig(max_output_tokens=None),
        cache_store=cache, require_exact_tokenizer=True,
        answer_model=args.answer_model, answer_base_url=args.answer_base_url,
        answer_api_key_env=args.answer_api_key_env,
        answer_request_profile=args.answer_request_profile,
        packing_model=args.packing_model)

    batch_size = max(1, args.checkpoint_every)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            answers = list(pool.map(stage.complete, batch))
        answer_rows = []
        usage_rows = []
        for frozen, answer in zip(batch, answers):
            base = dict(metadata[frozen.question_id])
            base.update({
                "prediction": answer.prediction,
                "answer_model": answer.answer_model,
                "prompt_payload_hash": answer.prompt_payload_hash,
            })
            answer_rows.append(base)
            usage_rows.append({
                "question_id": frozen.question_id,
                "benchmark": base.get("benchmark"),
                "stratum": base.get("stratum"),
                "answer_model": answer.answer_model,
                "prompt_payload_hash": answer.prompt_payload_hash,
                "packing_prompt_tokens": answer.prompt_tokens,
                "api_prompt_tokens": answer.api_prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "total_tokens": answer.api_total_tokens,
                "latency_ms": answer.latency_ms,
                "cached": answer.cached,
                "finish_reason": answer.finish_reason,
            })
        append_jsonl(answer_path, answer_rows)
        append_jsonl(usage_path, usage_rows)
        print(f"checkpointed {min(start + len(batch), len(pending))}/{len(pending)}",
              flush=True)

    answers = read_jsonl(answer_path)
    usage = read_jsonl(usage_path)
    answer_ids = [str(row["question_id"]) for row in answers]
    usage_ids = [str(row["question_id"]) for row in usage]
    if set(answer_ids) != set(prepared_ids) or set(usage_ids) != set(prepared_ids):
        raise RuntimeError(
            "prepared replay incomplete: "
            f"prepared={len(prepared_ids)} answers={len(set(answer_ids))} "
            f"usage={len(set(usage_ids))}")
    if len(answer_ids) != len(prepared_ids) or len(usage_ids) != len(prepared_ids):
        raise RuntimeError("prepared replay artifacts contain duplicate question IDs")
    answer_hashes = {str(row["question_id"]): str(
        row.get("prompt_payload_hash") or "") for row in answers}
    prompt_hash_mismatches = [
        row.question_id for row in prepared
        if answer_hashes.get(row.question_id) != row.prompt_payload_hash
    ]
    if prompt_hash_mismatches:
        raise RuntimeError(
            "replayed answer prompt hashes diverged for "
            f"{len(prompt_hash_mismatches)} questions")
    for benchmark, filename in (("longmemeval", "answers_longmemeval.jsonl"),
                                ("locomo", "answers_locomo.jsonl")):
        selected = [row for row in answers if row.get("benchmark") == benchmark]
        (args.output_root / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected),
            encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-prepared-answer-replay-v1",
        "answer_model": args.answer_model,
        "config_hash": config_hash(config),
        "answer_request_profile": args.answer_request_profile,
        "max_output_tokens": None,
        "prepared": str(args.prepared),
        "prepared_sha256": hashlib.sha256(
            args.prepared.read_bytes()).hexdigest(),
        "prepared_questions": len(prepared),
        "completed_questions": len(answers),
        "prompt_hashes": len({row.prompt_payload_hash for row in prepared}),
        "prompt_identity_audit": {
            "question_ids_match": set(answer_ids) == set(prepared_ids),
            "evidence_and_order_frozen_in_prepared_artifact": True,
            "prompt_hash_mismatches": len(prompt_hash_mismatches),
        },
        "api_tokens": {
            "prompt": stats(row.get("api_prompt_tokens", 0) for row in usage),
            "completion": stats(row.get("completion_tokens", 0) for row in usage),
            "total": stats(row.get("total_tokens", 0) for row in usage),
        },
        "api_tokens_by_benchmark": {
            benchmark: {
                "prompt": stats(row.get("api_prompt_tokens", 0) for row in usage
                                if row.get("benchmark") == benchmark),
                "completion": stats(row.get("completion_tokens", 0) for row in usage
                                    if row.get("benchmark") == benchmark),
                "total": stats(row.get("total_tokens", 0) for row in usage
                               if row.get("benchmark") == benchmark),
            } for benchmark in ("longmemeval", "locomo")
        },
        "output_truncated": sum(row.get("finish_reason") == "length" for row in usage),
        "api_usage_sums": {
            "prompt": sum(int(row.get("api_prompt_tokens") or 0) for row in usage),
            "completion": sum(int(row.get("completion_tokens") or 0) for row in usage),
            "total": sum(int(row.get("total_tokens") or 0) for row in usage),
        },
        "api_usage_additivity_ok": all(
            int(row.get("api_prompt_tokens") or 0)
            + int(row.get("completion_tokens") or 0)
            == int(row.get("total_tokens") or 0)
            for row in usage),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    cache.close(); store.close()


if __name__ == "__main__":
    main()
