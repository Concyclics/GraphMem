#!/usr/bin/env python3
"""Normalize the archived Mem0 Qwen3-30B top-50/top-200 baseline.

Answer usage is read per cutoff from question files.  LoCoMo build usage is
read only from the ten independent ingestion owners because its aggregate
JSON repeats a conversation build ledger for every question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


MODEL = "Qwen3-30B-A3B-Instruct-2507"
CUTOFFS = (50, 200)
TOKEN_FIELDS = {
    "input": "{stage}_cache_miss_input_tokens",
    "cached_input": "{stage}_cache_hit_input_tokens",
    "output": "{stage}_output_tokens",
    "total": "{stage}_total_tokens",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: Iterable[int | float], percentile: float) -> int | float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def summarize(values: Iterable[int | float], *, unit: str) -> dict:
    sample = list(values)
    if not sample:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(sample),
        "mean": sum(sample) / len(sample),
        "p95": nearest_rank(sample, 0.95),
        "p99": nearest_rank(sample, 0.99),
        "max": max(sample),
        "sum": sum(sample),
        "unit": unit,
        "percentile_method": "nearest_rank",
    }


def exact_mcnemar(losses: int, gains: int) -> float:
    discordant = losses + gains
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(losses, gains) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def paired_bootstrap(diffs: list[int], *, samples: int = 10_000) -> list[float]:
    rng = random.Random(0)
    draws = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs)
                   for _ in range(samples))
    return [nearest_rank(draws, 0.025), nearest_rank(draws, 0.975)]


def token_breakdown(records: list[dict], stage: str, *, unit: str) -> dict:
    result = {}
    for label, template in TOKEN_FIELDS.items():
        field = template.format(stage=stage)
        result[label] = summarize((record[field] for record in records), unit=unit)
    return result


def answer_records(questions: list[dict], cutoff: int) -> list[dict]:
    key = f"top_{cutoff}"
    records = []
    for question in questions:
        usage = question.get("token_stats", {}).get("answer_by_cutoff", {}).get(key)
        if usage is None:
            raise ValueError(f"missing {key} usage for {question.get('question_id')}")
        records.append(usage)
    return records


def judge_paths(run: Path, benchmark: str, cutoff: int) -> tuple[Path, Path]:
    root = run / "judge/v2-luna"
    if benchmark == "locomo":
        root /= "locomo"
    judge = root / f"judge_top_{cutoff}"
    return judge / "auto_eval.jsonl", judge / "judge_token_stats.json"


def import_benchmark(run: Path, benchmark: str, prediction_dir: str,
                     expected_questions: int) -> tuple[list[dict], dict]:
    predicted = run / "results/raw" / prediction_dir
    question_files = sorted(path for path in predicted.glob("*.json")
                            if not path.name.startswith("_"))
    all_questions = [read(path) for path in question_files]
    expected_all = 500 if benchmark == "longmemeval" else 1986
    if len(all_questions) != expected_all:
        raise ValueError(f"unexpected {benchmark} archive question count")
    questions = [row for row in all_questions
                 if benchmark == "longmemeval" or row.get("category") in (1, 2, 3, 4)]
    question_ids = {str(row["question_id"]) for row in questions}
    if len(questions) != expected_questions or len(question_ids) != expected_questions:
        raise ValueError(f"{benchmark} selected IDs are incomplete or duplicated")

    ingestion_files = sorted(predicted.glob("_ingestion_*.json"))
    expected_ingestion = 500 if benchmark == "longmemeval" else 10
    if len(ingestion_files) != expected_ingestion:
        raise ValueError(f"expected {expected_ingestion} ingestion owners for {benchmark}")
    ingestion = [read(path) for path in ingestion_files]
    build_records = [row["token_stats"] for row in ingestion]
    build_unit = ("tokens_per_memory" if benchmark == "longmemeval"
                  else "tokens_per_conversation")
    build_breakdown = token_breakdown(build_records, "build", unit=build_unit)
    failures = [{"owner": row.get("question_id"),
                 "failed_pairs": int(row.get("total_pairs_failed") or 0),
                 "processed_pairs": int(row.get("total_pairs_processed") or 0)}
                for row in ingestion if row.get("total_pairs_failed")]

    rows = []
    verdict_by_cutoff: dict[int, dict[str, bool]] = {}
    for cutoff in CUTOFFS:
        usage = answer_records(questions, cutoff)
        answer_breakdown = token_breakdown(
            usage, "answer", unit="tokens_per_question")
        verdict_path, judge_stats_path = judge_paths(run, benchmark, cutoff)
        verdicts = read_jsonl(verdict_path)
        if benchmark == "locomo":
            verdicts = [row for row in verdicts if int(row["category"]) in (1, 2, 3, 4)]
        verdict_map = {str(row["question_id"]): bool(row["correct"])
                       for row in verdicts}
        if len(verdict_map) != expected_questions or set(verdict_map) != question_ids:
            raise ValueError(f"{benchmark} top-{cutoff} verdict contract failed")
        verdict_by_cutoff[cutoff] = verdict_map
        correct = sum(verdict_map.values())
        judge_stats = read(judge_stats_path)
        if judge_stats.get("model") != "gpt-5.6-luna":
            raise ValueError(f"unexpected judge model in {judge_stats_path}")
        latencies = summarize(
            (record["answer_latency_sec"] for record in usage),
            unit="seconds_per_question")
        retrieval = summarize(
            (row["token_stats"]["retrieval_latency_sec"] for row in questions),
            unit="seconds_per_question")
        completion_cap_hits = sum(
            record["answer_output_tokens"] == 4096 for record in usage)
        rows.append({
            "method": "Mem0",
            "answer_model": MODEL,
            "model_snapshot": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            "runtime_precision": "bfloat16",
            "runtime_quantization": None,
            "benchmark": benchmark,
            "retrieval_setting": f"top-{cutoff}",
            "cutoff": cutoff,
            "status": "complete",
            "comparison_scope": "archived_baseline_as_run",
            "questions": expected_questions,
            "correct": correct,
            "accuracy": correct / expected_questions,
            "build_tokens": build_breakdown["total"],
            "build_api_tokens": build_breakdown,
            "build_unit": build_unit,
            "build_quality": {
                "owners": len(ingestion),
                "final_failed_pairs": sum(row["failed_pairs"] for row in failures),
                "owners_with_failed_pairs": len(failures),
                "failures": failures,
            },
            "answer_tokens": answer_breakdown["total"],
            "answer_api_tokens": answer_breakdown,
            "answer_latency_seconds": latencies,
            "retrieval_latency_seconds": retrieval,
            "effective_completion_cap": 4096 if completion_cap_hits else None,
            "completion_cap_hits": completion_cap_hits,
            "judge_model": judge_stats["model"],
            "judge_thinking": judge_stats.get("thinking"),
            "judge_reasoning_effort": judge_stats.get("reasoning_effort"),
            "judge_prompt_commit": (judge_stats.get("prompt_commit")
                                     or judge_stats.get("judge_prompt_commit")),
            "judge_prompt_sha256": (judge_stats.get("prompt_source_sha256")
                                     or judge_stats.get("judge_prompt_sha256")),
            "artifacts": str(run),
            "verdict_artifact": str(verdict_path),
            "config_artifact": str(next((run / "config").glob("local-*.yaml"))),
        })

    top50 = verdict_by_cutoff[50]
    top200 = verdict_by_cutoff[200]
    ids = sorted(question_ids)
    losses = sum(top50[qid] and not top200[qid] for qid in ids)
    gains = sum(not top50[qid] and top200[qid] for qid in ids)
    diffs = [int(top200[qid]) - int(top50[qid]) for qid in ids]
    comparison = {
        "questions": len(ids),
        "both_correct": sum(top50[qid] and top200[qid] for qid in ids),
        "top50_only": losses,
        "top200_only": gains,
        "both_wrong": sum(not top50[qid] and not top200[qid] for qid in ids),
        "accuracy_delta": sum(diffs) / len(diffs),
        "mcnemar_exact_p": exact_mcnemar(losses, gains),
        "paired_bootstrap_95ci": paired_bootstrap(diffs),
        "answer_total_mean_ratio": (
            rows[1]["answer_tokens"]["mean"] / rows[0]["answer_tokens"]["mean"]),
    }
    audit = {
        "questions_all": len(all_questions),
        "questions_selected": len(questions),
        "unique_question_ids": len(question_ids),
        "ingestion_owners": len(ingestion),
        "build_total_sum": build_breakdown["total"]["sum"],
        "final_failed_pairs": sum(row["failed_pairs"] for row in failures),
        "top200_vs_top50": comparison,
    }
    return rows, audit


def pending_gpt_rows() -> list[dict]:
    return [{
        "method": "Mem0", "answer_model": "gpt-5.4-mini",
        "benchmark": benchmark, "retrieval_setting": f"top-{cutoff}",
        "cutoff": cutoff, "status": "pending", "questions": None,
        "accuracy": None, "build_tokens": None, "answer_tokens": None,
        "artifacts": None,
    } for benchmark in ("longmemeval", "locomo") for cutoff in CUTOFFS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path("/shared/s3/GraphMem_eval"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_root = args.archive / "mem0/qwen3-30b-a3b-instruct-2507"
    lme_run = model_root / "longmemeval/run_20260806_full500_w96"
    locomo_run = model_root / "locomo/run_20260809_reliable"
    runtime_log = locomo_run / "logs/llm.log"
    runtime_text = runtime_log.read_text(encoding="utf-8", errors="replace")
    for expected in ("dtype=torch.bfloat16", "quantization=None",
                     "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"):
        if expected not in runtime_text:
            raise ValueError(f"runtime contract missing {expected}")

    lme_rows, lme_audit = import_benchmark(
        lme_run, "longmemeval", "predicted_full500_w96_30b", 500)
    locomo_rows, locomo_audit = import_benchmark(
        locomo_run, "locomo", "predicted_locomo_w32_30b_reliable", 1540)
    payload = {
        "schema_version": "graphmem-v5.19-mem0-cutoff-baseline-v1",
        "archive": str(args.archive),
        "method": "Mem0",
        "model": MODEL,
        "percentile_method": "nearest_rank",
        "judge_model": "gpt-5.6-luna",
        "judge_tokens_excluded": True,
        "locomo_categories": [1, 2, 3, 4],
        "cutoffs": list(CUTOFFS),
        "rows": [*lme_rows, *locomo_rows, *pending_gpt_rows()],
        "audit": {"longmemeval": lme_audit, "locomo": locomo_audit},
        "source_checksums": {
            "longmemeval_config": sha256(next((lme_run / "config").glob("local-*.yaml"))),
            "locomo_config": sha256(next((locomo_run / "config").glob("local-*.yaml"))),
        },
        "warnings": [
            "The archived Qwen3-30B baseline is BF16 without quantization; the current GraphMem answer service is FP8.",
            "LongMemEval has ten owners with one final failed ingestion pair each.",
            "LoCoMo answer completions hit an effective 4096-token cap; counts are recorded per cutoff.",
            "LoCoMo build tokens are deduplicated by ten conversation ingestion owners rather than summed from the per-question aggregate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "complete_rows": sum(row["status"] == "complete" for row in payload["rows"]),
        "pending_rows": sum(row["status"] == "pending" for row in payload["rows"]),
        "audit": payload["audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
