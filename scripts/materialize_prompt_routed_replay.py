#!/usr/bin/env python3
"""Assemble a replay from prior requests with byte-identical prompt hashes.

This is intentionally stricter than answer-text reuse: for every question the
selected PreparedAnswer, answer row, and API-usage row must all carry the same
prompt payload hash as the routed target.  Optional paired-judge artifacts are
selected from that same source and checked against the chosen prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
                    encoding="utf-8")


def _index(path: Path, field: str = "question_id") -> dict[str, dict]:
    rows = _read(path)
    result = {str(row[field]): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"duplicate {field} in {path}")
    return result


def _nearest(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _stats(values: list[int]) -> dict:
    return {
        "count": len(values), "mean": sum(values) / len(values),
        "p50": _nearest(values, .50), "p95": _nearest(values, .95),
        "p99": _nearest(values, .99), "max": max(values),
        "unit": "tokens_per_question", "percentile_method": "nearest-rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--source-name", action="append", required=True)
    parser.add_argument("--source-prepared", type=Path, action="append", required=True)
    parser.add_argument("--source-answer-dir", type=Path, action="append", required=True)
    parser.add_argument("--source-judge-lme", type=Path, action="append")
    parser.add_argument("--source-judge-locomo", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=2040)
    args = parser.parse_args()
    count = len(args.source_name)
    if not (count == len(args.source_prepared) == len(args.source_answer_dir)):
        raise ValueError("source name/prepared/answer-dir counts differ")
    if args.source_judge_lme and len(args.source_judge_lme) != count:
        raise ValueError("source judge-lme count differs")
    if args.source_judge_locomo and len(args.source_judge_locomo) != count:
        raise ValueError("source judge-locomo count differs")
    if args.output.exists():
        raise FileExistsError(args.output)

    target_rows = _read(args.prepared)
    if len(target_rows) != args.expected:
        raise ValueError(f"expected {args.expected} target rows, got {len(target_rows)}")
    target_ids = [str(row["question_id"]) for row in target_rows]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("duplicate question ID in routed target")

    sources = []
    for index, name in enumerate(args.source_name):
        answer_dir = args.source_answer_dir[index]
        sources.append({
            "name": name,
            "prepared": _index(args.source_prepared[index]),
            "answers": _index(answer_dir / "answers.jsonl"),
            "usage": _index(answer_dir / "answer_usage.jsonl"),
            "judge_lme": (_index(args.source_judge_lme[index])
                          if args.source_judge_lme else {}),
            "judge_locomo": (_index(args.source_judge_locomo[index])
                             if args.source_judge_locomo else {}),
        })

    selected_answers: list[dict] = []
    selected_usage: list[dict] = []
    selected_judges = {"longmemeval": [], "locomo": []}
    source_counts: Counter[str] = Counter()
    route_rows: list[dict] = []
    for target in target_rows:
        question_id = str(target["question_id"])
        prompt_hash = str(target["prompt_payload_hash"])
        matches = [source for source in sources
                   if question_id in source["prepared"]
                   and str(source["prepared"][question_id].get(
                       "prompt_payload_hash") or "") == prompt_hash]
        if not matches:
            raise ValueError(f"no byte-identical source for {question_id}")
        source = matches[0]
        answer = dict(source["answers"][question_id])
        usage = dict(source["usage"][question_id])
        if (str(answer.get("prompt_payload_hash") or "") != prompt_hash
                or str(usage.get("prompt_payload_hash") or "") != prompt_hash):
            raise ValueError(f"answer/usage hash mismatch for {question_id}")
        usage["reused_from"] = source["name"]
        selected_answers.append(answer)
        selected_usage.append(usage)
        source_counts[source["name"]] += 1
        benchmark = str(answer.get("benchmark") or "")
        judge_key = "judge_lme" if benchmark == "longmemeval" else "judge_locomo"
        judge = source[judge_key].get(question_id)
        if source[judge_key] and judge is None:
            raise ValueError(f"judge missing {question_id} in {source['name']}")
        if judge is not None:
            prediction_sha = hashlib.sha256(
                str(answer.get("prediction") or "").encode()).hexdigest()
            if str(judge.get("prediction_sha256") or "") != prediction_sha:
                raise ValueError(f"judge/prediction mismatch for {question_id}")
            selected_judges[benchmark].append(dict(judge))
        route_rows.append({
            "question_id": question_id,
            "source": source["name"],
            "prompt_payload_hash": prompt_hash,
        })

    args.output.mkdir(parents=True)
    _write(args.output / "prepared_answers.jsonl", target_rows)
    _write(args.output / "answers.jsonl", selected_answers)
    _write(args.output / "answer_usage.jsonl", selected_usage)
    _write(args.output / "prompt_routes.jsonl", route_rows)
    for benchmark in ("longmemeval", "locomo"):
        benchmark_answers = [row for row in selected_answers
                             if row.get("benchmark") == benchmark]
        _write(args.output / f"answers_{benchmark}.jsonl", benchmark_answers)
        if selected_judges[benchmark]:
            _write(args.output / f"paired_judge_{benchmark}.jsonl",
                   selected_judges[benchmark])

    manifest = {
        "schema_version": "graphmem-prompt-routed-replay-v1",
        "questions": len(target_rows),
        "source_counts": dict(source_counts),
        "prepared": str(args.prepared),
        "prepared_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "prompt_hash_mismatches": 0,
        "answer_calls": 0,
        "judge_calls": 0,
        "api_tokens": {
            name: _stats([int(row[field]) for row in selected_usage])
            for name, field in (("prompt", "api_prompt_tokens"),
                                ("completion", "completion_tokens"),
                                ("total", "total_tokens"))
        },
        "accuracy": {
            benchmark: {
                "questions": len(selected_judges[benchmark]),
                "correct": sum(bool(row.get("correct"))
                               for row in selected_judges[benchmark]),
                "accuracy": (sum(bool(row.get("correct"))
                                 for row in selected_judges[benchmark])
                             / len(selected_judges[benchmark])
                             if selected_judges[benchmark] else None),
            } for benchmark in ("longmemeval", "locomo")
        },
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
