#!/usr/bin/env python3
"""Materialize only unresolved, previously unseen answers for prefix k."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    if args.expected and len(rows) != args.expected:
        raise RuntimeError(
            f"expected {args.expected} candidate rows, got {len(rows)}")
    known: dict[tuple[str, str], bool] = {}
    already_correct: set[str] = set()
    for index in range(1, args.candidate_index):
        path = args.judge_root / f"candidate_{index}" / "auto_eval.jsonl"
        for verdict in read_jsonl(path):
            question_id = str(verdict["question_id"])
            digest = str(verdict["prediction_sha256"])
            flag = bool(verdict["correct"])
            previous = known.setdefault((question_id, digest), flag)
            if previous != flag:
                raise RuntimeError(
                    f"judge conflict for {question_id} prediction {digest}")
            if flag:
                already_correct.add(question_id)

    output = []
    duplicate_known_wrong = 0
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in already_correct:
            continue
        choices = row["candidates"]
        if args.candidate_index < 1 or args.candidate_index > len(choices):
            raise ValueError("candidate index exceeds generated choices")
        candidate = choices[args.candidate_index - 1]
        digest = str(candidate["prediction_sha256"])
        if (question_id, digest) in known:
            if known[(question_id, digest)]:
                raise RuntimeError("correct duplicate should already be resolved")
            duplicate_known_wrong += 1
            continue
        output.append({
            "question_id": question_id,
            "prediction": candidate["prediction"],
            "benchmark": "locomo",
            "stratum": f"locomo_cat{int(row['category'])}",
            "candidate_index": args.candidate_index,
            "prediction_sha256": digest,
            "prompt_payload_hash": row.get("prompt_payload_hash"),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-locomo-best-of-judge-prefix-v1",
        "candidate_index": args.candidate_index,
        "source_candidates": str(args.candidates),
        "source_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "questions_total": len(rows),
        "already_correct": len(already_correct),
        "duplicate_known_wrong": duplicate_known_wrong,
        "new_judge_requests": len(output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
