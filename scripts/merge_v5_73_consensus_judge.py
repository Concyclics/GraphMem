#!/usr/bin/env python3
"""Merge unchanged baseline verdicts with rejudged consensus deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    selections = read_jsonl(args.selections)
    baseline = {str(row["question_id"]): row for row in read_jsonl(args.baseline)}
    delta = {str(row["question_id"]): row for row in read_jsonl(args.delta)}
    output = []
    delta_used = 0
    for selection in selections:
        question_id = str(selection["question_id"])
        digest = hashlib.sha256(
            str(selection["prediction"]).encode()).hexdigest()
        source = baseline.get(question_id)
        if source is None or str(source.get("prediction_sha256")) != digest:
            source = delta.get(question_id)
            delta_used += 1
        if source is None or str(source.get("prediction_sha256")) != digest:
            raise RuntimeError(f"no matching verdict for {question_id}")
        output.append(source)
    if args.expected and len(output) != args.expected:
        raise RuntimeError(f"expected {args.expected} verdicts, got {len(output)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8")
    print(json.dumps({
        "questions": len(output), "delta_verdicts_used": delta_used,
        "correct": sum(bool(row["correct"]) for row in output),
        "accuracy": sum(bool(row["correct"]) for row in output) / len(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
