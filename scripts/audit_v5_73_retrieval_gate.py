#!/usr/bin/env python3
"""Fail closed when a V5.73 prompt pack regresses paired gold coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


def _load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("dev_question_id") or row.get("question_id") or "")
            if not key:
                raise ValueError(f"missing question id in {path}")
            if key in rows:
                raise ValueError(f"duplicate question id {key!r} in {path}")
            rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    common = sorted(set(baseline) & set(candidate))
    if len(common) != args.expected:
        raise SystemExit(
            f"paired retrieval gate expected {args.expected} rows, got {len(common)}")

    annotated = [key for key in common if baseline[key].get("has_turn_gold")]
    if not annotated:
        raise SystemExit("paired retrieval gate found no turn-level annotations")

    def mean(rows: dict[str, dict], field: str) -> float:
        return fmean(float(rows[key].get(field, 0.0)) for key in annotated)

    fields = ("turn_any_hit", "turn_all_hit", "turn_recall", "turn_precision")
    metrics = {
        field: {
            "baseline": mean(baseline, field),
            "candidate": mean(candidate, field),
        }
        for field in fields
    }
    for values in metrics.values():
        values["delta"] = values["candidate"] - values["baseline"]

    all_hit_delta = metrics["turn_all_hit"]["delta"]
    recall_delta = metrics["turn_recall"]["delta"]
    passed = all_hit_delta >= -1e-12 and recall_delta >= -1e-12
    result = {
        "schema_version": "graphmem-v5.73-retrieval-gate-v1",
        "paired_questions": len(common),
        "annotated_questions": len(annotated),
        "metrics": metrics,
        "passed": passed,
        "requirements": {
            "turn_all_hit_delta_min": 0.0,
            "turn_recall_delta_min": 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
