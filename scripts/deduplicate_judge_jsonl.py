#!/usr/bin/env python3
"""Repair duplicate resumable-judge rows while preserving first completion.

This is only needed when an interrupted parent shell leaves a remote-judge
worker alive long enough to overlap a resumed worker.  Prediction-hash
disagreement is a hard failure; repeated temperature-zero verdicts that flip
are recorded, and the first physically appended verdict is retained.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def signature(row: dict[str, Any]) -> tuple[str, bool, str]:
    return (str(row.get("verdict")), bool(row.get("correct")),
            str(row.get("prediction_sha256") or ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    raw_lines = [line for line in args.input.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    rows = [json.loads(line) for line in raw_lines]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        question_id = str(row["question_id"])
        if question_id not in grouped:
            order.append(question_id)
        grouped[question_id].append(row)
    duplicate_ids = {key: value for key, value in grouped.items()
                     if len(value) > 1}
    hash_conflicts = {
        key: sorted({str(row.get("prediction_sha256") or "") for row in values})
        for key, values in duplicate_ids.items()
        if len({str(row.get("prediction_sha256") or "") for row in values}) > 1}
    if hash_conflicts:
        raise RuntimeError(f"duplicate rows refer to different predictions: {hash_conflicts}")
    verdict_flips = {
        key: [list(signature(row)) for row in values]
        for key, values in duplicate_ids.items()
        if len({signature(row)[:2] for row in values}) > 1}
    backup = args.input.with_name(args.input.name + ".pre_dedup")
    if duplicate_ids:
        if not backup.exists():
            shutil.copy2(args.input, backup)
        output = "".join(json.dumps(grouped[key][0], ensure_ascii=True) + "\n"
                         for key in order)
        temporary = args.input.with_name(args.input.name + ".tmp")
        temporary.write_text(output, encoding="utf-8")
        temporary.replace(args.input)
    audit = {
        "schema_version": "graphmem-judge-jsonl-dedup-v1",
        "input": str(args.input), "policy": "preserve_first_physical_completion",
        "rows_before": len(rows), "rows_after": len(grouped),
        "duplicate_rows_removed": len(rows) - len(grouped),
        "duplicate_question_ids": len(duplicate_ids),
        "verdict_flip_question_ids": verdict_flips,
        "prediction_hash_conflicts": hash_conflicts,
        "backup": str(backup) if duplicate_ids else None,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
