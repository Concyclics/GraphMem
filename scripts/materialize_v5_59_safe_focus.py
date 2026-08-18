#!/usr/bin/env python3
"""Materialize the validated safe Query-Focus route over frozen dual-lane packs.

The broad-focus and baseline inputs must come from identical navigation
configuration.  A separate audited route artifact supplies question IDs for
which the final QueryIR gate enabled Query Focus.  No answer, gold annotation,
or judge verdict is read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def index(rows: list[dict]) -> dict[str, dict]:
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate question IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--broad-focus", type=Path, required=True)
    parser.add_argument("--safe-route", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=2040)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    baseline_rows = read(args.baseline)
    broad_rows = index(read(args.broad_focus))
    route_rows = index(read(args.safe_route))
    if len(baseline_rows) != args.expected:
        raise ValueError(f"expected {args.expected} rows, got {len(baseline_rows)}")
    if set(broad_rows) != {str(row["question_id"]) for row in baseline_rows}:
        raise ValueError("broad-focus IDs do not match baseline")
    if set(route_rows) != set(broad_rows):
        raise ValueError("safe-route IDs do not match baseline")

    selected: list[dict] = []
    focus_count = 0
    for baseline in baseline_rows:
        question_id = str(baseline["question_id"])
        use_focus = bool(route_rows[question_id].get("trace", {}).get(
            "query_focus_index"))
        row = broad_rows[question_id] if use_focus else baseline
        if use_focus and not row.get("trace", {}).get("query_focus_index"):
            raise ValueError(f"broad source lacks focus for {question_id}")
        if (row.get("memory_id") != baseline.get("memory_id")
                or (not use_focus and row.get("prompt_payload_hash")
                    != baseline.get("prompt_payload_hash"))):
            raise ValueError(f"source mismatch for {question_id}")
        selected.append(row)
        focus_count += int(use_focus)

    args.output.mkdir(parents=True)
    output = args.output / "prepared_answers.jsonl"
    output.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in selected),
        encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.59-safe-focus-prepare-v1",
        "questions": len(selected),
        "query_focus_questions": focus_count,
        "baseline_questions": len(selected) - focus_count,
        "routing_inputs": ["QueryIR operator", "query surface", "source shape"],
        "uses_answers_or_judges": False,
        "prepared_answers": str(output),
        "prepared_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (args.output / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
