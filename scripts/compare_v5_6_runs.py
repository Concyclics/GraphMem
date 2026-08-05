#!/usr/bin/env python3
"""Compare two retrieval runs on execution-relevant fields only.

A shadow change must leave execution untouched.  Traces legitimately gain new
diagnostic keys, so comparing whole records would always report a difference;
this compares what the pipeline actually produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Everything that decides an answer.  Trace/telemetry keys are deliberately out.
EXECUTION_FIELDS = (
    "packed_turn_ids", "dropped_turn_ids", "retrieved_turn_ids", "retrieved_session_ids",
    "seed_node_ids", "visited_path_node_ids", "visited_nodes", "visited_edges",
    "evidence_tokens", "budget_exhausted", "stop_reason", "proof_units", "certificate",
    "candidate_scores", "operand_coverage", "slot_coverage", "proof",
)


def load(run: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for line in (run / "navigation_results.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            rows.setdefault(record["configuration"], []).append(record)
    return rows


def projection(record: dict) -> str:
    return json.dumps({key: record.get(key) for key in EXECUTION_FIELDS},
                      sort_keys=True, ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--profiles", default="h0,h6,h8")
    args = parser.parse_args()

    left, right = load(args.baseline), load(args.candidate)
    failures = 0
    for profile in (item.strip() for item in args.profiles.split(",") if item.strip()):
        if profile not in left or profile not in right:
            print(f"{profile}: MISSING (baseline={profile in left}, candidate={profile in right})")
            failures += 1
            continue
        rows_left, rows_right = left[profile], right[profile]
        if len(rows_left) != len(rows_right):
            print(f"{profile}: question count differs {len(rows_left)} vs {len(rows_right)}")
            failures += 1
            continue
        differing = [index for index, (one, two) in enumerate(zip(rows_left, rows_right))
                     if projection(one) != projection(two)]
        status = "IDENTICAL" if not differing else f"DIFFERS on {len(differing)}/{len(rows_left)}"
        print(f"{profile}: {status}")
        if differing:
            failures += 1
            for index in differing[:3]:
                one, two = rows_left[index], rows_right[index]
                changed = [key for key in EXECUTION_FIELDS if one.get(key) != two.get(key)]
                print(f"    q{index} ({one.get('dev_question_id', '?')}): {changed}")
    print("OK: execution unchanged" if not failures else "FAIL: execution changed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
