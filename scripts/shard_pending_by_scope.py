#!/usr/bin/env python3
"""Split only unfinished benchmark questions while keeping each shard in one scope."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def completed_question_ids(root: Path, variant: str) -> set[str]:
    completed: set[str] = set()
    for path in root.glob(f"shard_*/{variant}/question_stats.jsonl"):
        completed.update(str(row["question_id"]) for row in _rows(path))
    return completed


def shard_pending(
    rows: list[dict[str, Any]],
    completed: set[str],
    *,
    scope_field: str,
    shards_per_scope: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    pending = [row for row in rows if str(row["question_id"]) not in completed]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pending:
        scope = str(row.get(scope_field) or "")
        if not scope:
            raise ValueError(f"question {row['question_id']} has no {scope_field}")
        groups[scope].append(row)
    shards: list[list[dict[str, Any]]] = []
    assignments: dict[str, list[int]] = {}
    for scope, group in sorted(groups.items()):
        bins = [[] for _ in range(min(shards_per_scope, len(group)))]
        for position, row in enumerate(sorted(group, key=lambda item: item["question_id"])):
            bins[position % len(bins)].append(row)
        assignments[scope] = list(range(len(shards), len(shards) + len(bins)))
        shards.extend(bins)
    manifest = {
        "input_questions": len(rows),
        "completed_questions": len(completed.intersection(
            str(row["question_id"]) for row in rows
        )),
        "pending_questions": len(pending),
        "scope_count": len(groups),
        "shard_count": len(shards),
        "loads": [len(shard) for shard in shards],
        "scope_assignments": assignments,
    }
    return shards, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--completed-root", type=Path, required=True)
    parser.add_argument(
        "--additional-completed-root", type=Path, action="append", default=[],
        help="Additional completed shard root; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="hierarchical_hybrid_graph_v4_1_query")
    parser.add_argument("--scope-field", default="locomo_sample_id")
    parser.add_argument("--shards-per-scope", type=int, default=4)
    args = parser.parse_args()
    if args.shards_per_scope < 1:
        raise SystemExit("--shards-per-scope must be positive")
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    completed = set()
    for root in [args.completed_root, *args.additional_completed_root]:
        completed.update(completed_question_ids(root, args.variant))
    shards, manifest = shard_pending(
        rows, completed,
        scope_field=args.scope_field,
        shards_per_scope=args.shards_per_scope,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        (args.output_dir / f"shard_{index:02d}.json").write_text(
            json.dumps(shard, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
