#!/usr/bin/env python3
"""Create deterministic, mutually exclusive round-robin benchmark shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--group-field")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = json.loads(args.data.read_text(encoding="utf-8"))
    if args.shards <= 0:
        raise ValueError("shards must be positive")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(args.shards)]
    if args.group_field:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = str(row.get(args.group_field) or "")
            if not value:
                raise ValueError(f"missing {args.group_field}: {row.get('question_id')}")
            groups[value].append(row)
        loads = [0] * args.shards
        for _, group in sorted(groups.items()):
            target = min(range(args.shards), key=lambda index: (loads[index], index))
            buckets[target].extend(group)
            loads[target] += len(group)
    else:
        for position, row in enumerate(sorted(rows, key=lambda item: str(item["question_id"]))):
            buckets[position % args.shards].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for index, bucket in enumerate(buckets):
        bucket.sort(key=lambda item: str(item["question_id"]))
        ids = {str(row["question_id"]) for row in bucket}
        if seen & ids:
            raise ValueError("shards overlap")
        seen.update(ids)
        (args.output_dir / f"shard_{index:02d}.json").write_text(
            json.dumps(bucket, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
    if len(seen) != len(rows):
        raise ValueError("sharding lost or duplicated questions")
    print(json.dumps({"questions": len(rows), "shards": len(buckets),
                      "loads": [len(bucket) for bucket in buckets]}))


if __name__ == "__main__":
    main()
