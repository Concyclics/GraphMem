#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedily shard GraphMem LoCoMo cases without splitting conversations."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    rows = json.loads(args.data.read_text(encoding="utf-8"))
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conversations[str(row["locomo_sample_id"])].append(row)
    bins: list[list[dict[str, Any]]] = [[] for _ in range(args.shards)]
    loads = [0] * args.shards
    assignments: dict[str, int] = {}
    for conversation_id, group in sorted(
        conversations.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        shard = min(range(args.shards), key=lambda index: (loads[index], index))
        bins[shard].extend(group)
        loads[shard] += len(group)
        assignments[conversation_id] = shard

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, shard_rows in enumerate(bins):
        shard_rows.sort(key=lambda row: row["question_id"])
        (args.output_dir / f"shard_{index}.json").write_text(
            json.dumps(shard_rows, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    manifest = {
        "shard_count": args.shards,
        "question_count": len(rows),
        "loads": loads,
        "conversation_assignments": assignments,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
