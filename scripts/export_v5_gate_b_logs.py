#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from graphmem.domain import canonical_json, dataclass_dict
from graphmem.storage import SQLiteGraphStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store = SQLiteGraphStore(args.sqlite)
    with gzip.open(args.output / "llm_calls.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in store._connection.execute("SELECT * FROM llm_calls ORDER BY created_at,call_id"):
            handle.write(canonical_json(dict(row)) + "\n")
    if args.graph:
        with gzip.open(args.output / "graph_snapshot.jsonl.gz", "wt", encoding="utf-8") as handle:
            memories = [row[0] for row in store._connection.execute(
                "SELECT memory_id FROM conversations ORDER BY memory_id"
            )]
            for memory_id in memories:
                for kind, rows in (
                    ("node", store.nodes(memory_id)), ("edge", store.edges(memory_id)),
                    ("evidence_group", store.evidence_groups(memory_id)),
                ):
                    for row in rows:
                        handle.write(canonical_json({"record_type": kind, **dataclass_dict(row)}) + "\n")
    store.close()


if __name__ == "__main__":
    main()
