#!/usr/bin/env python3
"""Merge disjoint recoarsening shards into one canonical SQLite graph."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def db_path(path: Path) -> Path:
    # A not-yet-created artifact directory does not satisfy ``is_dir``.  Treat
    # only an explicit .sqlite suffix as a file; otherwise create the standard
    # report directory layout.
    return path if path.suffix == ".sqlite" else path / "report_graph.sqlite"


def main() -> None:
    args = parse_args()
    target_path = db_path(args.output)
    if target_path.exists():
        raise FileExistsError(f"output already exists: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target = SQLiteGraphStore(target_path)
    seen: set[str] = set()
    rows = []
    totals: Counter[str] = Counter()
    for source_arg in args.inputs:
        source_path = db_path(source_arg)
        source = SQLiteGraphStore(source_path, read_only=True)
        memory_ids = sorted(str(row["memory_id"]) for row in source._read(
            "SELECT memory_id FROM conversations"))
        overlap = seen & set(memory_ids)
        if overlap:
            raise ValueError(f"shards overlap on memories: {sorted(overlap)[:5]}")
        for memory_id in memory_ids:
            conversation = source.conversation(memory_id)
            if conversation is None:
                raise RuntimeError(f"missing conversation {memory_id} in {source_path}")
            target.ingest_conversation(
                conversation, source.sessions(memory_id), source.turns(memory_id))
            nodes = tuple(source.nodes(memory_id))
            edges = tuple(source.edges(memory_id))
            groups = tuple(source.evidence_groups(memory_id))
            target.replace_graph(memory_id, nodes, edges, groups)
            totals.update({"memories": 1, "nodes": len(nodes),
                           "edges": len(edges), "evidence_groups": len(groups)})
        seen.update(memory_ids)
        manifest_path = source_path.parent / "recoarsen_manifest.json"
        rows.append({
            "input": str(source_path),
            "memories": len(memory_ids),
            "manifest": (json.loads(manifest_path.read_text(encoding="utf-8"))
                         if manifest_path.exists() else None),
        })
        source.close()
    manifest = {
        "schema_version": "graphmem-v5.13-shard-merge-v1",
        "output": str(target_path),
        "disjoint_memory_shards": True,
        "totals": dict(totals),
        "inputs": rows,
    }
    manifest_path = target_path.parent / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(target_path), "totals": dict(totals)},
                     ensure_ascii=False, indent=2))
    target.close()


if __name__ == "__main__":
    main()
