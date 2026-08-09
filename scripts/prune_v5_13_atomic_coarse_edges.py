#!/usr/bin/env python3
"""Clone a graph and remove generic coarse edges between two atomic nodes.

This is the exact post-build equivalent of V5.13's typed-restoration rule.  It
uses ``replace_graph`` per memory so graph versions/checksums remain valid; the
source snapshot is never modified.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import NodeType, RelationType  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


ATOMIC_TYPES = frozenset({
    NodeType.CANONICAL_FACT, NodeType.EVENT_FRAME, NodeType.EVENT_SKELETON,
    NodeType.STATE_HEAD, NodeType.STATE_VALUE,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("atomic_pair", "routing_regions"),
                        default="routing_regions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(
        f"file:{args.source.resolve()}?mode=ro", uri=True)
    target_connection = sqlite3.connect(args.output)
    source_connection.backup(target_connection)
    source_connection.close(); target_connection.close()

    store = SQLiteGraphStore(args.output)
    totals: Counter[str] = Counter()
    memory_ids = sorted(str(row["memory_id"]) for row in store._read(
        "SELECT memory_id FROM conversations"))
    per_memory = []
    for memory_id in memory_ids:
        nodes = tuple(store.nodes(memory_id))
        by_id = {node.node_id: node for node in nodes}
        edges = tuple(store.edges(memory_id))
        def remove(edge) -> bool:
            if edge.relation != RelationType.COARSE_RELATED:
                return False
            left = by_id.get(edge.src_id); right = by_id.get(edge.dst_id)
            if left is None or right is None:
                return False
            if args.policy == "atomic_pair":
                return (left.node_type in ATOMIC_TYPES
                        and right.node_type in ATOMIC_TYPES)
            return not (left.node_type in {
                NodeType.ROUTING_CARD, NodeType.SCENE}
                and right.node_type in {
                    NodeType.ROUTING_CARD, NodeType.SCENE})

        kept = tuple(edge for edge in edges if not remove(edge))
        removed = len(edges) - len(kept)
        if removed:
            store.replace_graph(
                memory_id, nodes, kept, tuple(store.evidence_groups(memory_id)))
        totals.update({"memories": 1, "edges_before": len(edges),
                       "edges_after": len(kept), "removed": removed})
        per_memory.append({"memory_id": memory_id, "removed": removed})
    manifest = {
        "schema_version": "graphmem-v5.13-atomic-coarse-prune-v1",
        "source": str(args.source), "output": str(args.output),
        "policy": args.policy,
        "atomic_types": sorted(map(str, ATOMIC_TYPES)),
        "totals": dict(totals), "per_memory": per_memory,
    }
    manifest_path = args.output.parent / (
        args.output.stem + "_atomic_coarse_prune_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output), "totals": dict(totals)},
                     ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
