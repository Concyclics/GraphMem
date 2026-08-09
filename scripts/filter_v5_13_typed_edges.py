#!/usr/bin/env python3
"""Clone a graph and retain only typed edges that pass the online contract."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build.coarsen import admit_llm_refined_relation  # noqa: E402
from graphmem.domain import RelationType  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


TYPED_RELATIONS = frozenset({
    RelationType.COREFERENCE,
    RelationType.TEMPORAL_CONTINUATION,
    RelationType.CAUSAL,
    RelationType.CONTRADICTION_UPDATE,
    RelationType.SAME_ENTITY_STATE,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.82)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min-confidence must be in [0,1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(
        f"file:{args.source.resolve()}?mode=ro", uri=True)
    target_connection = sqlite3.connect(args.output)
    source_connection.backup(target_connection)
    source_connection.close(); target_connection.close()

    store = SQLiteGraphStore(args.output)
    totals: Counter[str] = Counter()
    per_memory = []
    memory_ids = sorted(str(row["memory_id"]) for row in store._read(
        "SELECT memory_id FROM conversations"))
    for memory_id in memory_ids:
        nodes = tuple(store.nodes(memory_id))
        by_id = {node.node_id: node for node in nodes}
        edges = tuple(store.edges(memory_id))
        kept = []
        removed: Counter[str] = Counter()
        for edge in edges:
            if edge.relation not in TYPED_RELATIONS:
                kept.append(edge)
                continue
            left = by_id.get(edge.src_id); right = by_id.get(edge.dst_id)
            admitted = bool(
                left is not None and right is not None
                and admit_llm_refined_relation(
                    edge.relation, left, right, edge.confidence,
                    min_confidence=args.min_confidence))
            if admitted:
                kept.append(edge)
                totals[f"kept:{edge.relation}"] += 1
            else:
                removed[str(edge.relation)] += 1
                totals[f"removed:{edge.relation}"] += 1
        if removed:
            store.replace_graph(
                memory_id, nodes, tuple(kept),
                tuple(store.evidence_groups(memory_id)))
        totals.update({"memories": 1, "edges_before": len(edges),
                       "edges_after": len(kept)})
        per_memory.append({"memory_id": memory_id,
                           "removed": dict(sorted(removed.items()))})
    manifest = {
        "schema_version": "graphmem-v5.13-online-typed-filter-v1",
        "source": str(args.source), "output": str(args.output),
        "min_confidence": args.min_confidence,
        "typed_relations": sorted(map(str, TYPED_RELATIONS)),
        "totals": dict(sorted(totals.items())),
        "per_memory": per_memory,
    }
    manifest_path = args.output.parent / (
        args.output.stem + "_typed_filter_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output), "totals": dict(totals)},
                     ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
