#!/usr/bin/env python3
"""Audit the full-corpus object/value relation construction experiment."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build.coarsen import promotable_object_value  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def _metadata(source: str) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    signals: tuple[str, ...] = ()
    witnesses: dict[str, tuple[str, ...]] = {}
    for component in source.split("|"):
        if component.startswith("relation_mask:"):
            signals = tuple(filter(None, component.split(":", 1)[1].split(",")))
        elif component.startswith("relation_witness:"):
            try:
                row = json.loads(component.split(":", 1)[1])
            except (TypeError, ValueError):
                continue
            if isinstance(row, dict):
                witnesses = {
                    str(key): tuple(map(str, values))
                    for key, values in row.items() if isinstance(values, list)}
    return signals, witnesses


def _snapshot(store: SQLiteGraphStore) -> dict:
    totals: Counter[str] = Counter()
    signal_edges: Counter[str] = Counter()
    object_witness_edges: Counter[str] = Counter()
    relation_edges: Counter[str] = Counter()
    per_memory: list[dict] = []
    for memory_id in store.memory_ids():
        promoted: set[str] = set()
        role_objects: set[str] = set()
        nodes = store.nodes(memory_id)
        for node in nodes:
            if node.node_type != NodeType.CANONICAL_FACT:
                continue
            totals["canonical_facts"] += 1
            key = promotable_object_value(node.attributes.get(
                "value_key", node.attributes.get("value", "")))
            if key:
                promoted.add(key)
                totals["promotable_fact_values"] += 1
            roles = node.attributes.get("relation_entity_roles", {})
            if isinstance(roles, dict):
                values = roles.get("object", ())
                if isinstance(values, (str, bytes)):
                    values = (values,)
                role_objects.update(map(str, values or ()))
        local: Counter[str] = Counter()
        endpoint_relations: set[tuple[str, str, str]] = set()
        for edge in store.edges(memory_id):
            relation_edges[str(edge.relation)] += 1
            key = tuple(sorted((edge.src_id, edge.dst_id))) + (str(edge.relation),)
            if key in endpoint_relations:
                local["duplicate_endpoint_relations"] += 1
            endpoint_relations.add(key)
            signals, witnesses = _metadata(edge.source)
            if signals:
                local["masked_edges"] += 1
            if len(signals) > 1:
                local["multi_attribute_edges"] += 1
            for signal in signals:
                signal_edges[signal] += 1
            shared = set(witnesses.get("shared_entity", ()))
            if shared & promoted:
                local["promoted_object_witness_edges"] += 1
                for value in shared & promoted:
                    object_witness_edges[value] += 1
        totals.update(local)
        totals["promoted_object_keys"] += len(promoted)
        totals["role_object_keys"] += len(role_objects)
        version, checksum = store.graph_identity(memory_id)
        per_memory.append({
            "memory_id": memory_id,
            "graph_version": version,
            "graph_checksum": checksum,
            "promoted_object_keys": len(promoted),
            "role_object_keys": len(role_objects),
            **dict(local),
        })
    return {
        "memories": len(per_memory),
        "totals": dict(totals),
        "relation_edges": dict(sorted(relation_edges.items())),
        "relation_signal_edges": dict(sorted(signal_edges.items())),
        "top_promoted_object_witnesses": object_witness_edges.most_common(100),
        "rows": per_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-db", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = SQLiteGraphStore(args.baseline_db, read_only=True)
    candidate = SQLiteGraphStore(args.candidate_db, read_only=True)
    try:
        before = _snapshot(baseline)
        after = _snapshot(candidate)
        before_checksums = {row["memory_id"]: row["graph_checksum"]
                            for row in before["rows"]}
        changed = sum(
            row["graph_checksum"] != before_checksums.get(row["memory_id"])
            for row in after["rows"])
        payload = {
            "schema_version": "graphmem-v5.32-object-relation-audit-v1",
            "baseline_db": str(args.baseline_db),
            "candidate_db": str(args.candidate_db),
            "graph_checksums_changed": changed,
            "baseline": before,
            "candidate": after,
        }
    finally:
        baseline.close()
        candidate.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "graph_checksums_changed": changed,
        "baseline": {k: v for k, v in before.items() if k != "rows"},
        "candidate": {k: v for k, v in after.items() if k != "rows"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
