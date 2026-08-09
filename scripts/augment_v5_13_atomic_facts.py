#!/usr/bin/env python3
"""Clone a frozen graph and materialize V5.10 atomic rescue facts into it.

The extracted packets are query-agnostic and quote-grounded.  The source DB is
opened read-only and copied with SQLite backup before any graph mutation.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build.pipeline import GraphBuildPipeline  # noqa: E402
from graphmem.build.semantic import ScenePacket, SemanticFact  # noqa: E402
from graphmem.config import load_config  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=WORKSPACE /
                        "artifacts/v5_9/diagnostic_ablations/dense_dev200/"
                        "report_graph_dense.sqlite")
    parser.add_argument("--extracted", type=Path, default=ROOT /
                        "artifacts/report/v5_10/atomic_gate_v3/extracted.json")
    parser.add_argument("--config", type=Path, default=ROOT /
                        "configs/v5/v5_10_report.json")
    parser.add_argument("--output-db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_13/atomic_augmented_dev200/"
                        "report_graph.sqlite")
    return parser.parse_args()


def clone_sqlite(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(
        f"file:{source.resolve()}?mode=ro", uri=True)
    target_connection = sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()


def main() -> None:
    args = parse_args()
    clone_sqlite(args.source_db, args.output_db)
    extracted = json.loads(args.extracted.read_text(encoding="utf-8"))
    packets_by_turn = {str(row["turn_id"]): row for row in extracted}
    config = load_config(args.config)
    store = SQLiteGraphStore(args.output_db)
    turn_locations = {
        turn.turn_id: turn.memory_id
        for row in store._read("SELECT DISTINCT memory_id FROM conversations")
        for turn in store.turns(str(row["memory_id"]))
        if turn.turn_id in packets_by_turn
    }
    selected_by_memory: dict[str, list[str]] = defaultdict(list)
    for turn_id, memory_id in turn_locations.items():
        selected_by_memory[memory_id].append(turn_id)
    builder = GraphBuildPipeline(store, dataset_hash="v5_13_atomic_augmentation")
    rows = []
    totals: Counter[str] = Counter()
    for index, memory_id in enumerate(sorted(selected_by_memory), 1):
        turns = tuple(store.turns(memory_id))
        turn_map = {turn.turn_id: turn for turn in turns}
        groups = tuple(store.evidence_groups(memory_id))
        group_by_turn = {
            member.turn_id: group
            for group in groups for member in group.members}
        old_nodes = tuple(store.nodes(memory_id))
        old_edges = tuple(store.edges(memory_id))
        scene_nodes = {
            node.node_id: node for node in old_nodes
            if node.node_type == NodeType.SCENE}
        scene_by_group = {
            group_id: scene_id
            for scene_id, node in scene_nodes.items()
            for group_id in node.all_evidence_group_ids}
        packets = []
        unresolved_turns = []
        fallback_facts = 0
        for turn_id in sorted(selected_by_memory[memory_id]):
            group = group_by_turn.get(turn_id)
            scene_id = (scene_by_group.get(group.evidence_group_id)
                        if group is not None else None)
            if scene_id is None:
                unresolved_turns.append(turn_id)
                continue
            source = packets_by_turn[turn_id]
            facts = []
            for row in source.get("facts", ()):
                facts.append(SemanticFact(
                    owner=str(row["owner"]),
                    predicate=str(row["predicate"]),
                    value=str(row["value"]),
                    value_type=str(row.get("value_type") or "text"),
                    scope=str(row.get("scope") or "general"),
                    polarity=str(row.get("polarity") or "positive"),
                    time=(str(row["time"]) if row.get("time") else None),
                    confidence=float(row.get("confidence", 0.0)),
                    evidence=tuple(
                        (str(evidence[0]), int(evidence[1]), int(evidence[2]))
                        for evidence in row.get("evidence", ())),
                    information_unit_ids=tuple(map(
                        int, row.get("information_unit_ids", ()))),
                ))
            fallback_facts += int(bool(source.get("raw_fallback_turn_ids")))
            packets.append(ScenePacket(
                scene_id=scene_id,
                summary=" ".join(turn_map[turn_id].raw_text.split()[:48]),
                facts=tuple(facts),
                unresolved=tuple(map(str, source.get("unresolved_unit_ids", ()))),
                fallback=bool(source.get("raw_fallback_turn_ids")),
            ))
        new_nodes, new_edges = builder._lean_semantic_graph(
            memory_id, packets, turn_map, group_by_turn, config, scene_nodes)
        nodes_by_id = {node.node_id: node for node in old_nodes}
        before_facts = sum(node.node_type == NodeType.CANONICAL_FACT
                           for node in old_nodes)
        nodes_by_id.update({node.node_id: node for node in new_nodes})
        edges_by_id = {edge.edge_id: edge for edge in old_edges}
        edges_by_id.update({edge.edge_id: edge for edge in new_edges})
        merged_nodes = builder._propagate_time_ranges(
            list(nodes_by_id.values()), list(edges_by_id.values()))
        store.replace_graph(memory_id, merged_nodes,
                            tuple(edges_by_id.values()), groups)
        after_facts = sum(node.node_type == NodeType.CANONICAL_FACT
                          for node in merged_nodes)
        row = {
            "memory_id": memory_id,
            "selected_turns": len(selected_by_memory[memory_id]),
            "resolved_turns": len(packets),
            "unresolved_turn_ids": unresolved_turns,
            "raw_fallback_packets": fallback_facts,
            "facts_before": before_facts,
            "facts_after": after_facts,
            "facts_added": after_facts - before_facts,
            "edges_added": len(edges_by_id) - len(old_edges),
        }
        rows.append(row)
        totals.update({key: int(row[key]) for key in (
            "selected_turns", "resolved_turns", "raw_fallback_packets",
            "facts_added", "edges_added")})
        print(f"{index}/{len(selected_by_memory)} {memory_id}: "
              f"turns={row['resolved_turns']} facts+={row['facts_added']}",
              flush=True)
    manifest = {
        "schema_version": "graphmem-v5.13-atomic-augmentation-v1",
        "source_db": str(args.source_db),
        "output_db": str(args.output_db),
        "extracted": str(args.extracted),
        "source_read_only": True,
        "query_agnostic_extraction": True,
        "totals": dict(totals),
        "turns_absent_from_source": sorted(
            set(packets_by_turn) - set(turn_locations)),
        "rows": rows,
    }
    manifest_path = args.output_db.parent / "atomic_augmentation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_db),
        "manifest": str(manifest_path),
        "totals": dict(totals),
        "turns_absent_from_source": len(manifest["turns_absent_from_source"]),
    }, ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
