#!/usr/bin/env python3
"""Build an immutable V5.5 Cat1 CollectionManifest sidecar without LLM calls."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from graphmem.domain import GraphEdge, GraphNode, NodeType, RelationType, stable_id
from graphmem.storage import SQLiteGraphStore
from graphmem.text import normalize_key


def edge(memory_id: str, left: GraphNode, relation: RelationType, right: GraphNode) -> GraphEdge:
    groups = tuple(dict.fromkeys((*left.all_evidence_group_ids, *right.all_evidence_group_ids)))
    return GraphEdge(stable_id("edge", memory_id, left.node_id, relation, right.node_id), memory_id,
                     left.node_id, relation, right.node_id, groups[0], True, 1.0,
                     "v5_5_collection_manifest", groups[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--h6-metrics", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--speaker-terminal-postings", action="store_true")
    args = parser.parse_args()
    if args.output_db.exists():
        raise FileExistsError(args.output_db)
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_db, args.output_db)
    residual = set()
    for line in args.h6_metrics.open():
        row = json.loads(line)
        if row.get("configuration") == "h6" and row.get("stratum") == "locomo_multihop" and not row.get("candidate_turn_all_hit"):
            residual.add(str(row["memory_id"]))
    store = SQLiteGraphStore(args.output_db); details = {}
    for memory_id in sorted(residual):
        nodes = list(store.nodes(memory_id)); edges = list(store.edges(memory_id)); by_id = {node.node_id: node for node in nodes}
        chains: dict[tuple[str, str, str, str, str, str], list[GraphNode]] = defaultdict(list)
        for node in nodes:
            if node.node_type != NodeType.CANONICAL_FACT:
                continue
            attrs = node.attributes
            chains[(str(attrs.get("owner_id", "")), str(attrs.get("predicate", "")), str(attrs.get("scope", "")),
                    str(attrs.get("collection_key", "")), str(attrs.get("polarity", "positive")),
                    str(attrs.get("modality", "asserted")))].append(node)
        added = 0
        for key, facts in sorted(chains.items()):
            owner_id, predicate, scope, collection_key, polarity, modality = key
            owner = by_id.get(owner_id)
            if not owner or not facts:
                continue
            ordered = sorted(facts, key=lambda row: (str(row.attributes.get("session_id", "")),
                                                       int(row.attributes.get("turn_index", 0)), row.node_id))
            groups = tuple(dict.fromkeys(group for fact in ordered for group in fact.all_evidence_group_ids))
            manifest_id = stable_id("node", memory_id, "collection-manifest", *key)
            manifest = GraphNode(
                manifest_id, memory_id, NodeType.COLLECTION_MANIFEST, 1,
                f"{owner.summary} {predicate} " + " ".join(str(row.attributes.get("value", "")) for row in ordered[:8]),
                groups[0], groups[1:], attributes={
                    "owner_id": owner_id, "predicate": predicate, "scope": scope,
                    "collection_key": collection_key, "polarity": polarity, "modality": modality,
                    "member_fact_ids": tuple(row.node_id for row in ordered),
                    "member_value_keys": tuple(dict.fromkeys(str(row.attributes.get("value_key", "")) for row in ordered)),
                    "observed_member_count": len(ordered),
                    "collection_semantics": ("event_instances" if any(row.attributes.get("event_instance_id") for row in ordered)
                                             else "distinct_values"),
                    "extraction_coverage": "semantic_observed", "open_world": True,
                    "roles": ("collection_scope", "route"), "provenance_scope": "route",
                })
            if manifest_id not in by_id:
                nodes.append(manifest); by_id[manifest_id] = manifest; added += 1
            edges.append(edge(memory_id, owner, RelationType.HAS_FACT, manifest))
            for fact in ordered:
                edges.append(edge(memory_id, manifest, RelationType.MEMBER_OF, fact))
        speaker_edges = 0
        if args.speaker_terminal_postings:
            aliases: dict[str, str] = {}
            for node in nodes:
                if node.node_type != NodeType.CANONICAL_ENTITY:
                    continue
                for alias in (node.summary, *node.attributes.get("aliases", ())):
                    normalized = normalize_key(str(alias))
                    if normalized:
                        aliases.setdefault(normalized, node.node_id)
            refs = {str(node.attributes.get("turn_id")): node for node in nodes
                    if node.node_type == NodeType.EVIDENCE_GROUP_REF and node.attributes.get("turn_id")}
            for turn in store.turns(memory_id):
                owner_id = aliases.get(normalize_key(turn.speaker)); ref = refs.get(turn.turn_id)
                if owner_id and ref and owner_id in by_id:
                    edges.append(edge(memory_id, by_id[owner_id], RelationType.HAS_FACT, ref)); speaker_edges += 1
        nodes = list({node.node_id: node for node in nodes}.values())
        edges = list({row.edge_id: row for row in edges}.values())
        store.replace_graph(memory_id, nodes, edges, store.evidence_groups(memory_id))
        details[memory_id] = {"collection_manifests_added": added, "fact_chains": len(chains),
                              "speaker_terminal_postings": speaker_edges}
    args.manifest.write_text(json.dumps({"source_db": str(args.source_db), "output_db": str(args.output_db),
                                         "residual_memories": sorted(residual), "details": details,
                                         "speaker_terminal_postings": args.speaker_terminal_postings,
                                         "generative_llm_calls": 0}, indent=2) + "\n")
    store.close()


if __name__ == "__main__":
    main()
