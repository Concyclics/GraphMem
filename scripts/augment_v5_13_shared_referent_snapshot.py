#!/usr/bin/env python3
"""Clone a graph and add bounded cross-session scene referent edges safely."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_shared_referent_edges import STOPWORDS, WORD, scene_terms  # noqa: E402
from graphmem.domain import GraphEdge, RelationType, stable_id  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--df-share", type=float, default=0.05)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--max-degree", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.df_share <= 1 or args.min_shared < 1 or args.max_degree < 1:
        raise ValueError("invalid shared-referent bounds")
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
        scene_turns, scene_session, turn_text, df, session_count = scene_terms(
            store._connection, memory_id)
        threshold = max(2, session_count * args.df_share)
        rare = {term for term, count in df.items() if count <= threshold}
        idf = {term: math.log(session_count / max(1, df[term])) for term in rare}
        scenes = []
        for scene_id, session_id in scene_session.items():
            terms = set()
            for turn_id in scene_turns.get(scene_id, ()):
                terms |= {word.lower() for word in WORD.findall(
                    turn_text.get(turn_id, ""))} - STOPWORDS
            terms &= rare
            if terms:
                scenes.append((scene_id, session_id, terms))
        postings: dict[str, list[int]] = defaultdict(list)
        for index, (_scene_id, _session_id, terms) in enumerate(scenes):
            for term in terms:
                postings[term].append(index)
        pair_score: dict[tuple[int, int], float] = defaultdict(float)
        pair_count: Counter[tuple[int, int]] = Counter()
        for term, members in postings.items():
            if len(members) > 64:
                continue
            for left_index, left in enumerate(members):
                for right in members[left_index + 1:]:
                    if scenes[left][1] == scenes[right][1]:
                        continue
                    pair = (min(left, right), max(left, right))
                    pair_score[pair] += idf[term]
                    pair_count[pair] += 1
        ranked: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for (left, right), count in pair_count.items():
            if count >= args.min_shared:
                ranked[left].append((pair_score[(left, right)], right))
                ranked[right].append((pair_score[(left, right)], left))
        kept_pairs = set()
        for left, candidates in ranked.items():
            for _score, right in sorted(
                    candidates, key=lambda row: (-row[0], row[1]))[:args.max_degree]:
                kept_pairs.add((min(left, right), max(left, right)))

        nodes = tuple(store.nodes(memory_id))
        by_id = {node.node_id: node for node in nodes}
        edges = [edge for edge in store.edges(memory_id)
                 if edge.relation != RelationType.SHARED_REFERENT]
        added = 0
        for left_index, right_index in sorted(kept_pairs):
            left_id, right_id = scenes[left_index][0], scenes[right_index][0]
            left = by_id.get(left_id); right = by_id.get(right_id)
            if left is None or right is None:
                continue
            groups = tuple(dict.fromkeys((left.evidence_group_id,
                                          right.evidence_group_id)))
            edges.append(GraphEdge(
                stable_id("edge", memory_id, RelationType.SHARED_REFERENT,
                          left_id, right_id),
                memory_id, left_id, RelationType.SHARED_REFERENT, right_id,
                groups[0], False, min(1.0, pair_score[
                    (left_index, right_index)] / 20.0),
                "shared_referent_v2_bounded", groups[1:]))
            added += 1
        if added:
            store.replace_graph(
                memory_id, nodes, tuple(edges),
                tuple(store.evidence_groups(memory_id)))
        totals.update({"memories": 1, "edges_added": added,
                       "candidate_pairs": len(pair_count)})
        per_memory.append({"memory_id": memory_id, "sessions": session_count,
                           "candidate_pairs": len(pair_count), "edges_added": added})
    manifest = {
        "schema_version": "graphmem-v5.13-shared-referent-snapshot-v2",
        "source": str(args.source), "output": str(args.output),
        "parameters": {"df_share": args.df_share,
                       "min_shared": args.min_shared,
                       "max_degree": args.max_degree,
                       "max_term_posting": 64},
        "totals": dict(totals), "per_memory": per_memory,
    }
    manifest_path = args.output.parent / (
        args.output.stem + "_shared_referent_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output), "totals": dict(totals)},
                     ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
