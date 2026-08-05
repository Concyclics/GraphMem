#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase

from graphmem.domain import RelationType
from graphmem.runtime import Neo4jCachedRuntime, Neo4jDirectRuntime, SQLiteSnapshotRuntime
from graphmem.storage import SQLiteGraphStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = (args.user, args.password)
    store = SQLiteGraphStore(args.sqlite)
    sqlite_runtime = SQLiteSnapshotRuntime(store)
    direct = Neo4jDirectRuntime(args.uri, auth)
    cached = Neo4jCachedRuntime(args.uri, auth)
    memories = [row[0] for row in store._connection.execute(
        "SELECT memory_id FROM conversations ORDER BY memory_id"
    )]
    sample_indices = sorted({0, len(memories) // 4, len(memories) // 2,
                             3 * len(memories) // 4, len(memories) - 1})
    parity = []
    for index in sample_indices:
        memory_id = memories[index]
        node_ids = [node.node_id for node in store.nodes(memory_id)[:5]]
        sqlite_edges = sqlite_runtime.expand(memory_id, node_ids, list(RelationType), limit=100)
        direct_edges = direct.expand(memory_id, node_ids, list(RelationType), limit=100)
        cached_edges = cached.expand(memory_id, node_ids, list(RelationType), limit=100)
        ids = lambda rows: sorted(edge.edge_id for edge in rows)
        if ids(sqlite_edges) != ids(direct_edges) or ids(sqlite_edges) != ids(cached_edges):
            raise RuntimeError(f"runtime path parity failed for {memory_id}")
        parity.append({"memory_id": memory_id, "frontier": node_ids,
                       "edge_ids": ids(sqlite_edges)})
    driver = GraphDatabase.driver(args.uri, auth=auth)
    forbidden = {"raw_text", "request", "request_json", "response", "response_json",
                 "embedding", "vector"}
    with driver.session() as session:
        counts = dict(session.run(
            "MATCH (p:GraphMemProjection) WITH count(p) AS memories "
            "MATCH (n:GraphMemNode) WITH memories,count(n) AS nodes "
            "MATCH ()-[r:GRAPHMEM_EDGE]->() RETURN memories,nodes,count(r) AS edges"
        ).single())
        node_keys = set(session.run(
            "MATCH (n:GraphMemNode) UNWIND keys(n) AS key RETURN collect(DISTINCT key) AS keys"
        ).single()["keys"])
        edge_keys = set(session.run(
            "MATCH ()-[r:GRAPHMEM_EDGE]->() UNWIND keys(r) AS key RETURN collect(DISTINCT key) AS keys"
        ).single()["keys"])
    driver.close()
    leaked = sorted(forbidden & (node_keys | edge_keys))
    if leaked:
        raise RuntimeError(f"forbidden Neo4j properties: {leaked}")
    sqlite_counts = {
        "memories": len(memories),
        "nodes": int(store._connection.execute("SELECT count(*) FROM graph_nodes").fetchone()[0]),
        "edges": int(store._connection.execute("SELECT count(*) FROM graph_edges").fetchone()[0]),
    }
    neo4j_counts = {key: int(value) for key, value in counts.items()}
    if sqlite_counts != neo4j_counts:
        raise RuntimeError(f"global projection counts differ: {sqlite_counts} != {neo4j_counts}")
    payload = {
        "sqlite_counts": sqlite_counts, "neo4j_counts": neo4j_counts,
        "forbidden_property_leaks": leaked, "runtime_parity_samples": parity,
        "runtime_modes": ["sqlite_snapshot", "neo4j_direct", "neo4j_cached"],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cached.close(); direct.close(); store.close()
    print(args.output)


if __name__ == "__main__":
    main()
