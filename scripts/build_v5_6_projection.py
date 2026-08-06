#!/usr/bin/env python3
"""Project extra deterministic structure onto a frozen graph, into a new database.

Zero LLM calls and zero gold labels: every rule reads only attributes the build
already wrote.  The source database is opened read-only and never modified; the
arm is written to its own copy so P-series arms can be compared side by side.

The copy carries the graph and its provenance but drops llm_calls, llm_cache and
embeddings, which are identical across arms and account for ~70% of the bytes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import NodeType  # noqa: E402
from graphmem.projection import ARMS, ProjectionConfig, build_manifests, manifest_stats  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", default="P1", help=f"one of {sorted(ARMS)}")
    parser.add_argument("--keep-ledger", action="store_true",
                        help="keep llm_calls/llm_cache/embeddings instead of dropping them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.arm not in ARMS:
        raise SystemExit(f"unknown arm {args.arm!r}; choose from {sorted(ARMS)}")
    config: ProjectionConfig = ARMS[args.arm]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = args.output_root / f"projection_{args.arm}_{stamp}"
    root.mkdir(parents=True)
    target = root / "graphmem.sqlite"

    print(f"copying {args.source_db} -> {target}", flush=True)
    shutil.copy2(args.source_db, target)
    if not args.keep_ledger:
        # These tables are byte-identical across arms; dropping them turns a
        # ~12GB arm into a ~2.5GB one at 510 memories.
        connection = sqlite3.connect(target)
        for table in ("llm_calls", "llm_cache", "embeddings", "embedding_calls"):
            try:
                connection.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        connection.commit()
        connection.execute("VACUUM")
        connection.close()

    store = SQLiteGraphStore(target)
    memories = [row[0] for row in store._read("SELECT memory_id FROM conversations ORDER BY memory_id")]
    print(f"projecting arm {args.arm} over {len(memories)} memories", flush=True)

    totals = {"memories": 0, "manifest_nodes": 0, "member_edges": 0}
    per_memory = []
    for index, memory_id in enumerate(memories, 1):
        nodes = list(store.nodes(memory_id))
        edges = list(store.edges(memory_id))
        before = len(nodes)
        manifest_nodes, manifest_edges, rows = build_manifests(memory_id, nodes, config)
        if manifest_nodes or manifest_edges:
            # Evidence groups are passed through unchanged: the projection cites
            # existing provenance and never mints a new group, so re-writing them
            # keeps replace_graph's referential checks satisfied.
            store.replace_graph(memory_id, nodes + manifest_nodes, edges + manifest_edges,
                                list(store.evidence_groups(memory_id)))
        stats = dict(manifest_stats(rows))
        stats.update({"memory_id": memory_id, "nodes_before": before,
                      "nodes_after": before + len(manifest_nodes),
                      "facts": sum(1 for node in nodes if node.node_type == NodeType.CANONICAL_FACT)})
        per_memory.append(stats)
        totals["memories"] += 1
        totals["manifest_nodes"] += len(manifest_nodes)
        totals["member_edges"] += len(manifest_edges)
        if index % 25 == 0:
            print(f"  projected {index}/{len(memories)}", flush=True)

    summary = {
        "arm": args.arm, "config": config.__dict__ if hasattr(config, "__dict__") else str(config),
        "config_digest": config.digest(), "source_db": str(args.source_db),
        "target_db": str(target), "generated_at": stamp,
        **totals,
        "single_member_manifests": sum(int(row.get("single_member_manifests", 0)) for row in per_memory),
        "invisible_to_frozen_build": sum(int(row.get("invisible_to_frozen_build", 0)) for row in per_memory),
        "generative_llm_calls": 0,
    }
    (root / "projection_summary.json").write_text(
        json.dumps({"summary": summary, "per_memory": per_memory}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    store.close()


if __name__ == "__main__":
    main()
