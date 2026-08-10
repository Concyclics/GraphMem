#!/usr/bin/env python3
"""Clone frozen extraction/vector caches while removing every materialized graph."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


GRAPH_TABLES = (
    "graph_edges", "graph_nodes", "evidence_members", "evidence_groups",
    "graph_checksum_state", "graph_versions", "outbox",
    "incremental_job_events", "incremental_jobs", "run_ledger",
)


def backup(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as reader:
        with sqlite3.connect(target) as writer:
            reader.backup(writer)


def reset_graph(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        existing = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        before = {}
        connection.execute("BEGIN IMMEDIATE")
        for table in GRAPH_TABLES:
            if table in existing:
                before[table] = int(connection.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                connection.execute(f"DELETE FROM {table}")
        # Cache payloads and vectors are the frozen source for the paired
        # ablation.  Call ledgers are run-specific and must start empty so a
        # cache miss is visible rather than hidden by the canonical arm.
        for table in ("llm_calls", "embedding_calls"):
            if table in existing:
                before[table] = int(connection.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                connection.execute(f"DELETE FROM {table}")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")
        return before
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--target-db", type=Path, required=True)
    parser.add_argument("--relation-source", type=Path)
    parser.add_argument("--relation-target", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.relation_source) != bool(args.relation_target):
        raise ValueError("relation-source and relation-target must be supplied together")

    backup(args.source_db, args.target_db)
    removed = reset_graph(args.target_db)
    relation = None
    if args.relation_source and args.relation_target:
        backup(args.relation_source, args.relation_target)
        relation_removed = reset_graph(args.relation_target)
        relation = {"source": str(args.relation_source),
                    "target": str(args.relation_target),
                    "removed_rows": relation_removed}
    payload = {
        "schema_version": "graphmem-v5.19-ablation-seed-v1",
        "source_db": str(args.source_db), "target_db": str(args.target_db),
        "relation": relation, "removed_rows": removed,
        "retained": ["conversations", "sessions", "source_turns", "llm_cache",
                     "embeddings"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
