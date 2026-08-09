#!/usr/bin/env python3
"""Compile versioned per-memory turn-vector indexes outside query workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.retrieval.dense_sidecar import DENSE_INDEX_SCHEMA  # noqa: E402
from graphmem.serving import sync_dense_sidecars  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="graph build config that identifies the embedding model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "numpy_exact", "faiss_flat"),
                        default="auto")
    parser.add_argument("--memory-ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    store = SQLiteGraphStore(args.db, read_only=True)
    try:
        selected = tuple(
            value.strip() for value in args.memory_ids.split(",") if value.strip())
        memory_ids = selected or store.memory_ids()
    finally:
        store.close()
    if args.limit > 0:
        memory_ids = memory_ids[:args.limit]
    result = sync_dense_sidecars(
        args.db, args.output,
        model_id=config.models.embedding_model,
        backend=args.backend,
        memory_ids=memory_ids,
        workers=args.workers,
        force=args.force,
    )
    rows = list(result["rows"])
    manifest = {
        "schema_version": "graphmem-v5.18-dense-precompile-manifest-v1",
        "dense_artifact_schema": DENSE_INDEX_SCHEMA,
        "db": str(args.db),
        "output": str(args.output),
        "requested_backend": args.backend,
        "model_id": config.models.embedding_model,
        "requested": len(memory_ids),
        "compiled": int(result["compiled"]),
        "current": int(result["current"]),
        "failed": int(result["failed"]),
        "wall_time_sec": float(result["elapsed_ms"]) / 1000,
        "compile_latency_ms_mean": (
            statistics.fmean(float(row["elapsed_ms"]) for row in rows) if rows else 0.0),
        "data_bytes": sum(int(row.get("data_bytes", 0)) for row in rows),
        "artifacts": rows,
        "failures": result["failures"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "artifacts"}, ensure_ascii=False, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
