#!/usr/bin/env python3
"""Compile immutable graph/turn/provenance views outside hot query workers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.retrieval.compiled_memory import COMPILED_MEMORY_SCHEMA  # noqa: E402
from graphmem.serving import sync_compiled_sidecars  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=WORKSPACE /
        "artifacts/v5_10/full_benchmark_20260809/graph/report_graph.sqlite")
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE /
        "artifacts/report/v5_11/compiled_memory_views")
    parser.add_argument(
        "--memory-ids", default="",
        help="Optional comma-separated memory ids; default is every published graph.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--approximate-bytes", action="store_true",
        help="Skip the offline deep retained-size traversal (faster, less precise).")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    return rows[min(len(rows) - 1, max(0, math.ceil(len(rows) * q) - 1))]


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    store = SQLiteGraphStore(args.db, read_only=True)
    try:
        selected = tuple(
            value.strip() for value in args.memory_ids.split(",") if value.strip())
        memory_ids = selected or store.memory_ids()
    finally:
        store.close()
    if args.limit > 0:
        memory_ids = memory_ids[:args.limit]
    if not memory_ids:
        raise SystemExit("no published graph memories selected")
    result = sync_compiled_sidecars(
        args.db,
        args.output,
        memory_ids=memory_ids,
        workers=args.workers,
        force=args.force,
        account_bytes=not args.approximate_bytes,
    )
    rows = list(result["rows"])
    failures = list(result["failures"])
    elapsed_rows = [float(row["elapsed_ms"]) for row in rows]
    manifest = {
        "schema_version": "graphmem-v5.11-precompile-manifest-v1",
        "compiled_artifact_schema": COMPILED_MEMORY_SCHEMA,
        "db": str(args.db),
        "output": str(args.output),
        "workers": args.workers,
        "accounted_retained_bytes": not args.approximate_bytes,
        "requested": len(memory_ids),
        "compiled": int(result["compiled"]),
        "current": int(result["current"]),
        "failed": len(failures),
        "wall_time_sec": float(result["elapsed_ms"]) / 1000,
        "compile_latency_ms": {
            "mean": statistics.fmean(elapsed_rows) if elapsed_rows else 0.0,
            "p95": percentile(elapsed_rows, .95),
        },
        "serialized_bytes": sum(int(row.get("serialized_bytes", 0)) for row in rows),
        "retained_bytes": sum(int(row.get("total_retained_bytes", 0)) for row in rows),
        "artifacts": rows,
        "failures": failures,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"artifacts"}}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
