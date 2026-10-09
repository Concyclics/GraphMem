#!/usr/bin/env python3
"""Verify that a QueryIR-routed prompt is an exact per-route composition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(
            encoding="utf-8").split("\n") if line.strip()
    ]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate question IDs in {path}")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed", type=Path, required=True)
    parser.add_argument("--mapped", type=Path, required=True)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--mapped-routes", default="inference")
    parser.add_argument("--expected", type=int, default=1540)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    routed = keyed(args.routed)
    mapped = keyed(args.mapped)
    plain = keyed(args.plain)
    audit = keyed(args.audit)
    expected_ids = set(routed)
    for label, rows in (("mapped", mapped), ("plain", plain),
                        ("audit", audit)):
        if set(rows) != expected_ids:
            raise RuntimeError(f"{label} question IDs differ from routed")
    if args.expected and len(routed) != args.expected:
        raise RuntimeError(f"expected {args.expected}, got {len(routed)}")
    mapped_routes = frozenset(
        value.strip() for value in args.mapped_routes.split(",")
        if value.strip())
    if not mapped_routes:
        raise ValueError("mapped-routes cannot be empty")

    mismatches: list[dict[str, str]] = []
    route_counts: dict[str, int] = {}
    deltas: list[int] = []
    for question_id, row in routed.items():
        route = str(audit[question_id]["query_route"])
        route_counts[route] = route_counts.get(route, 0) + 1
        source = mapped if route in mapped_routes else plain
        expected_row = source[question_id]
        if (row.get("prompt_payload_hash")
                != expected_row.get("prompt_payload_hash")
                or row.get("messages") != expected_row.get("messages")):
            mismatches.append({
                "question_id": question_id,
                "query_route": route,
                "expected_arm": (
                    "mapped" if route in mapped_routes else "plain"),
            })
        deltas.append(
            int(row.get("packing_prompt_tokens") or 0)
            - int(plain[question_id].get("packing_prompt_tokens") or 0))

    payload = {
        "schema_version": "graphmem-v5.79-routed-prompt-audit-v1",
        "passed": not mismatches,
        "questions": len(routed),
        "mapped_routes": sorted(mapped_routes),
        "mapped_questions": sum(
            count for route, count in route_counts.items()
            if route in mapped_routes),
        "plain_questions": sum(
            count for route, count in route_counts.items()
            if route not in mapped_routes),
        "route_counts": dict(sorted(route_counts.items())),
        "exact_prompt_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "token_delta_vs_plain": {
            "mean": sum(deltas) / max(1, len(deltas)),
            "p95": nearest_rank(deltas, 0.95),
            "p99": nearest_rank(deltas, 0.99),
            "max": max(deltas, default=0),
            "sum": sum(deltas),
            "nonzero_questions": sum(value != 0 for value in deltas),
            "percentile_method": "nearest_rank",
        },
        "inputs": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in (
                ("routed", args.routed), ("mapped", args.mapped),
                ("plain", args.plain), ("audit", args.audit))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
