#!/usr/bin/env python3
"""Overlay runtime-hydrated aggregation prompts on a validated base policy.

The route is deliberately label-free: every question mechanically classified
as an aggregation operation uses the freshly packed V5.50 execution-card row;
all other questions remain byte-identical to the supplied base policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _index(path: Path) -> tuple[list[str], dict[str, dict]]:
    rows = _read(path)
    order = [str(row["question_id"]) for row in rows]
    indexed = {str(row["question_id"]): row for row in rows}
    if len(order) != len(indexed):
        raise ValueError(f"duplicate question ID in {path}")
    return order, indexed


def _operation(row: dict) -> str:
    return str((((row.get("trace") or {}).get("aggregation_ledger") or {})
                .get("operation") or ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runtime-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    order, base = _index(args.base)
    runtime_order, runtime = _index(args.runtime_card)
    if runtime_order != order:
        raise ValueError("runtime-card question order differs from base")

    routes: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    packing_deltas: list[int] = []
    output_rows: list[dict] = []
    for question_id in order:
        base_row = base[question_id]
        runtime_row = runtime[question_id]
        operation = _operation(runtime_row)
        trace = runtime_row.get("trace") or {}
        ledger = trace.get("aggregation_ledger") or {}
        if operation and ledger.get("execution_card"):
            chosen = runtime_row
            route = "runtime_card"
            operations[operation] += 1
        else:
            chosen = base_row
            route = "base"
            if chosen.get("prompt_payload_hash") != base_row.get(
                    "prompt_payload_hash"):
                raise AssertionError("base route lost prompt identity")
        routes[route] += 1
        packing_deltas.append(
            int(chosen.get("packing_prompt_tokens") or 0)
            - int(base_row.get("packing_prompt_tokens") or 0))
        output_rows.append(chosen)

    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.50-runtime-card-policy-v1",
        "sources": {
            "base": {"path": str(args.base),
                     "sha256": hashlib.sha256(args.base.read_bytes()).hexdigest()},
            "runtime_card": {
                "path": str(args.runtime_card),
                "sha256": hashlib.sha256(
                    args.runtime_card.read_bytes()).hexdigest(),
            },
        },
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(order),
        "routes": dict(sorted(routes.items())),
        "operations": dict(sorted(operations.items())),
        "routing_inputs": ["mechanically classified aggregation operation"],
        "benchmark_gold_prediction_or_judge_routing": False,
        "packing_token_delta": {
            "count": len(packing_deltas),
            "mean": sum(packing_deltas) / max(1, len(packing_deltas)),
            "min": min(packing_deltas, default=0),
            "max": max(packing_deltas, default=0),
            "increased_questions": sum(value > 0 for value in packing_deltas),
            "decreased_questions": sum(value < 0 for value in packing_deltas),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
