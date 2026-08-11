#!/usr/bin/env python3
"""Materialize the input-routed V5.45 prompt policy.

The route uses only observable request content:

* anonymous two-role memories receive the typed readout contract; and
* named multi-party memories retain the frozen prompt, except that
  what/when/who/which queries place the strongest graph block last.

Aggregation-ledger and preference prompts remain byte-identical.  The policy
does not inspect benchmark labels, expected answers, gold turns, or judge
outcomes.  It also requires every output prompt to stay within its source
packing-token count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from materialize_v5_42_recency_layout import transform as recency_transform
from materialize_v5_43_typed_readout import (
    _named_multi_party,
    _split_evidence,
    transform as typed_transform,
)

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.tokenization import ExactTokenCounter  # noqa: E402


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _nearest(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return (ordered[max(0, math.ceil(probability * len(ordered)) - 1)]
            if ordered else 0)


def _stats(values: list[int]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / max(1, len(values)),
        "p50": _nearest(values, .50),
        "p95": _nearest(values, .95),
        "p99": _nearest(values, .99),
        "max": max(values, default=0),
        "percentile_method": "nearest-rank",
    }


def _route(row: dict) -> str:
    messages = row.get("messages") or ()
    trace = row.get("trace") or {}
    if not messages:
        return "deterministic"
    if trace.get("aggregation_ledger") or trace.get("preference_synthesis"):
        return "specialized_frozen"
    _prefix, evidence, _suffix = _split_evidence(
        str(messages[-1].get("content") or ""))
    return "named" if _named_multi_party(evidence) else "anonymous_typed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = _read(args.source)
    counter = ExactTokenCounter("frozen-packing-model", args.tokenizer)
    transformed: list[dict] = []
    routes: Counter[str] = Counter()
    deltas: list[int] = []
    for row in rows:
        route = _route(row)
        if route == "anonymous_typed":
            output, _delta, _kind, _recency = typed_transform(row, counter)
        elif route == "named":
            # The typed materializer already implements the conservative
            # named-speaker + question-prefix gate.  Use it only to decide
            # whether recency applies; keep the typed contract out of this
            # branch.
            _typed, _delta, _kind, recency = typed_transform(row, counter)
            if recency:
                output, _delta, _blocks = recency_transform(row, counter)
                route = "named_recency"
            else:
                output = dict(row)
                route = "named_frozen"
        else:
            output = dict(row)
        delta = int(output.get("packing_prompt_tokens") or 0) - int(
            row.get("packing_prompt_tokens") or 0)
        if delta > 0:
            raise ValueError(
                f"{row['question_id']}: routed prompt increased by {delta} tokens")
        if set(output.get("evidence_turn_ids") or ()) != set(
                row.get("evidence_turn_ids") or ()):
            raise ValueError(f"{row['question_id']}: evidence set changed")
        transformed.append(output)
        routes[route] += 1
        deltas.append(delta)

    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in transformed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    changed = [
        str(before["question_id"])
        for before, after in zip(rows, transformed)
        if before.get("prompt_payload_hash") != after.get("prompt_payload_hash")
    ]
    manifest = {
        "schema_version": "graphmem-v5.45-input-routed-prompt-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(rows),
        "changed": len(changed),
        "changed_question_ids": changed,
        "routes": dict(sorted(routes.items())),
        "route_inputs": [
            "named-versus-anonymous speaker form",
            "question prefix",
            "existing specialized-path trace",
        ],
        "benchmark_gold_or_judge_routing": False,
        "aggregation_and_preference_frozen": True,
        "evidence_set_frozen": True,
        "packing_token_delta": _stats(deltas),
        "increased_questions": sum(delta > 0 for delta in deltas),
        "tokenizer": counter.describe(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
