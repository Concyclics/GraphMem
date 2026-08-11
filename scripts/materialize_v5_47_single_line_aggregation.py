#!/usr/bin/env python3
"""Add a budget-nonincreasing single-line stop contract to V5.46 cards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import canonical_json  # noqa: E402
from graphmem.tokenization import ExactTokenCounter  # noqa: E402


VERSION = "graphmem-v5.47-single-line-aggregation-stop-v1"
OLD = (
    "Use graph adjacency only to find related evidence; sharing a graph block "
    "does not by itself make a memory an operand.\n"
    "Compute and answer this exact Question now: "
)
NEW = (
    "Graph proximity is not operand proof.\n"
    "Question: "
)
STOP = (
    "\nOutput one concise final-answer line only, then stop; never list evidence "
    "or repeat."
)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _nearest(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)] if ordered else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    counter = ExactTokenCounter("frozen-packing-model", args.tokenizer)
    source_rows = _read(args.source)
    output_rows: list[dict] = []
    deltas: list[int] = []
    changed = 0
    for row in source_rows:
        trace = dict(row.get("trace") or {})
        messages = [dict(message) for message in row.get("messages") or ()]
        if not trace.get("aggregation_execution_card") or not messages:
            output_rows.append(dict(row)); deltas.append(0); continue
        user = str(messages[-1].get("content") or "")
        if OLD not in user:
            raise ValueError(f"V5.46 card suffix absent for {row['question_id']}")
        messages[-1]["content"] = user.replace(OLD, NEW, 1) + STOP
        source_tokens = int(row.get("packing_prompt_tokens") or 0)
        prompt_tokens = sum(counter.count(message["content"]) for message in messages)
        if prompt_tokens > source_tokens:
            raise ValueError(
                f"{row['question_id']}: stop contract increased "
                f"{source_tokens} -> {prompt_tokens}")
        prompt_version = "+".join(filter(None, (
            str(trace.get("prompt_version") or ""), VERSION)))
        trace.update({
            "prompt_version": prompt_version,
            "aggregation_single_line_stop": True,
            "aggregation_single_line_source_payload_hash": row.get(
                "prompt_payload_hash"),
        })
        output = dict(row)
        output.update({
            "messages": messages,
            "packing_prompt_tokens": prompt_tokens,
            "prompt_hash": hashlib.sha256(
                (prompt_version + messages[0]["content"]).encode()).hexdigest(),
            "prompt_payload_hash": hashlib.sha256(
                canonical_json(messages).encode()).hexdigest(),
            "trace": trace,
            "preparation_latency_ms": 0.0,
        })
        if output.get("evidence_turn_ids") != row.get("evidence_turn_ids"):
            raise ValueError(f"{row['question_id']}: evidence order changed")
        output_rows.append(output)
        deltas.append(prompt_tokens - source_tokens)
        changed += 1

    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.47-single-line-aggregation-arm-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(source_rows), "changed": changed,
        "evidence_set_and_order_frozen": True,
        "benchmark_gold_or_judge_routing": False,
        "packing_token_delta": {
            "count": len(deltas), "mean": sum(deltas) / max(1, len(deltas)),
            "p50": _nearest(deltas, .50), "p95": _nearest(deltas, .95),
            "p99": _nearest(deltas, .99), "max": max(deltas, default=0),
            "percentile_method": "nearest-rank",
        },
        "increased_questions": sum(delta > 0 for delta in deltas),
        "tokenizer": counter.describe(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
