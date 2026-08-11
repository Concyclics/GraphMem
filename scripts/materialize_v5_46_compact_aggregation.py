#!/usr/bin/env python3
"""Replace the duplicated 32-row aggregation ledger with an execution card.

The evidence set and graph layout stay frozen.  The old ledger duplicated
snippets from up to 32 already-present turns and placed many irrelevant numbers
next to the answer boundary.  This arm retains the mechanically classified
operation, removes the duplicated candidate list, and places a concise,
operation-specific readout procedure plus the original question last.

The transformation reads no benchmark label, expected answer, gold turn, or
judge outcome and is required to reduce (never increase) packing tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.prompts import AGGREGATION_LEDGER_APPENDIX  # noqa: E402
from graphmem.domain import canonical_json  # noqa: E402
from graphmem.tokenization import ExactTokenCounter  # noqa: E402


VERSION = "graphmem-v5.46-compact-aggregation-execution-v1"
LEDGER_MARKER = "\n\nAggregation ledger ("
OUTPUT_MARKER = "\n\nOutput contract:"
QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(?P<question>[^\n]+)")
OPERATION_RE = re.compile(r"(?:^|\n)Operation:\s*(?P<operation>[a-z_]+)")

COMPACT_SYSTEM_APPENDIX = (
    " An Aggregation execution card follows the graph-grouped memories. It "
    "specifies the requested operation but is not evidence. Select the complete "
    "operand set only from direct memories matching the exact subject, relation, "
    "time, polarity, and completion state. Deduplicate repeated mentions of one "
    "occurrence, retain distinct occurrences, and never treat an absent operand "
    "as zero. Return only the computed answer."
)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _nearest(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return (ordered[max(0, math.ceil(probability * len(ordered)) - 1)]
            if ordered else 0)


def _stats(values: list[int]) -> dict:
    return {
        "count": len(values), "mean": sum(values) / max(1, len(values)),
        "p50": _nearest(values, .50), "p95": _nearest(values, .95),
        "p99": _nearest(values, .99), "max": max(values, default=0),
        "percentile_method": "nearest-rank",
    }


def _rule(operation: str) -> str:
    rules = {
        "count_distinct": (
            "Enumerate the complete set of distinct qualifying items or completed "
            "occurrences from direct statements; exclude plans, suggestions, "
            "near-matches, and duplicate mentions; then count the set once."),
        "sum": (
            "Collect every distinct unit-compatible amount for the exact subject "
            "and scope; exclude plans, unrelated amounts, stated subtotals, and "
            "duplicate mentions; then add the operands once."),
        "difference": (
            "Bind the exact two requested quantities. For remaining or needed, "
            "use target minus the latest current amount; preserve the requested "
            "order, sign, and unit."),
        "date_difference": (
            "Bind the exact start and end events, resolve each from its own memory "
            "date or [source-time], subtract the calendar endpoints in the "
            "requested unit, and ignore unrelated durations or booking lead time."),
        "mean": (
            "Bind the complete requested population, use one value per distinct "
            "member, sum those values, and divide once by the member count."),
        "minimum": (
            "Bind all qualifying values for the exact scope and return the minimum "
            "with its original unit; ignore unrelated values and near-matches."),
        "maximum": (
            "Bind all qualifying values for the exact scope and return the maximum "
            "with its original unit; ignore unrelated values and near-matches."),
    }
    return rules.get(operation, (
        "Select the complete exact operand set, apply the named operation once, "
        "and preserve the requested unit."))


def transform(row: dict, counter: ExactTokenCounter) -> tuple[dict, int, str]:
    trace = dict(row.get("trace") or {})
    ledger_trace = trace.get("aggregation_ledger") or {}
    messages = [dict(message) for message in row.get("messages") or ()]
    if not ledger_trace or not messages:
        return dict(row), 0, "frozen"
    if len(messages) != 2 or messages[-1].get("role") != "user":
        raise ValueError(f"unexpected aggregation messages for {row['question_id']}")

    user = str(messages[-1].get("content") or "")
    ledger_at = user.find(LEDGER_MARKER)
    output_at = user.find(OUTPUT_MARKER, ledger_at + 1)
    if ledger_at < 0:
        raise ValueError(f"aggregation marker absent for {row['question_id']}")
    # Older specialized prompts end at the ledger; newer grounded variants
    # append an output contract.  Both layouts are valid frozen inputs.
    if output_at < 0:
        output_at = len(user)
    question_match = QUESTION_RE.search(user)
    ledger_text = user[ledger_at:output_at]
    operation_match = OPERATION_RE.search(ledger_text)
    if question_match is None or operation_match is None:
        raise ValueError(f"question/operation absent for {row['question_id']}")
    question = " ".join(question_match.group("question").split())
    operation = operation_match.group("operation")
    traced_operation = str(ledger_trace.get("operation") or "")
    if traced_operation and operation != traced_operation:
        raise ValueError(f"operation trace mismatch for {row['question_id']}")

    card = (
        "\n\nAggregation execution card:\n"
        f"Operation: {operation}\n"
        f"Procedure: {_rule(operation)}\n"
        "Use graph adjacency only to find related evidence; sharing a graph block "
        "does not by itself make a memory an operand.\n"
        f"Compute and answer this exact Question now: {question}"
    )
    # Keep the short output contract, then finish with the operation and query.
    messages[-1]["content"] = user[:ledger_at] + user[output_at:] + card
    system = str(messages[0].get("content") or "")
    if AGGREGATION_LEDGER_APPENDIX not in system:
        raise ValueError(f"aggregation system appendix absent for {row['question_id']}")
    messages[0]["content"] = system.replace(
        AGGREGATION_LEDGER_APPENDIX, COMPACT_SYSTEM_APPENDIX, 1)

    source_tokens = int(row.get("packing_prompt_tokens") or 0)
    prompt_tokens = sum(counter.count(message["content"]) for message in messages)
    if prompt_tokens > source_tokens:
        raise ValueError(
            f"{row['question_id']}: compact aggregation increased "
            f"{source_tokens} -> {prompt_tokens}")
    prompt_version = "+".join(filter(None, (
        str(trace.get("prompt_version") or ""), VERSION)))
    trace.update({
        "prompt_version": prompt_version,
        "aggregation_execution_card": True,
        "aggregation_execution_operation": operation,
        "aggregation_ledger_candidates_rendered": 0,
        "aggregation_ledger_candidates_available": len(
            ledger_trace.get("candidate_turn_ids") or ()),
        "aggregation_execution_source_payload_hash": row.get(
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
    return output, prompt_tokens - source_tokens, operation


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
    deltas: list[int] = []
    operations: Counter[str] = Counter()
    for row in rows:
        output, delta, operation = transform(row, counter)
        if set(output.get("evidence_turn_ids") or ()) != set(
                row.get("evidence_turn_ids") or ()):
            raise ValueError(f"{row['question_id']}: evidence set changed")
        if output.get("evidence_turn_ids") != row.get("evidence_turn_ids"):
            raise ValueError(f"{row['question_id']}: evidence order changed")
        transformed.append(output)
        deltas.append(delta)
        operations[operation] += 1

    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in transformed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    changed = sum(
        before.get("prompt_payload_hash") != after.get("prompt_payload_hash")
        for before, after in zip(rows, transformed))
    manifest = {
        "schema_version": "graphmem-v5.46-compact-aggregation-arm-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(rows), "changed": changed,
        "operations": dict(sorted(operations.items())),
        "evidence_set_and_order_frozen": True,
        "benchmark_gold_or_judge_routing": False,
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
