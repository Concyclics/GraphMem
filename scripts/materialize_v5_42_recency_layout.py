#!/usr/bin/env python3
"""Move the strongest frozen topology blocks nearest the answer boundary.

This is a prompt-only, budget-nonincreasing arm.  It changes neither the
evidence set nor the order *inside* a graph/chain/auxiliary block.  The source
V5.40 layout is strongest-first; reversing whole blocks tests whether placing
the best evidence next to the final question improves long-context readout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import canonical_json  # noqa: E402
from graphmem.tokenization import ExactTokenCounter  # noqa: E402


VERSION = "graphmem-v5.42-topological-recency-readout-v1"
MEMORY_MARKER = "Conversation memories:\n"
END_MARKERS = (
    "\n\nAggregation ledger (",
    "\n\nOutput contract:",
    "\n\nFinal check:",
    "\n\nAnswer the original Question now:",
)
LABEL_RE = re.compile(
    r"^\[(?:CHAIN (?P<chain>\d+) (?:step=\d+|support)|"
    r"GRAPH (?P<graph>\d+) step=\d+|"
    r"AUX (?P<aux>\d+) rank=\d+)\]\s",
    re.M,
)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _nearest(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0


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


def _split_evidence(user: str) -> tuple[str, str, str]:
    start = user.find(MEMORY_MARKER)
    if start < 0:
        raise ValueError("conversation memory marker is absent")
    evidence_start = start + len(MEMORY_MARKER)
    stops = [position for marker in END_MARKERS
             if (position := user.find(marker, evidence_start)) >= 0]
    evidence_end = min(stops) if stops else len(user)
    return user[:evidence_start], user[evidence_start:evidence_end], user[evidence_end:]


def _chunks(evidence: str) -> list[tuple[str, str]]:
    matches = list(LABEL_RE.finditer(evidence))
    if not matches or matches[0].start() != 0:
        raise ValueError("topological evidence does not begin with a known label")
    rows: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        key = (
            f"chain:{match.group('chain')}" if match.group("chain") else
            f"graph:{match.group('graph')}" if match.group("graph") else
            f"aux:{match.group('aux')}"
        )
        rows.append((key, evidence[match.start():end]))
    return rows


def _reverse_blocks(evidence: str) -> tuple[str, list[int], int]:
    rows = _chunks(evidence)
    blocks: list[list[tuple[int, str, str]]] = []
    for index, (key, text) in enumerate(rows):
        if not blocks or blocks[-1][-1][1] != key:
            blocks.append([])
        blocks[-1].append((index, key, text))
    reversed_rows = [row for block in reversed(blocks) for row in block]
    return "".join(row[2] for row in reversed_rows), [row[0] for row in reversed_rows], len(blocks)


def transform(row: dict, counter: ExactTokenCounter) -> tuple[dict, int, int]:
    output = dict(row)
    messages = [dict(message) for message in row.get("messages", ())]
    if not messages:
        return output, 0, 0
    trace = dict(row.get("trace") or {})
    if trace.get("aggregation_ledger") or trace.get("preference_synthesis"):
        return output, 0, 0
    if len(messages) != 2 or messages[1].get("role") != "user":
        raise ValueError(f"unexpected messages for {row['question_id']}")

    prefix, evidence, suffix = _split_evidence(str(messages[1]["content"]))
    reordered, permutation, block_count = _reverse_blocks(evidence)
    evidence_ids = list(row.get("evidence_turn_ids") or ())
    if len(permutation) != len(evidence_ids):
        raise ValueError(
            f"{row['question_id']}: {len(permutation)} rendered turns != "
            f"{len(evidence_ids)} evidence ids")
    if permutation == list(range(len(permutation))):
        return output, 0, block_count

    system = str(messages[0].get("content") or "")
    old = "Blocks are ranked by relevance, then graph or time order."
    new = "Blocks run weakest-to-strongest; keep internal order."
    if old not in system:
        raise ValueError(f"compact topology contract absent for {row['question_id']}")
    messages[0]["content"] = system.replace(old, new, 1)
    messages[1]["content"] = prefix + reordered + suffix
    source_tokens = int(row.get("packing_prompt_tokens") or 0)
    prompt_tokens = sum(counter.count(message["content"]) for message in messages)
    if prompt_tokens > source_tokens:
        raise ValueError(
            f"{row['question_id']}: prompt tokens increased "
            f"{source_tokens} -> {prompt_tokens}")

    source_version = str(trace.get("prompt_version") or "")
    prompt_version = source_version + "+" + VERSION
    prompt_hash = hashlib.sha256(
        (prompt_version + messages[0]["content"]).encode()).hexdigest()
    trace.update({
        "prompt_version": prompt_version,
        "evidence_order": "topological_recency",
        "resolved_evidence_order": "topological_recency",
        "recency_layout_source_payload_hash": row.get("prompt_payload_hash"),
        "recency_layout_blocks": block_count,
    })
    output.update({
        "messages": messages,
        "evidence_turn_ids": [evidence_ids[index] for index in permutation],
        "packing_prompt_tokens": prompt_tokens,
        "prompt_hash": prompt_hash,
        "prompt_payload_hash": hashlib.sha256(
            canonical_json(messages).encode()).hexdigest(),
        "trace": trace,
        "preparation_latency_ms": 0.0,
    })
    return output, prompt_tokens - source_tokens, block_count


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
    blocks: list[int] = []
    for row in rows:
        output, delta, block_count = transform(row, counter)
        transformed.append(output)
        deltas.append(delta)
        if block_count:
            blocks.append(block_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in transformed)
    args.output.write_text(payload, encoding="utf-8")
    changed = [before.get("question_id") for before, after in zip(rows, transformed)
               if before.get("prompt_payload_hash") != after.get("prompt_payload_hash")]
    manifest = {
        "schema_version": "graphmem-v5.42-recency-layout-arm-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(rows),
        "changed": len(changed),
        "changed_question_ids": changed,
        "aggregation_and_preference_frozen": True,
        "evidence_set_frozen": all(
            set(before.get("evidence_turn_ids") or ())
            == set(after.get("evidence_turn_ids") or ())
            for before, after in zip(rows, transformed)),
        "packing_token_delta": _stats(deltas),
        "increased_questions": sum(delta > 0 for delta in deltas),
        "layout_blocks": _stats(blocks),
        "tokenizer": counter.describe(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
