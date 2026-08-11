#!/usr/bin/env python3
"""Route explicit modal questions to a grounded one-step inference readout."""
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


VERSION = "graphmem-v5.48-grounded-inference-synthesis-v1"
QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(?P<question>[^\n]+)")
INFERENCE_RE = re.compile(
    r"\b(?:would|might|likely|could|considered|personality traits|"
    r"underlying condition|attributes describe|based on)\b", re.I)
FOOTER_MARKER = "\n\nAnswer the original Question now:"
STRICT_ABSENCE = (
    "If the exact requested entity or relation was never mentioned, say so and "
    "name the near-match only when useful."
)
INFERENCE_SYSTEM = (
    "For modal inference questions, infer from stated facts and ordinary "
    "knowledge; the conclusion need not be verbatim."
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
    changed_ids: list[str] = []
    for row in source_rows:
        messages = [dict(message) for message in row.get("messages") or ()]
        trace = dict(row.get("trace") or {})
        if not messages:
            output_rows.append(dict(row)); deltas.append(0); continue
        user = str(messages[-1].get("content") or "")
        match = QUESTION_RE.search(user)
        question = " ".join(match.group("question").split()) if match else ""
        if (not question or not INFERENCE_RE.search(question)
                or trace.get("aggregation_ledger")
                or trace.get("preference_synthesis")):
            output_rows.append(dict(row)); deltas.append(0); continue
        footer_at = user.find(FOOTER_MARKER)
        if footer_at < 0:
            raise ValueError(f"question footer absent for {row['question_id']}")
        system = str(messages[0].get("content") or "")
        if STRICT_ABSENCE not in system:
            raise ValueError(f"strict absence clause absent for {row['question_id']}")
        messages[0]["content"] = system.replace(
            STRICT_ABSENCE, INFERENCE_SYSTEM, 1)
        messages[-1]["content"] = user[:footer_at] + (
            f"\n\nInference Question: {question}\n"
            "Infer one plausible answer from stated personal facts and ordinary "
            "knowledge. Implicit is allowed; invent no facts. Output one line."
        )
        source_tokens = int(row.get("packing_prompt_tokens") or 0)
        prompt_tokens = sum(counter.count(message["content"]) for message in messages)
        if prompt_tokens > source_tokens:
            raise ValueError(
                f"{row['question_id']}: inference prompt increased "
                f"{source_tokens} -> {prompt_tokens}")
        prompt_version = "+".join(filter(None, (
            str(trace.get("prompt_version") or ""), VERSION)))
        trace.update({
            "prompt_version": prompt_version,
            "inference_synthesis": True,
            "inference_synthesis_source_payload_hash": row.get(
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
            "trace": trace, "preparation_latency_ms": 0.0,
        })
        if output.get("evidence_turn_ids") != row.get("evidence_turn_ids"):
            raise ValueError(f"{row['question_id']}: evidence order changed")
        output_rows.append(output)
        deltas.append(prompt_tokens - source_tokens)
        changed_ids.append(str(row["question_id"]))

    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.48-inference-synthesis-arm-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(source_rows), "changed": len(changed_ids),
        "changed_question_ids": changed_ids,
        "routing_inputs": ["question wording", "existing specialized-path trace"],
        "benchmark_gold_or_judge_routing": False,
        "evidence_set_and_order_frozen": True,
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
