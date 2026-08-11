#!/usr/bin/env python3
"""Compose the validated V5.49 policy from byte-identical full-run arms.

Routing depends only on speaker form, mechanically classified aggregation
operation, and explicit modal/counterfactual wording.  It never reads dataset
labels, expected answers, gold evidence, predictions, or judge outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from materialize_v5_43_typed_readout import _named_multi_party, _split_evidence
from materialize_v5_48_inference_synthesis import INFERENCE_RE, QUESTION_RE


COUNTERFACTUAL_RE = re.compile(
    r"\bif\b.*\b(?:hadn't|had\s+not|didn't|did\s+not|weren't|were\s+not|"
    r"wouldn't|would\s+not)\b", re.I)
ANONYMOUS_COMPACT_OPERATIONS = frozenset({
    "date_difference", "difference", "mean",
})


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _index(path: Path) -> tuple[list[str], dict[str, dict]]:
    rows = _read(path)
    ids = [str(row["question_id"]) for row in rows]
    result = {str(row["question_id"]): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"duplicate question ID in {path}")
    return ids, result


def _question(row: dict) -> str:
    messages = row.get("messages") or ()
    if not messages:
        return ""
    match = QUESTION_RE.search(str(messages[-1].get("content") or ""))
    return " ".join(match.group("question").split()) if match else ""


def _named(row: dict) -> bool:
    messages = row.get("messages") or ()
    if not messages:
        return False
    _prefix, evidence, _suffix = _split_evidence(
        str(messages[-1].get("content") or ""))
    return _named_multi_party(evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--single-line", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    order, base = _index(args.base)
    sources = {"base": base}
    for name, path in (("compact", args.compact),
                       ("single_line", args.single_line),
                       ("inference", args.inference)):
        ids, rows = _index(path)
        if ids != order:
            raise ValueError(f"{name} question order differs from base")
        sources[name] = rows

    output_rows: list[dict] = []
    routes: Counter[str] = Counter()
    token_deltas: list[int] = []
    for question_id in order:
        row = base[question_id]
        trace = row.get("trace") or {}
        operation = str((trace.get("aggregation_ledger") or {}).get(
            "operation") or "")
        question = _question(row)
        named = _named(row) if row.get("messages") else False
        if (not operation and question and INFERENCE_RE.search(question)
                and not COUNTERFACTUAL_RE.search(question)
                and not trace.get("preference_synthesis")):
            route = "inference"
        elif operation == "date_difference" and named:
            route = "single_line"
        elif operation in ANONYMOUS_COMPACT_OPERATIONS and not named:
            route = "compact"
        else:
            route = "base"
        chosen = dict(sources[route][question_id])
        if set(chosen.get("evidence_turn_ids") or ()) != set(
                row.get("evidence_turn_ids") or ()):
            raise ValueError(f"{question_id}: evidence set changed in {route}")
        # Prompt-only sources may deliberately reverse whole graph blocks, but
        # aggregation and inference arms used here preserve the V5.45 order.
        if chosen.get("evidence_turn_ids") != row.get("evidence_turn_ids"):
            raise ValueError(f"{question_id}: evidence order changed in {route}")
        output_rows.append(chosen)
        routes[route] += 1
        token_deltas.append(
            int(chosen.get("packing_prompt_tokens") or 0)
            - int(row.get("packing_prompt_tokens") or 0))

    if any(delta > 0 for delta in token_deltas):
        raise ValueError("routed policy increased a packing prompt")
    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.49-routed-policy-v1",
        "sources": {
            name: {"path": str(path),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("base", args.base), ("compact", args.compact),
                               ("single_line", args.single_line),
                               ("inference", args.inference))
        },
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(order), "routes": dict(sorted(routes.items())),
        "routing_inputs": [
            "named-versus-anonymous speaker form",
            "mechanical aggregation operation",
            "modal and counterfactual question wording",
            "existing specialized-path trace",
        ],
        "benchmark_gold_prediction_or_judge_routing": False,
        "evidence_set_and_order_frozen": True,
        "packing_token_delta": {
            "count": len(token_deltas),
            "mean": sum(token_deltas) / max(1, len(token_deltas)),
            "min": min(token_deltas, default=0),
            "max": max(token_deltas, default=0),
            "increased_questions": sum(delta > 0 for delta in token_deltas),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
