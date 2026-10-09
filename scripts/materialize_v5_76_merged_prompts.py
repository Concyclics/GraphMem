#!/usr/bin/env python3
"""Materialize one-call adaptive prompts for retrieval-gated questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.ensemble import question_from_messages  # noqa: E402
from graphmem.answer.incremental import (  # noqa: E402
    MERGED_EXPANSION_PROMPT_VERSION, build_merged_expansion_messages,
    plan_incremental_evidence,
)
from graphmem.answer.rendering import AnswerConfig  # noqa: E402
from graphmem.domain import canonical_json  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    values = read_rows(path)
    result = {str(row[key]): row for row in values}
    if len(result) != len(values):
        raise RuntimeError(f"duplicate {key} in {path}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: Iterable[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)

    def nearest(fraction: float) -> int | float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] \
            if ordered else 0

    return {
        "count": len(ordered), "mean": sum(ordered) / max(1, len(ordered)),
        "p50": nearest(0.50), "p95": nearest(0.95),
        "p99": nearest(0.99), "max": max(ordered, default=0),
        "sum": sum(ordered), "percentile_method": "nearest_rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--base-prepared", type=Path, required=True)
    parser.add_argument("--expanded-prepared", type=Path, required=True)
    parser.add_argument("--base-retrieval", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--packing-model", default=(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"))
    parser.add_argument("--expected-all", type=int, default=1540)
    parser.add_argument("--expected-expanded", type=int, default=0)
    parser.add_argument(
        "--exclude-already-triggered", action="store_true",
        help=("Materialize only post-gate expansions; questions already "
              "expanded inside the base retrieval keep their one-call prompt."))
    parser.add_argument(
        "--only-already-triggered", action="store_true",
        help="Materialize only questions expanded by the intrinsic retrieval gate.")
    args = parser.parse_args()
    if args.exclude_already_triggered and args.only_already_triggered:
        raise ValueError("triggered filters are mutually exclusive")

    decision_rows = read_rows(args.decisions)
    decisions = {str(row["question_id"]): row for row in decision_rows}
    base = keyed(args.base_prepared)
    expanded = keyed(args.expanded_prepared)
    retrieval = keyed(args.base_retrieval, "dev_question_id")
    ids = set(decisions)
    for label, values in (("base", base), ("expanded", expanded),
                          ("retrieval", retrieval)):
        if set(values) != ids:
            raise RuntimeError(f"{label} question IDs differ from decisions")
    if args.expected_all and len(ids) != args.expected_all:
        raise RuntimeError(f"expected {args.expected_all} questions, got {len(ids)}")
    selected = []
    for row in decision_rows:
        if not bool(row.get("expand")):
            continue
        triggered = bool(retrieval[str(row["question_id"])].get(
            "adaptive_recall_triggered", False))
        if args.exclude_already_triggered and triggered:
            continue
        if args.only_already_triggered and not triggered:
            continue
        selected.append(row)
    if args.expected_expanded and len(selected) != args.expected_expanded:
        raise RuntimeError(
            f"expected {args.expected_expanded} expanded rows, got {len(selected)}")

    store = SQLiteGraphStore(args.source_db, read_only=True)
    counter = resolve_token_counter(args.packing_model)
    config = AnswerConfig.v5_63()
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    try:
        for decision in selected:
            question_id = str(decision["question_id"])
            base_row = base[question_id]
            expanded_row = expanded[question_id]
            retrieval_row = retrieval[question_id]
            plan = plan_incremental_evidence(
                base_row.get("evidence_turn_ids", ()),
                expanded_row.get("evidence_turn_ids", ()), max_anchors=0)
            turns = store.turns_by_ids(plan.added_turn_ids)
            if len(turns) != len(plan.added_turn_ids):
                raise RuntimeError(f"{question_id}: expanded source turns missing")
            question = question_from_messages(base_row["messages"])
            messages = build_merged_expansion_messages(
                base_messages=base_row["messages"], question=question,
                added_turns=turns,
                route=str(retrieval_row.get("adaptive_recall_route") or "lookup"),
                escalation_reasons=tuple(decision.get("reasons", ())),
                missing_obligation_count=len(retrieval_row.get(
                    "evidence_certificate_missing_slots", ())),
                answer_config=config,
            )
            prompt_tokens = sum(counter.count_many([
                str(message["content"]) for message in messages]))
            payload_hash = hashlib.sha256(
                canonical_json(list(messages)).encode()).hexdigest()
            row = dict(base_row)
            trace = dict(row.get("trace", {}))
            trace.update({
                "prompt_version": MERGED_EXPANSION_PROMPT_VERSION,
                "adaptive_answer_budget": decision,
                "merged_expansion": True,
                "source_base_prompt_payload_hash": base_row.get(
                    "prompt_payload_hash"),
                "source_expanded_prompt_payload_hash": expanded_row.get(
                    "prompt_payload_hash"),
                "added_turn_ids": list(plan.added_turn_ids),
                "retained_turn_ids": list(plan.retained_turn_ids),
                "removed_turn_ids": list(plan.removed_turn_ids),
                "token_counter": counter.describe(),
            })
            evidence_ids = tuple(dict.fromkeys((
                *base_row.get("evidence_turn_ids", ()), *plan.added_turn_ids)))
            row.update({
                "messages": list(messages),
                "evidence_turn_ids": list(evidence_ids),
                "packing_prompt_tokens": prompt_tokens,
                "prompt_hash": hashlib.sha256((
                    MERGED_EXPANSION_PROMPT_VERSION
                    + messages[0]["content"]).encode()).hexdigest(),
                "prompt_payload_hash": payload_hash,
                "trace": trace,
            })
            output.append(row)
            audit.append({
                "question_id": question_id,
                "base_turns": len(base_row.get("evidence_turn_ids", ())),
                "expanded_turns": len(expanded_row.get("evidence_turn_ids", ())),
                "added_turns": len(plan.added_turn_ids),
                "merged_unique_turns": len(evidence_ids),
                "base_prompt_tokens": int(base_row.get(
                    "packing_prompt_tokens", 0)),
                "fixed64_prompt_tokens": int(expanded_row.get(
                    "packing_prompt_tokens", 0)),
                "merged_prompt_tokens": prompt_tokens,
            })
    finally:
        store.close()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, values in (("prepared_answers.jsonl", output),
                         ("audit.jsonl", audit)):
        (args.output_root / name).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in values),
            encoding="utf-8")
    manifest = {
        "schema_version": MERGED_EXPANSION_PROMPT_VERSION,
        "questions": len(output), "uses_gold_or_judge": False,
        "model_calls_per_question": 1,
        "exclude_already_triggered": args.exclude_already_triggered,
        "only_already_triggered": args.only_already_triggered,
        "token_counter": counter.describe(),
        "turns": {key: stats(int(row[key]) for row in audit)
                  for key in ("base_turns", "expanded_turns", "added_turns",
                              "merged_unique_turns")},
        "prompt_tokens": {key: stats(int(row[key]) for row in audit)
                          for key in ("base_prompt_tokens",
                                      "fixed64_prompt_tokens",
                                      "merged_prompt_tokens")},
        "inputs": {name: {"path": str(path), "sha256": digest(path)}
                   for name, path in (
                       ("decisions", args.decisions),
                       ("base_prepared", args.base_prepared),
                       ("expanded_prepared", args.expanded_prepared),
                       ("base_retrieval", args.base_retrieval),
                       ("source_db", args.source_db))},
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
