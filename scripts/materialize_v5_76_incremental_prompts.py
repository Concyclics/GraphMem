#!/usr/bin/env python3
"""Materialize label-free incremental prompts for expanded-budget questions.

The output contains only questions selected by the online budget controller.
It never reads reference answers, gold turns, or judge verdicts.  The full
expanded evidence pack is used solely to compute ``expanded - base``; it is
not copied into the continuation request.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.ensemble import question_from_messages  # noqa: E402
from graphmem.answer.incremental import (  # noqa: E402
    INCREMENTAL_PROMPT_VERSION, build_incremental_answer_messages,
    plan_incremental_evidence,
)
from graphmem.answer.rendering import AnswerConfig  # noqa: E402
from graphmem.domain import canonical_json  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    rows = read_rows(path)
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate {key} in {path}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: Iterable[int | float]) -> dict[str, int | float]:
    rows = sorted(values)

    def nearest(fraction: float) -> int | float:
        return rows[max(0, math.ceil(fraction * len(rows)) - 1)] if rows else 0

    return {
        "count": len(rows),
        "mean": sum(rows) / max(1, len(rows)),
        "p50": nearest(0.50), "p95": nearest(0.95),
        "p99": nearest(0.99), "max": max(rows, default=0),
        "sum": sum(rows), "percentile_method": "nearest_rank",
    }


def candidate_rows(row: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    values = list(row.get("candidates", ()))[:limit]
    if len(values) < limit:
        raise RuntimeError(
            f"{row.get('question_id')}: expected {limit} base candidates")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--base-prepared", type=Path, required=True)
    parser.add_argument("--expanded-prepared", type=Path, required=True)
    parser.add_argument("--base-retrieval", type=Path, required=True)
    parser.add_argument("--base-candidates", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-prefix", type=int, default=4)
    parser.add_argument("--max-anchors", type=int, default=4)
    parser.add_argument("--packing-model", default=(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"))
    parser.add_argument("--expected-all", type=int, default=1540)
    parser.add_argument("--expected-expanded", type=int, default=0)
    args = parser.parse_args()
    if args.base_prefix <= 0 or args.max_anchors < 0:
        raise ValueError("base-prefix must be positive and max-anchors non-negative")

    decisions_rows = read_rows(args.decisions)
    decisions = {str(row["question_id"]): row for row in decisions_rows}
    base = keyed(args.base_prepared)
    expanded = keyed(args.expanded_prepared)
    retrieval = keyed(args.base_retrieval, "dev_question_id")
    candidates = keyed(args.base_candidates)
    ids = set(decisions)
    for label, rows in (("base prepared", base), ("expanded prepared", expanded),
                        ("base retrieval", retrieval),
                        ("base candidates", candidates)):
        if set(rows) != ids:
            raise RuntimeError(
                f"{label} differs from decision IDs: {len(set(rows) ^ ids)}")
    if args.expected_all and len(ids) != args.expected_all:
        raise RuntimeError(f"expected {args.expected_all} decisions, got {len(ids)}")
    selected = [row for row in decisions_rows if bool(row.get("expand"))]
    if args.expected_expanded and len(selected) != args.expected_expanded:
        raise RuntimeError(
            f"expected {args.expected_expanded} expanded rows, got {len(selected)}")

    store = SQLiteGraphStore(args.source_db, read_only=True)
    counter = resolve_token_counter(args.packing_model)
    answer_config = AnswerConfig.v5_63()
    prepared_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    try:
        for decision in selected:
            question_id = str(decision["question_id"])
            base_row = base[question_id]
            expanded_row = expanded[question_id]
            retrieval_row = retrieval[question_id]
            plan = plan_incremental_evidence(
                base_row.get("evidence_turn_ids", ()),
                expanded_row.get("evidence_turn_ids", ()),
                preferred_anchor_ids=retrieval_row.get(
                    "typed_evidence_card_turn_ids", ()),
                max_anchors=args.max_anchors,
            )
            requested_ids = plan.transmitted_turn_ids
            turns = store.turns_by_ids(requested_ids)
            turn_map = {turn.turn_id: turn for turn in turns}
            missing = [turn_id for turn_id in requested_ids
                       if turn_id not in turn_map]
            if missing:
                raise RuntimeError(
                    f"{question_id}: source DB misses {len(missing)} turns")
            question = question_from_messages(base_row["messages"])
            base_choices = candidate_rows(
                candidates[question_id], args.base_prefix)
            messages = build_incremental_answer_messages(
                question=question,
                first_pass_candidates=base_choices,
                anchor_turns=tuple(turn_map[item]
                                   for item in plan.anchor_turn_ids),
                added_turns=tuple(turn_map[item]
                                  for item in plan.added_turn_ids),
                route=str(retrieval_row.get("adaptive_recall_route") or "lookup"),
                escalation_reasons=tuple(decision.get("reasons", ())),
                missing_obligation_count=len(retrieval_row.get(
                    "evidence_certificate_missing_slots", ())),
                answer_config=answer_config,
                proposal_limit=args.base_prefix,
            )
            prompt_tokens = sum(counter.count_many([
                str(message["content"]) for message in messages]))
            payload_hash = hashlib.sha256(
                canonical_json(list(messages)).encode()).hexdigest()
            prompt_hash = hashlib.sha256(
                (INCREMENTAL_PROMPT_VERSION + messages[0]["content"]).encode()
            ).hexdigest()
            trace = {
                "prompt_version": INCREMENTAL_PROMPT_VERSION,
                "adaptive_answer_budget": decision,
                "incremental_evidence": True,
                "source_base_prompt_payload_hash": base_row.get(
                    "prompt_payload_hash"),
                "source_expanded_prompt_payload_hash": expanded_row.get(
                    "prompt_payload_hash"),
                "base_evidence_turns": len(base_row.get("evidence_turn_ids", ())),
                "expanded_evidence_turns": len(expanded_row.get(
                    "evidence_turn_ids", ())),
                "added_turn_ids": list(plan.added_turn_ids),
                "anchor_turn_ids": list(plan.anchor_turn_ids),
                "removed_turn_ids": list(plan.removed_turn_ids),
                "retained_turn_ids": list(plan.retained_turn_ids),
                "token_counter": counter.describe(),
            }
            prepared_rows.append({
                "question_id": question_id,
                "memory_id": base_row.get("memory_id"),
                "messages": list(messages),
                "evidence_turn_ids": list(requested_ids),
                "dropped_turn_ids": list(plan.removed_turn_ids),
                "evidence_tokens": sum(counter.count_many([
                    turn_map[item].raw_text for item in requested_ids])),
                "packing_prompt_tokens": prompt_tokens,
                "closed_form": False, "draft_text": "",
                "draft_certified": False, "budget_relaxed": False,
                "prompt_hash": prompt_hash,
                "prompt_payload_hash": payload_hash,
                "warnings": ([] if counter.exact else [
                    "materialization token count is heuristic; use API usage "
                    "for reported cost"]),
                "preparation_latency_ms": 0.0,
                "trace": trace,
            })
            audit_rows.append({
                "question_id": question_id,
                "memory_id": base_row.get("memory_id"),
                "route": retrieval_row.get("adaptive_recall_route"),
                "reasons": decision.get("reasons", []),
                "base_turns": len(base_row.get("evidence_turn_ids", ())),
                "expanded_turns": len(expanded_row.get("evidence_turn_ids", ())),
                "added_turns": len(plan.added_turn_ids),
                "anchor_turns": len(plan.anchor_turn_ids),
                "transmitted_turns": len(requested_ids),
                "retained_turns": len(plan.retained_turn_ids),
                "removed_turns": len(plan.removed_turn_ids),
                "materialized_prompt_tokens": prompt_tokens,
                "full_expanded_prompt_tokens": int(
                    expanded_row.get("packing_prompt_tokens", 0)),
            })
    finally:
        store.close()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("prepared_answers.jsonl", prepared_rows),
                       ("audit.jsonl", audit_rows)):
        (args.output_root / name).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
    manifest = {
        "schema_version": INCREMENTAL_PROMPT_VERSION,
        "questions": len(prepared_rows),
        "uses_gold_or_judge": False,
        "base_prefix": args.base_prefix,
        "max_anchors": args.max_anchors,
        "token_counter": counter.describe(),
        "turns": {
            key: stats(int(row[key]) for row in audit_rows)
            for key in ("base_turns", "expanded_turns", "added_turns",
                        "anchor_turns", "transmitted_turns", "retained_turns",
                        "removed_turns")
        },
        "materialized_prompt_tokens": stats(
            int(row["materialized_prompt_tokens"]) for row in audit_rows),
        "full_expanded_prompt_tokens": stats(
            int(row["full_expanded_prompt_tokens"]) for row in audit_rows),
        "inputs": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in (
                ("decisions", args.decisions),
                ("base_prepared", args.base_prepared),
                ("expanded_prepared", args.expanded_prepared),
                ("base_retrieval", args.base_retrieval),
                ("base_candidates", args.base_candidates),
                ("source_db", args.source_db),
            )
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
