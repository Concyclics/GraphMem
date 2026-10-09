#!/usr/bin/env python3
"""Overlay gated semantic witnesses on a frozen prepared evidence ordering.

The first ``protected-turns`` source turns remain byte-for-byte stable.  Fact
and predicate witnesses selected by the online navigator are inserted after
that floor; duplicates and witnesses rejected by the navigator's final token
pack are ignored.  This permits a paired readout experiment without replacing
the validated base retrieval/controller policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protected-turns", type=int, default=32)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()
    if args.protected_turns < 0:
        raise ValueError("protected-turns must be non-negative")

    prepared_rows = read_jsonl(args.prepared)
    retrieval_rows = read_jsonl(args.retrieval)
    prepared = {str(row["question_id"]): row for row in prepared_rows}
    retrieval = {str(row["dev_question_id"]): row for row in retrieval_rows}
    if args.expected and len(prepared) != args.expected:
        raise RuntimeError(
            f"expected {args.expected} prepared rows, got {len(prepared)}")
    if len(prepared) != len(prepared_rows):
        raise RuntimeError("duplicate prepared question IDs")
    if len(retrieval) != len(retrieval_rows):
        raise RuntimeError("duplicate retrieval question IDs")
    if set(prepared) != set(retrieval):
        raise RuntimeError("prepared/retrieval question IDs do not match")

    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    changed = 0
    inserted_fact = 0
    inserted_predicate = 0
    for source in prepared_rows:
        question_id = str(source["question_id"])
        route = retrieval[question_id]
        base = tuple(dict.fromkeys(map(
            str, source.get("evidence_turn_ids", ()))))
        finally_packed = frozenset(map(
            str, route.get("retrieved_turn_ids", ())))
        fact = tuple(
            str(turn_id)
            for turn_id in route.get("semantic_fact_witness_turn_ids", ())
            if str(turn_id) in finally_packed)
        predicate = tuple(
            str(turn_id)
            for turn_id in route.get(
                "semantic_predicate_witness_turn_ids", ())
            if str(turn_id) in finally_packed)
        protected = base[:args.protected_turns]
        protected_set = frozenset(protected)
        witnesses = tuple(dict.fromkeys((
            *(turn_id for turn_id in fact if turn_id not in protected_set),
            *(turn_id for turn_id in predicate if turn_id not in protected_set),
        )))
        witness_set = frozenset(witnesses)
        tail = tuple(
            turn_id for turn_id in base[args.protected_turns:]
            if turn_id not in witness_set)
        selected = (*protected, *witnesses, *tail)
        row = dict(source)
        trace = dict(source.get("trace", {}))
        trace["semantic_witness_overlay"] = {
            "protected_turns": len(protected),
            "fact_witnesses": len(tuple(
                turn_id for turn_id in witnesses if turn_id in fact)),
            "predicate_witnesses": len(tuple(
                turn_id for turn_id in witnesses if turn_id in predicate)),
            "inserted": len(witnesses),
            "base_turns": len(base),
            "output_turns": len(selected),
        }
        row.update({
            "evidence_turn_ids": list(selected),
            "trace": trace,
        })
        output.append(row)
        changed += selected != base
        inserted_fact += len(tuple(
            turn_id for turn_id in witnesses if turn_id in fact))
        inserted_predicate += len(tuple(
            turn_id for turn_id in witnesses if turn_id in predicate))
        audit.append({
            "question_id": question_id,
            "changed": selected != base,
            "base_turns": len(base),
            "output_turns": len(selected),
            "inserted_turn_ids": list(witnesses),
            "fact_turn_ids": list(fact),
            "predicate_turn_ids": list(predicate),
        })

    args.output_root.mkdir(parents=True, exist_ok=False)
    output_path = args.output_root / "prepared_answers.jsonl"
    audit_path = args.output_root / "audit.jsonl"
    output_path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8")
    audit_path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in audit),
        encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.78-semantic-overlay-v1",
        "questions": len(output),
        "protected_turns": args.protected_turns,
        "changed_questions": changed,
        "inserted_fact_witnesses": inserted_fact,
        "inserted_predicate_witnesses": inserted_predicate,
        "uses_gold_or_judge": False,
        "inputs": {
            "prepared": {
                "path": str(args.prepared), "sha256": digest(args.prepared)},
            "retrieval": {
                "path": str(args.retrieval), "sha256": digest(args.retrieval)},
        },
        "output": {"path": str(output_path), "sha256": digest(output_path)},
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
