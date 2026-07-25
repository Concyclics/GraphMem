#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.hierarchical_v2 import build_evidence_ledger, query_kind
from graphmem_demo.models import AtomicFactNode, LeafNode, StateChain


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay deterministic V2 ledger operators from saved nodes.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge", type=Path)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument(
        "--include-answer", action="store_true",
        help="Include gold answers in output (off by default to preserve blind-test isolation).",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    cases = {row["question_id"]: row for row in json.loads(args.data.read_text(encoding="utf-8"))}
    requested = set(args.question_id)
    if args.judge:
        requested.update(
            row["question_id"] for row in read_jsonl(args.judge)
            if not row.get("correct", False)
        )
    if not requested:
        requested = set(cases)

    leaves: dict[str, list[LeafNode]] = {qid: [] for qid in requested}
    facts: dict[str, list[AtomicFactNode]] = {qid: [] for qid in requested}
    for row in read_jsonl(args.run_dir / "nodes.jsonl"):
        qid = row.get("question_id")
        if qid not in requested:
            continue
        node_type = row.pop("node_type", "")
        if node_type == "leaf":
            leaves[qid].append(LeafNode(**row))
        elif node_type == "atomic_fact":
            facts[qid].append(AtomicFactNode(**row))

    chains: dict[str, list[StateChain]] = {qid: [] for qid in requested}
    chain_path = args.run_dir / "state_chains.jsonl"
    if chain_path.exists():
        for row in read_jsonl(chain_path):
            qid = row.get("question_id")
            if qid in requested:
                row.pop("node_type", None)
                chains[qid].append(StateChain(**row))

    for qid in sorted(requested):
        case = cases[qid]
        kind = query_kind(case["question"])
        ledger = build_evidence_ledger(
            kind,
            facts[qid],
            chains[qid],
            leaves[qid],
            case["question"],
            case.get("question_date"),
            operator_facts=facts[qid],
            operator_leaves=leaves[qid],
            complete_facts=facts[qid],
            complete_leaves=leaves[qid],
        )
        operators = [row for row in ledger if "operator" in row]
        if args.compact:
            operators=[{
                "operator":row["operator"],
                "result":row.get("result"),
                "candidate_pool_complete":row.get("candidate_pool_complete"),
            } for row in operators if row["operator"]!="event_order"]
        output={
            "question_id": qid,
            "query_kind": kind,
            "question": case["question"],
            "operators": operators,
        }
        if args.include_answer:
            output["answer"]=case.get("answer")
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
