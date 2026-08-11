#!/usr/bin/env python3
"""Compare navigation gold coverage with the evidence that reaches the model."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def _index(path: Path, key: str) -> dict[str, dict]:
    return {str(row[key]): row for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", choices=("longmemeval", "locomo"))
    args = parser.parse_args()

    prepared = _index(args.prepared, "question_id")
    retrieval = _index(args.retrieval, "dev_question_id")
    questions = load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))
    store = SQLiteGraphStore(args.source_db, read_only=True)
    totals: dict[str, Counter] = defaultdict(Counter)
    losses: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        if args.benchmark and question.benchmark != args.benchmark:
            continue
        if (question.question_id not in prepared
                or question.question_id not in retrieval):
            continue
        if not question.has_turn_gold:
            continue
        turns = {(row.session_id, row.turn_index): row.turn_id
                 for row in store.turns(question.memory_id)}
        gold = {turns[(ref.session_id, ref.turn_index)] for ref in question.gold_turns
                if (ref.session_id, ref.turn_index) in turns}
        prompt = set(prepared[question.question_id]["evidence_turn_ids"])
        nav = retrieval[question.question_id]
        ledger_ids = set((nav.get("aggregation_ledger") or {}).get(
            "candidate_turn_ids", ()))
        bucket = totals[question.benchmark]
        bucket["questions"] += 1
        bucket["navigation_all_hit"] += bool(nav["turn_all_hit"])
        bucket["prompt_all_hit"] += gold <= prompt
        bucket["navigation_to_prompt_all_hit_loss"] += (
            bool(nav["turn_all_hit"]) and not gold <= prompt)
        bucket["prompt_any_hit"] += bool(gold & prompt)
        bucket["gold_turns"] += len(gold)
        bucket["prompt_gold_hits"] += len(gold & prompt)
        bucket["navigation_turns"] += int(nav["packed_turns"])
        bucket["prompt_turns"] += len(prompt)
        bucket["aggregation_routed"] += bool(nav.get("aggregation_ledger"))
        bucket["aggregation_gold_turns"] += (
            len(gold) if ledger_ids else 0)
        bucket["aggregation_gold_hits"] += len(gold & ledger_ids)
        bucket["aggregation_gold_all_hit"] += bool(
            ledger_ids and gold <= ledger_ids)
        bucket["aggregation_routed_all_hit_loss"] += (
            bool(nav.get("aggregation_ledger")) and bool(nav["turn_all_hit"])
            and not gold <= prompt)
        if bool(nav["turn_all_hit"]) and not gold <= prompt:
            turn_rows = {row.turn_id: row for row in store.turns(question.memory_id)}
            missing = sorted(gold - prompt)
            losses[question.benchmark].append({
                "question_id": question.question_id,
                "stratum": question.stratum,
                "question": question.query,
                "gold_answer": question.raw.get("answer", ""),
                "missing_gold_turn_ids": missing,
                "missing_gold_text": [turn_rows[turn_id].raw_text for turn_id in missing],
                "missing_in_ledger_candidates": [
                    turn_id for turn_id in missing if turn_id in ledger_ids],
                "ledger_candidate_count": len(ledger_ids),
                "prompt_turn_count": len(prompt),
                "aggregation_operation": (
                    (nav.get("aggregation_ledger") or {}).get("operation")),
            })
    store.close()
    result = {}
    for benchmark, counter in totals.items():
        result[benchmark] = dict(counter)
        result[benchmark]["navigation_all_hit_rate"] = (
            counter["navigation_all_hit"] / counter["questions"])
        result[benchmark]["prompt_all_hit_rate"] = (
            counter["prompt_all_hit"] / counter["questions"])
        result[benchmark]["prompt_gold_recall"] = (
            counter["prompt_gold_hits"] / counter["gold_turns"])
        result[benchmark]["mean_navigation_turns"] = (
            counter["navigation_turns"] / counter["questions"])
        result[benchmark]["mean_prompt_turns"] = (
            counter["prompt_turns"] / counter["questions"])
        result[benchmark]["aggregation_gold_recall"] = (
            counter["aggregation_gold_hits"]
            / max(1, counter["aggregation_gold_turns"]))
        result[benchmark]["navigation_to_prompt_losses"] = losses[benchmark]
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
