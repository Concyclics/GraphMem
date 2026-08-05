#!/usr/bin/env python3
"""Per-question proof funnel and packing oracles for the V5.6 harness.

Aggregate gaps cannot say *where* evidence is lost.  This walks each gold turn
through every stage it must survive, and adds two oracles that bound what the
query side could achieve without touching the graph:

  R0 gold_in_reservoir          the turn is reachable at all
  R1 gold_in_candidates         it survived into the scored candidate pool
  R2 gold_has_evidence_group    an EvidenceGroup resolves to it
  R3 gold_has_canonical_fact    a CanonicalFact cites that group
  R3b gold_fact_reached         that fact was actually reached as a graph node
  R4 gold_has_operand_binding   some operand actually bound that fact
  R5 gold_selected_by_algebra   the algebra kept the binding
  R6 gold_in_proof_unit         it became mandatory evidence
  R7 gold_packed                it reached the final pack

  budget_oracle    could every gold turn fit in the turn/token budget at all?
  binding_oracle   best packed all-hit reachable using only existing bindings

Read-only: it re-runs navigation but writes nothing back to the graph.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from graphmem.config import load_config
from graphmem.domain import QueryBudget
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval import load_dev_questions, load_gold_turns
from graphmem.retrieval import GraphNavigator, HarnessProfile
from graphmem.storage import SQLiteGraphStore
from graphmem.tokenization import resolve_token_counter

BACKBONE = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
STAGES = ("gold_in_reservoir", "gold_in_candidates", "gold_has_evidence_group",
          "gold_has_canonical_fact", "gold_fact_reached", "gold_has_operand_binding",
          "gold_selected_by_algebra", "gold_in_proof_unit", "gold_packed")


def gold_turn_ids(store, question) -> set[str]:
    keys = {(row.session_id, row.turn_index) for row in question.gold_turns}
    return {turn.turn_id for turn in store.turns(question.memory_id)
            if (turn.session_id, turn.turn_index) in keys}


def _turn_to_groups(store, memory_id: str) -> dict[str, set[str]]:
    """turn_id -> evidence groups that cite it."""
    rows: dict[str, set[str]] = {}
    for group in store.evidence_groups(memory_id):
        for member in group.members:
            rows.setdefault(member.turn_id, set()).add(group.evidence_group_id)
    return rows


def _turn_to_facts(store, memory_id: str, turn_groups: dict[str, set[str]]) -> dict[str, set[str]]:
    """turn_id -> CanonicalFact nodes whose provenance includes that turn."""
    group_to_facts: dict[str, set[str]] = {}
    for node in store.nodes(memory_id):
        if node.node_type.value != "canonical_fact":
            continue
        for group_id in node.all_evidence_group_ids:
            group_to_facts.setdefault(group_id, set()).add(node.node_id)
    rows: dict[str, set[str]] = {}
    for turn_id, groups in turn_groups.items():
        facts: set[str] = set()
        for group_id in groups:
            facts |= group_to_facts.get(group_id, set())
        if facts:
            rows[turn_id] = facts
    return rows


def funnel_row(store, question, result, budget, counter) -> dict:
    gold = gold_turn_ids(store, question)
    turns = {turn.turn_id: turn for turn in store.turns(question.memory_id)}
    turn_groups = _turn_to_groups(store, question.memory_id)
    turn_facts = _turn_to_facts(store, question.memory_id, turn_groups)

    reservoir = {row.turn_id for row in result.candidate_scores}
    reached_facts = set(result.reached_fact_node_ids)
    bound_facts = set(result.bound_fact_node_ids)
    selected_facts = set(result.selected_fact_node_ids)
    algebra_turns: set[str] = set()
    for unit in result.proof_units:
        algebra_turns |= set(unit.source_turn_ids)
    proof_turns = algebra_turns
    packed = set(result.packed_turn_ids)

    # Which gold turns clear each stage.
    stages = {
        "gold_in_reservoir": {t for t in gold if t in reservoir},
        "gold_in_candidates": {t for t in gold if t in reservoir},
        "gold_has_evidence_group": {t for t in gold if turn_groups.get(t)},
        "gold_has_canonical_fact": {t for t in gold if turn_facts.get(t)},
        # Separating these two says whether the fix is node retrieval or the
        # binding discriminant: a fact that was never reached cannot bind.
        "gold_fact_reached": {t for t in gold if turn_facts.get(t, set()) & reached_facts},
        "gold_has_operand_binding": {t for t in gold if turn_facts.get(t, set()) & bound_facts},
        "gold_selected_by_algebra": {t for t in gold if turn_facts.get(t, set()) & selected_facts},
        "gold_in_proof_unit": {t for t in gold if t in proof_turns},
        "gold_packed": {t for t in gold if t in packed},
    }

    gold_tokens = sum(counter.count(turns[t].raw_text) for t in gold if t in turns)
    budget_oracle = (len(gold) <= budget.max_evidence_turns
                     and gold_tokens <= budget.max_evidence_tokens)
    # Best case for the query side: every gold turn that has a binding *and* is
    # reachable could in principle be made mandatory.
    bindable = stages["gold_has_operand_binding"] & stages["gold_in_reservoir"]
    binding_oracle = bool(gold) and bindable >= gold and budget_oracle

    row = {
        "dev_question_id": question.question_id,
        "memory_id": question.memory_id,
        "stratum": question.stratum,
        "gold_turns": len(gold),
        "gold_tokens": gold_tokens,
        "budget_oracle_fits": budget_oracle,
        "binding_oracle_all_hit": binding_oracle,
        "operator": str(result.trace.get("query_operator", "")),
        "operator_ast": str(result.trace.get("operator_ast", "")),
        "ast_diverges": bool(result.trace.get("ast_diverges", False)),
        "packed_turns": len(packed),
        "reservoir_size": len(reservoir),
    }
    for stage, hits in stages.items():
        row[stage] = len(hits) == len(gold) and bool(gold)
        row[stage + "_recall"] = len(hits) / max(1, len(gold))
    # The first stage that loses a gold turn is where to look.
    lost = next((stage for stage in STAGES if not row[stage]), "")
    row["first_loss_stage"] = lost
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profiles", default="h6,h8")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()

    store = SQLiteGraphStore(args.source_db, read_only=True)
    config = load_config(args.config)
    questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    if args.max_questions:
        questions = questions[:args.max_questions]
    embedding = QwenEmbeddingIndex(store, config, record_usage=False) if args.embedding else None
    counter = resolve_token_counter(BACKBONE, require_exact=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    all_rows: list[dict] = []
    for label in (item.strip() for item in args.profiles.split(",") if item.strip()):
        profile = HarnessProfile(label)
        budget = replace(config.query_budget, max_evidence_turns=32)
        navigator = GraphNavigator(store, dense_search=embedding.search if embedding else None,
                                   harness_profile=profile)
        rows = []
        for index, question in enumerate(questions, 1):
            result = navigator.navigate(question.memory_id, question.query, budget)
            row = funnel_row(store, question, result, budget, counter)
            row["configuration"] = label
            rows.append(row); all_rows.append(row)
            if index % 25 == 0:
                print(f"[{label}] funnel {index}/{len(questions)}", flush=True)
        by_stratum: dict[str, dict] = {}
        for stratum in sorted({row["stratum"] for row in rows}):
            subset = [row for row in rows if row["stratum"] == stratum]
            by_stratum[stratum] = {
                **{stage: sum(row[stage] for row in subset) / len(subset) for stage in STAGES},
                "budget_oracle_fits": sum(row["budget_oracle_fits"] for row in subset) / len(subset),
                "binding_oracle_all_hit": sum(row["binding_oracle_all_hit"] for row in subset) / len(subset),
                "questions": len(subset),
            }
        summary[label] = {
            "overall": {
                **{stage: sum(row[stage] for row in rows) / len(rows) for stage in STAGES},
                "budget_oracle_fits": sum(row["budget_oracle_fits"] for row in rows) / len(rows),
                "binding_oracle_all_hit": sum(row["binding_oracle_all_hit"] for row in rows) / len(rows),
                "ast_diverges": sum(row["ast_diverges"] for row in rows) / len(rows),
                "questions": len(rows),
            },
            "by_stratum": by_stratum,
            "first_loss_stage": dict(Counter(row["first_loss_stage"] for row in rows).most_common()),
        }

    (args.output_root / "funnel_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_root / "funnel.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    store.close()


if __name__ == "__main__":
    main()
