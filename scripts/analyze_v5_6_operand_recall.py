#!/usr/bin/env python3
"""Can an operand identify the collection its question is about?

Aggregation is 40% of the development set and its worst category.  The blocker
found in step 1 was that operand predicate candidates are retrieved from the
graph by embedding similarity rather than parsed from the question, so matching
a COLLECTION_MANIFEST against them selected nearly every manifest.

This measures, per question:

* whether any manifest matches the operand at all (identification rate);
* whether the matched manifests' members include the facts that cite the gold
  turns (**collection recall** -- the number that decides whether a count can
  ever be right);
* how tight the match is (**precision proxy**: matched members vs whole memory).

Gold is used only here, in the evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.principals import build_principal_registry  # noqa: E402
from graphmem.retrieval.query_ir import compile_query  # noqa: E402
from graphmem.runtime import GraphReadView  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.text import content_terms  # noqa: E402


def gold_fact_ids(store, question, view) -> set[str]:
    """CanonicalFact node ids whose evidence cites a gold turn."""
    keys = {(row.session_id, row.turn_index) for row in question.gold_turns}
    if not keys:
        return set()
    turn_ids = {turn.turn_id for turn in store.turns(question.memory_id)
                if (turn.session_id, turn.turn_index) in keys}
    groups = {group.evidence_group_id for group in store.evidence_groups(question.memory_id)
              if any(member.turn_id in turn_ids for member in group.members)}
    return {node.node_id for node in view.nodes.values()
            if node.node_type == NodeType.CANONICAL_FACT
            and set(node.all_evidence_group_ids) & groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()

    store = SQLiteGraphStore(args.source_db, read_only=True)
    load_config(args.config)
    questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    if args.max_questions:
        questions = questions[:args.max_questions]

    view_cache: dict[str, GraphReadView] = {}
    registry_cache: dict[str, object] = {}
    rows = []
    for index, question in enumerate(questions, 1):
        memory_id = question.memory_id
        if memory_id not in view_cache:
            view_cache.clear(); registry_cache.clear()
            view_cache[memory_id] = GraphReadView(store.nodes(memory_id), store.edges(memory_id))
            registry_cache[memory_id] = build_principal_registry(
                store, memory_id, view_cache[memory_id])
        view = view_cache[memory_id]
        ir = compile_query(question.query, view, registry=registry_cache[memory_id])

        manifests = [node for node in view.nodes.values()
                     if node.node_type == NodeType.COLLECTION_MANIFEST]
        all_facts = {node.node_id for node in view.nodes.values()
                     if node.node_type == NodeType.CANONICAL_FACT}
        gold = gold_fact_ids(store, question, view)
        strategies = {
            "head_action": frozenset(ir.slots.head_terms) | frozenset(ir.slots.action_terms),
            "content": frozenset(ir.slots.content_terms),
        }
        variants = {}
        for name, question_terms in strategies.items():
            hits = [node for node in manifests
                    if question_terms and (
                        (question_terms & content_terms(str(node.attributes.get("predicate", ""))))
                        or any(question_terms & content_terms(str(value))
                               for value in node.attributes.get("value_keys", ())))]
            picked: set[str] = set()
            for node in hits:
                picked.update(str(item) for item in node.attributes.get("member_ids", ()))
            variants[name] = (hits, picked)
        # "reservoir" uses retrieval rather than lexical matching: keep any
        # manifest that already holds a fact the navigator reached.  It is the
        # upper bound the lexical strategies are chasing.
        reachable = {node.node_id for node in view.nodes.values()
                     if node.node_type == NodeType.CANONICAL_FACT}
        matched, members = variants["head_action"]
        for name, (hits, picked) in variants.items():
            rows_extra = {
                f"{name}_identified": bool(hits),
                f"{name}_members": len(picked),
                f"{name}_recall": (len(gold & picked) / len(gold)) if gold else None,
                f"{name}_all_hit": (bool(gold) and gold <= picked) if gold else None,
                f"{name}_coverage": (len(picked) / len(reachable)) if reachable else None,
            }
            variants[name] = (hits, picked, rows_extra)
        extra = {}
        for name in strategies:
            extra.update(variants[name][2])
        rows.append({
            **extra,
            "dev_question_id": question.question_id, "stratum": question.stratum,
            "is_count": ir.slots.is_count, "is_list": ir.slots.is_list,
            "aggregation": ir.slots.is_count or ir.slots.is_list,
            "head_terms": list(ir.slots.head_terms),
            "action_terms": list(ir.slots.action_terms)[:6],
            "manifests_in_memory": len(manifests),
            "manifests_matched": len(matched),
            "identified": bool(matched),
            "members_matched": len(members),
            "facts_in_memory": len(all_facts),
            # Share of the memory the matched collection covers.  1.0 means the
            # "collection" is the whole memory, i.e. no constraint at all.
            "coverage_of_memory": (len(members) / len(all_facts)) if all_facts else None,
            "gold_facts": len(gold),
            # Does the identified collection contain the facts that prove the answer?
            "collection_recall": (len(gold & members) / len(gold)) if gold else None,
            "collection_all_hit": (bool(gold) and gold <= members) if gold else None,
        })
        if index % 50 == 0:
            print(f"operand recall {index}/{len(questions)}", flush=True)

    def mean(key, subset):
        values = [row[key] for row in subset if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    agg = [row for row in rows if row["aggregation"]]
    with_gold = [row for row in rows if row["gold_facts"]]
    summary = {
        "questions": len(rows),
        "aggregation_questions": len(agg),
        "identified_rate": mean("identified", rows),
        "identified_rate_aggregation": mean("identified", agg),
        "collection_recall": mean("collection_recall", with_gold),
        "collection_all_hit": mean("collection_all_hit", with_gold),
        "coverage_of_memory": mean("coverage_of_memory", rows),
        "manifests_matched_mean": mean("manifests_matched", rows),
        "manifests_in_memory_mean": mean("manifests_in_memory", rows),
        "by_stratum": {},
    }
    for stratum in sorted({row["stratum"] for row in rows}):
        subset = [row for row in rows if row["stratum"] == stratum]
        gold_subset = [row for row in subset if row["gold_facts"]]
        summary["by_stratum"][stratum] = {
            "questions": len(subset),
            "identified_rate": mean("identified", subset),
            "collection_recall": mean("collection_recall", gold_subset),
            "collection_all_hit": mean("collection_all_hit", gold_subset),
            "coverage_of_memory": mean("coverage_of_memory", subset),
        }
    summary["head_term_empty"] = sum(1 for row in rows if not row["head_terms"])
    summary["common_heads"] = Counter(
        term for row in agg for term in row["head_terms"]).most_common(15)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, indent=2))
    store.close()


if __name__ == "__main__":
    main()
