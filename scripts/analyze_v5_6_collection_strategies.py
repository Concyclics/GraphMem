#!/usr/bin/env python3
"""Locate a question's collection by retrieval instead of by lexicon.

Lexical matching plateaus at 38.2% collection recall / 19.3% all-hit: widening
the term set from head+action to every question content word moves recall by
only 4.5pp, because a collection is keyed by its *predicate* ("inherit") while
the question names its *values* ("antique items"), and the graph's per-memory
predicate vocabulary does not coincide with the question's verb.

The fact reservoir, by contrast, reaches every gold turn for 200 of 200
questions.  So this compares strategies that admit a manifest because it
*contains a fact retrieval already reached*, using question terms only to rank:

  lexical            manifests whose predicate or values overlap question terms
  reservoir          manifests holding any fact in the wide reservoir
  bound              manifests holding any fact that actually bound
  bound_ranked_k     as `bound`, keeping the k best by question-term overlap

Recall is the ceiling each strategy could support; coverage is what it costs.
Gold is used only here, in the evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.runtime import GraphReadView  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.text import content_terms  # noqa: E402

TOP_K = (1, 3, 8)


def gold_fact_ids(store, question, view) -> set[str]:
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


def members_of(nodes) -> set[str]:
    picked: set[str] = set()
    for node in nodes:
        picked.update(str(item) for item in node.attributes.get("member_ids", ()))
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()

    store = SQLiteGraphStore(args.source_db, read_only=True)
    config = load_config(args.config)
    questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    if args.max_questions:
        questions = questions[:args.max_questions]
    budget = replace(config.query_budget, max_evidence_turns=32)
    embedding = QwenEmbeddingIndex(store, config, record_usage=False) if args.embedding else None
    navigator = GraphNavigator(store, dense_search=embedding.search if embedding else None,
                               harness_profile=HarnessProfile.H10_AST)

    views: dict[str, GraphReadView] = {}
    rows = []
    for index, question in enumerate(questions, 1):
        memory_id = question.memory_id
        if memory_id not in views:
            views = {memory_id: GraphReadView(store.nodes(memory_id), store.edges(memory_id))}
        view = views[memory_id]
        result = navigator.navigate(memory_id, question.query, budget)

        manifests = [node for node in view.nodes.values()
                     if node.node_type == NodeType.COLLECTION_MANIFEST]
        all_facts = {node.node_id for node in view.nodes.values()
                     if node.node_type == NodeType.CANONICAL_FACT}
        gold = gold_fact_ids(store, question, view)
        question_terms = frozenset(content_terms(question.query))

        reservoir = set(result.reservoir_fact_node_ids) or set(result.reached_fact_node_ids)
        bound = set(result.bound_fact_node_ids)

        def overlap(node) -> int:
            score = len(question_terms & content_terms(str(node.attributes.get("predicate", ""))))
            for value in node.attributes.get("value_keys", ()):
                score += len(question_terms & content_terms(str(value)))
            return score

        strategies: dict[str, list] = {
            "lexical": [node for node in manifests if overlap(node)],
            "reservoir": [node for node in manifests
                          if reservoir & set(map(str, node.attributes.get("member_ids", ())))],
            "bound": [node for node in manifests
                      if bound & set(map(str, node.attributes.get("member_ids", ())))],
        }
        ranked = sorted(strategies["bound"],
                        key=lambda node: (-overlap(node), node.node_id))
        for k in TOP_K:
            strategies[f"bound_ranked_{k}"] = ranked[:k]

        row = {"dev_question_id": question.question_id, "stratum": question.stratum,
               "gold_facts": len(gold), "facts_in_memory": len(all_facts),
               "manifests": len(manifests), "reservoir_facts": len(reservoir),
               "bound_facts": len(bound)}
        for name, nodes in strategies.items():
            picked = members_of(nodes)
            row[f"{name}_manifests"] = len(nodes)
            row[f"{name}_members"] = len(picked)
            row[f"{name}_identified"] = bool(nodes)
            row[f"{name}_recall"] = (len(gold & picked) / len(gold)) if gold else None
            row[f"{name}_all_hit"] = (bool(gold) and gold <= picked) if gold else None
            row[f"{name}_coverage"] = (len(picked) / len(all_facts)) if all_facts else None
        rows.append(row)
        if index % 25 == 0:
            print(f"strategies {index}/{len(questions)}", flush=True)

    names = ["lexical", "reservoir", "bound"] + [f"bound_ranked_{k}" for k in TOP_K]

    def mean(key, subset):
        values = [row[key] for row in subset if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    with_gold = [row for row in rows if row["gold_facts"]]
    summary = {"questions": len(rows), "strategies": {}, "by_stratum": {}}
    for name in names:
        summary["strategies"][name] = {
            "identified": mean(f"{name}_identified", rows),
            "recall": mean(f"{name}_recall", with_gold),
            "all_hit": mean(f"{name}_all_hit", with_gold),
            "coverage": mean(f"{name}_coverage", rows),
            "manifests": mean(f"{name}_manifests", rows),
            "members": mean(f"{name}_members", rows),
        }
    for stratum in sorted({row["stratum"] for row in rows}):
        subset = [row for row in rows if row["stratum"] == stratum]
        gold_subset = [row for row in subset if row["gold_facts"]]
        summary["by_stratum"][stratum] = {
            name: {"recall": mean(f"{name}_recall", gold_subset),
                   "all_hit": mean(f"{name}_all_hit", gold_subset),
                   "coverage": mean(f"{name}_coverage", subset)}
            for name in names}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary["strategies"], indent=2))
    store.close()


if __name__ == "__main__":
    main()
