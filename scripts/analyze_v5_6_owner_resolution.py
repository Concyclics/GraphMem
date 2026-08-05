#!/usr/bin/env python3
"""Intrinsic evaluation of principal-aware owner resolution.

Owner resolution is measured on its own, before it is allowed to influence
binding, because PR4b showed that changing binding on top of a broken owner
signal only moves the damage around.

Gold fact owners are used **here only**, in the evaluator.  The resolver itself
sees nothing but the question and the memory's own turns.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from graphmem.domain import NodeType
from graphmem.eval import load_dev_questions, load_gold_turns
from graphmem.principals import build_principal_registry, resolve_query_owners
from graphmem.retrieval.bindings import UNTRUSTED_OWNER_ALIASES
from graphmem.retrieval.query_ir import _query_owners
from graphmem.runtime import GraphReadView
from graphmem.storage import SQLiteGraphStore


def gold_fact_owners(store, question) -> set[str]:
    """Owner ids of the CanonicalFacts that cite this question's gold turns."""
    keys = {(row.session_id, row.turn_index) for row in question.gold_turns}
    gold_turns = {turn.turn_id for turn in store.turns(question.memory_id)
                  if (turn.session_id, turn.turn_index) in keys}
    groups = {group.evidence_group_id for group in store.evidence_groups(question.memory_id)
              if any(member.turn_id in gold_turns for member in group.members)}
    owners: set[str] = set()
    for node in store.nodes(question.memory_id):
        if node.node_type == NodeType.CANONICAL_FACT and set(node.all_evidence_group_ids) & groups:
            owner = str(node.attributes.get("owner_id", "") or "")
            if owner:
                owners.add(owner)
    return owners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()

    store = SQLiteGraphStore(args.source_db, read_only=True)
    questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    if args.max_questions:
        questions = questions[:args.max_questions]

    views: dict[str, GraphReadView] = {}
    registries: dict[str, object] = {}
    rows: list[dict] = []
    for index, question in enumerate(questions, 1):
        memory_id = question.memory_id
        if memory_id not in views:
            views = {memory_id: GraphReadView(store.nodes(memory_id), store.edges(memory_id))}
            registries = {memory_id: build_principal_registry(store, memory_id, views[memory_id])}
        view, registry = views[memory_id], registries[memory_id]

        resolved, warnings = resolve_query_owners(question.query, registry)
        new_entities = {item for row in resolved for item in row.canonical_entity_ids}
        strong_entities = {item for row in resolved if row.strong for item in row.canonical_entity_ids}
        legacy = _query_owners(question.query, view)
        legacy_entities = {item for _, ids in legacy for item in ids}
        legacy_aliases = [alias for alias, _ in legacy]

        gold_owners = gold_fact_owners(store, question)
        rows.append({
            "dev_question_id": question.question_id,
            "stratum": question.stratum,
            "has_gold_owner": bool(gold_owners),
            # --- new resolver ---
            "resolved": bool(resolved),
            "kinds": sorted({row.resolution_kind for row in resolved}),
            "first_person": any(row.resolution_kind == "first_person" for row in resolved),
            "owner_count": len(resolved),
            "gold_owner_recall": bool(gold_owners & new_entities) if gold_owners else None,
            "strong_owner_correct": bool(gold_owners & strong_entities) if strong_entities else None,
            "wrong_strong_owner": bool(strong_entities and gold_owners
                                       and not (gold_owners & strong_entities)),
            "common_word_owner": any(row.mention_text in UNTRUSTED_OWNER_ALIASES
                                     for row in resolved),
            "multi_owner": len(resolved) > 1,
            "warnings": sorted(warnings),
            # --- legacy resolver, same questions ---
            "legacy_resolved": bool(legacy),
            "legacy_gold_owner_recall": bool(gold_owners & legacy_entities) if gold_owners else None,
            "legacy_common_word_owner": any(alias in UNTRUSTED_OWNER_ALIASES
                                            for alias in legacy_aliases),
            "legacy_aliases": legacy_aliases,
        })
        if index % 50 == 0:
            print(f"owner resolution {index}/{len(questions)}", flush=True)

    def rate(key: str, subset=None) -> float:
        pool = [row for row in (subset if subset is not None else rows) if row.get(key) is not None]
        return sum(bool(row[key]) for row in pool) / max(1, len(pool))

    with_owner = [row for row in rows if row["has_gold_owner"]]
    first_person_rows = [row for row in rows
                         if any(word in row["dev_question_id"] for word in ()) or row["first_person"]]
    summary = {
        "questions": len(rows),
        "questions_with_gold_owner": len(with_owner),
        "new": {
            "resolved_rate": rate("resolved"),
            "gold_owner_recall": rate("gold_owner_recall", with_owner),
            "wrong_strong_owner_rate": sum(row["wrong_strong_owner"] for row in rows) / max(1, len(rows)),
            "common_word_owner_rate": sum(row["common_word_owner"] for row in rows) / max(1, len(rows)),
            "first_person_rate": sum(row["first_person"] for row in rows) / max(1, len(rows)),
            "multi_owner_rate": sum(row["multi_owner"] for row in rows) / max(1, len(rows)),
            "no_owner_rate": 1 - rate("resolved"),
        },
        "legacy": {
            "resolved_rate": rate("legacy_resolved"),
            "gold_owner_recall": rate("legacy_gold_owner_recall", with_owner),
            "common_word_owner_rate": sum(row["legacy_common_word_owner"] for row in rows) / max(1, len(rows)),
            "no_owner_rate": 1 - rate("legacy_resolved"),
        },
        "resolution_kinds": dict(Counter(kind for row in rows for kind in row["kinds"]).most_common()),
        "warnings": dict(Counter(item for row in rows for item in row["warnings"]).most_common()),
        "by_stratum": {},
    }
    for stratum in sorted({row["stratum"] for row in rows}):
        subset = [row for row in rows if row["stratum"] == stratum]
        owned = [row for row in subset if row["has_gold_owner"]]
        summary["by_stratum"][stratum] = {
            "questions": len(subset),
            "new_gold_owner_recall": rate("gold_owner_recall", owned),
            "legacy_gold_owner_recall": rate("legacy_gold_owner_recall", owned),
            "first_person_rate": sum(row["first_person"] for row in subset) / max(1, len(subset)),
            "common_word_owner_rate": sum(row["common_word_owner"] for row in subset) / max(1, len(subset)),
            "legacy_common_word_owner_rate": sum(row["legacy_common_word_owner"] for row in subset) / max(1, len(subset)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, indent=2))
    store.close()


if __name__ == "__main__":
    main()
