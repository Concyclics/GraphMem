#!/usr/bin/env python3
"""Audit exact-vs-morphological state keys on the complete benchmark.

The online relation graph currently carries ``(owner, exact predicate)``
witnesses.  Frozen extraction often emits tense/aspect variants such as
``attended`` and ``attending`` for the same relation, so this audit measures
how many gold session pairs are disconnected only by that exact-phrase gate.
Gold is used solely by this offline diagnostic; the proposed family itself is
deterministic and benchmark independent.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import NodeType  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.principals import build_principal_registry  # noqa: E402
from graphmem.retrieval.query_ir import compile_query  # noqa: E402
from graphmem.runtime import GraphReadView  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.text import normalize_key, predicate_family  # noqa: E402


def _fact_rows(view: GraphReadView) -> list[dict[str, str]]:
    rows = []
    for node in view.nodes.values():
        if node.node_type != NodeType.CANONICAL_FACT:
            continue
        attrs = node.attributes
        roles = attrs.get("relation_entity_roles", {})
        owners = roles.get("owner", ()) if isinstance(roles, dict) else ()
        if isinstance(owners, str):
            owners = (owners,)
        predicate = normalize_key(str(attrs.get("predicate", "")))
        family = predicate_family(predicate)
        session = str(attrs.get("session_id", ""))
        if not session or not predicate or not family:
            continue
        for owner in owners:
            owner_key = normalize_key(str(owner))
            if owner_key:
                rows.append({
                    "owner": owner_key,
                    "predicate": predicate,
                    "family": family,
                    "polarity": normalize_key(str(attrs.get("polarity", "positive"))),
                    "modality": normalize_key(str(attrs.get("modality", "asserted"))),
                    "session": session,
                })
    return rows


def _keys_by_session(rows: list[dict[str, str]], *, family: bool) -> dict[str, set[tuple[str, ...]]]:
    result: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    predicate_key = "family" if family else "predicate"
    for row in rows:
        result[row["session"]].add((
            row["owner"], row[predicate_key], row["polarity"], row["modality"]))
    return result


def _query_keys(ir, *, family: bool) -> set[tuple[str, str]]:
    result = set()
    for operand in ir.operands:
        owners = [normalize_key(value) for value in operand.owner_aliases]
        predicates = [normalize_key(value) for value in operand.predicate_candidates]
        if family:
            predicates = [predicate_family(value) for value in predicates]
        for owner in filter(None, owners):
            for predicate in filter(None, predicates):
                result.add((owner, predicate))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))
    store = SQLiteGraphStore(args.source_db, read_only=True)
    by_memory = defaultdict(list)
    for question in questions:
        by_memory[question.memory_id].append(question)

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    examples = []
    family_sizes: Counter[tuple[str, str, str, str, str]] = Counter()
    for memory_index, (memory_id, memory_questions) in enumerate(
            sorted(by_memory.items()), 1):
        view = GraphReadView(store.nodes(memory_id), store.edges(memory_id))
        registry = build_principal_registry(store, memory_id, view)
        facts = _fact_rows(view)
        exact_by_session = _keys_by_session(facts, family=False)
        family_by_session = _keys_by_session(facts, family=True)
        phrase_sets: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        for row in facts:
            key = (row["owner"], row["family"], row["polarity"], row["modality"])
            phrase_sets[key].add(row["predicate"])
        family_sizes.update({(memory_id, *key): len(values)
                             for key, values in phrase_sets.items()})

        for full in memory_questions:
            question = full.question
            bucket = question.stratum
            gold_sessions = tuple(dict.fromkeys(question.gold_sessions))
            if len(gold_sessions) < 2:
                continue
            counters[bucket]["multi_gold_questions"] += 1
            pairs = tuple(itertools.combinations(gold_sessions, 2))
            exact_shared = set().union(*(
                exact_by_session.get(left, set()) & exact_by_session.get(right, set())
                for left, right in pairs))
            family_shared = set().union(*(
                family_by_session.get(left, set()) & family_by_session.get(right, set())
                for left, right in pairs))
            if exact_shared:
                counters[bucket]["exact_connected"] += 1
            if family_shared:
                counters[bucket]["family_connected"] += 1
            if family_shared and not exact_shared:
                counters[bucket]["family_only_connected"] += 1

            ir = compile_query(question.query, view, registry=registry).promote_ast()
            exact_query = _query_keys(ir, family=False)
            family_query = _query_keys(ir, family=True)
            exact_relevant = {key for key in exact_shared if key[:2] in exact_query}
            family_relevant = {key for key in family_shared if key[:2] in family_query}
            if exact_relevant:
                counters[bucket]["exact_query_relevant"] += 1
            if family_relevant:
                counters[bucket]["family_query_relevant"] += 1
            if family_relevant and not exact_relevant:
                counters[bucket]["family_only_query_relevant"] += 1
                if len(examples) < 100:
                    examples.append({
                        "question_id": question.question_id,
                        "benchmark": question.benchmark,
                        "stratum": bucket,
                        "question": question.query,
                        "gold_sessions": gold_sessions,
                        "query_family_keys": sorted(family_query),
                        "family_shared": sorted(family_relevant),
                    })
        if memory_index % 50 == 0:
            print(f"audited {memory_index}/{len(by_memory)} memories", flush=True)

    store.close()
    aggregate = Counter()
    for row in counters.values():
        aggregate.update(row)
    family_histogram = Counter(family_sizes.values())
    output = {
        "source_db": str(args.source_db),
        "questions": len(questions),
        "memories": len(by_memory),
        "aggregate": dict(sorted(aggregate.items())),
        "by_stratum": {key: dict(sorted(value.items()))
                       for key, value in sorted(counters.items())},
        "predicate_family_phrase_count_histogram": {
            str(key): value for key, value in sorted(family_histogram.items())},
        "predicate_family_groups": len(family_sizes),
        "predicate_family_groups_with_variants": sum(
            1 for value in family_sizes.values() if value > 1),
        "predicate_family_max_phrases": max(family_sizes.values(), default=0),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
