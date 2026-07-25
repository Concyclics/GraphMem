#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * ratio(numerator, denominator), 2)


def evidence_leaf_id(question_id: str, label: str) -> str | None:
    match = re.fullmatch(r"D(\d+):(\d+)", str(label), flags=re.IGNORECASE)
    if not match:
        return None
    return (
        f"{question_id}:session_{int(match.group(1))}:leaf:"
        f"{(int(match.group(2)) - 1) // 2}"
    )


def owner_id(question_id: str) -> str:
    match = re.match(r"locomo(\d+)_", question_id)
    if not match:
        raise ValueError(f"unexpected LoCoMo question id: {question_id}")
    return f"locomo{int(match.group(1)):02d}_0000"


def owner_node_id(node_id: str, question_id: str) -> str:
    return owner_id(question_id) + node_id[node_id.index(":") :]


def coverage_bucket(hits: int, total: int) -> str:
    if hits <= 0:
        return "none"
    if hits >= total:
        return "all"
    return "partial"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variant = args.run_dir / "hierarchical_state_graph_v2"
    cases = {row["question_id"]: row for row in json.loads(args.data.read_text())}
    answers = {row["question_id"]: row for row in read_jsonl(variant / "answers.jsonl")}
    retrieval = {
        row["question_id"]: row for row in read_jsonl(variant / "retrieval_results.jsonl")
    }
    judgments = {
        row["question_id"]: row for row in read_jsonl(args.judge_dir / "auto_eval.jsonl")
    }

    facts_by_id: dict[str, dict[str, Any]] = {}
    facts_by_source: dict[str, set[str]] = defaultdict(set)
    for node in read_jsonl(variant / "nodes.jsonl"):
        if node.get("node_type") != "atomic_fact":
            continue
        facts_by_id[node["node_id"]] = node
        for source_id in node.get("source_leaf_ids") or []:
            facts_by_source[source_id].add(node["node_id"])

    aggregates: dict[str, Counter[str]] = defaultdict(Counter)
    channel_hits: dict[str, Counter[str]] = defaultdict(Counter)
    per_question: list[dict[str, Any]] = []
    relation_totals: dict[str, Counter[str]] = defaultdict(Counter)

    def add(group: str, correct: bool, evidence_count: int, metrics: dict[str, Any]) -> None:
        bucket = aggregates[group]
        bucket["questions"] += 1
        bucket["correct"] += int(correct)
        bucket["evidence_turns"] += evidence_count
        for key in (
            "index_fact_hits",
            "semantic_hits",
            "bm25_hits",
            "entity_hits",
            "postpack_leaf_hits",
            "exact_text_hits",
            "selected_fact_source_hits",
            "prompt_support_hits",
        ):
            bucket[key] += int(metrics[key])
        bucket[f"support_{metrics['prompt_support_bucket']}"] += 1
        bucket[f"exact_{metrics['exact_text_bucket']}"] += 1
        bucket["abstentions"] += int(metrics["abstention"])
        bucket["wrong_abstentions"] += int(metrics["abstention"] and not correct)
        bucket["retained_graph_questions"] += int(metrics["retained_graph_count"] > 0)
        bucket["retained_graph_nodes"] += int(metrics["retained_graph_count"])
        bucket["novel_operator_questions"] += int(metrics["novel_operator_count"] > 0)
        bucket["novel_operator_facts"] += int(metrics["novel_operator_count"])

    for question_id, judgment in judgments.items():
        case = cases[question_id]
        answer = answers[question_id]
        row = retrieval[question_id]
        trace = row.get("retrieval_trace") or {}
        evidence_labels = [str(item) for item in case.get("locomo_evidence") or []]
        evidence_ids = [
            item
            for label in evidence_labels
            if (item := evidence_leaf_id(question_id, label)) is not None
        ]
        evidence_owner_ids = [owner_node_id(item, question_id) for item in evidence_ids]
        evidence_turn_text: dict[str, str] = {}
        for session in case.get("haystack_sessions") or []:
            for turn in session:
                evidence_turn_text[str(turn.get("dia_id"))] = str(turn.get("content") or "")

        leaf_channels = trace.get("leaf_channels") or {}
        semantic = set((leaf_channels.get("semantic_rank_ids") or [])[:28])
        bm25 = set((leaf_channels.get("bm25_rank_ids") or [])[:28])
        entity = set((leaf_channels.get("entity_rank_ids") or [])[:28])
        postpack_leaves = set((trace.get("postpack") or {}).get("leaf_ids") or [])
        selected_fact_ids = {
            owner_node_id(item, question_id)
            for item in ((trace.get("postpack") or {}).get("fact_ids") or [])
        }
        context = str(row.get("context_text") or "").casefold()

        index_hits = sum(bool(facts_by_source.get(item)) for item in evidence_owner_ids)
        semantic_hits = sum(item in semantic for item in evidence_ids)
        bm25_hits = sum(item in bm25 for item in evidence_ids)
        entity_hits = sum(item in entity for item in evidence_ids)
        postpack_leaf_hits = sum(item in postpack_leaves for item in evidence_ids)
        exact_text_hits = sum(
            bool((text := evidence_turn_text.get(label, "").strip()) and text.casefold() in context)
            for label in evidence_labels
        )
        selected_fact_source_hits = sum(
            bool(facts_by_source.get(item, set()) & selected_fact_ids)
            for item in evidence_owner_ids
        )
        prompt_support_hits = sum(
            (
                evidence_ids[index] in postpack_leaves
                or bool(facts_by_source.get(evidence_owner_ids[index], set()) & selected_fact_ids)
            )
            for index in range(len(evidence_ids))
        )

        prediction = str(answer.get("prediction") or "")
        abstention = bool(
            re.search(
                r"\b(?:not enough information|no information|not mentioned|"
                r"cannot determine|can't determine|insufficient evidence|"
                r"does not (?:say|state|mention)|doesn't (?:say|state|mention))\b",
                prediction.casefold(),
            )
        )
        initial = set(trace.get("initial_card_ids") or []) | set(
            trace.get("initial_fact_ids") or []
        )
        expanded = set(trace.get("typed_expanded_node_ids") or []) | set(
            trace.get("operand_expanded_fact_ids") or []
        )
        postpack_nodes = set((trace.get("postpack") or {}).get("card_ids") or []) | set(
            (trace.get("postpack") or {}).get("fact_ids") or []
        )
        operator_sources = set(trace.get("operator_operand_fact_ids") or [])
        retained_graph = expanded & postpack_nodes
        novel_operator = (expanded - initial) & operator_sources
        for relation, counts in (trace.get("relation_contributions") or {}).items():
            relation_totals[relation]["expanded"] += int(counts.get("expanded") or 0)
            relation_totals[relation]["postpack"] += int(counts.get("postpack") or 0)
            relation_totals[relation]["operator_source"] += int(
                counts.get("operator_source") or 0
            )
            relation_totals[relation]["questions"] += 1
            relation_totals[relation]["wrong_postpack"] += int(
                (not judgment["correct"]) and bool(counts.get("postpack"))
            )

        metrics = {
            "index_fact_hits": index_hits,
            "semantic_hits": semantic_hits,
            "bm25_hits": bm25_hits,
            "entity_hits": entity_hits,
            "postpack_leaf_hits": postpack_leaf_hits,
            "exact_text_hits": exact_text_hits,
            "selected_fact_source_hits": selected_fact_source_hits,
            "prompt_support_hits": prompt_support_hits,
            "prompt_support_bucket": coverage_bucket(prompt_support_hits, len(evidence_ids)),
            "exact_text_bucket": coverage_bucket(exact_text_hits, len(evidence_ids)),
            "abstention": abstention,
            "retained_graph_count": len(retained_graph),
            "novel_operator_count": len(novel_operator),
        }
        correct = bool(judgment["correct"])
        add("overall", correct, len(evidence_ids), metrics)
        add(f"category_{case['locomo_category']}", correct, len(evidence_ids), metrics)
        add(f"query_kind_{row.get('query_kind')}", correct, len(evidence_ids), metrics)
        add(f"support_{metrics['prompt_support_bucket']}", correct, len(evidence_ids), metrics)
        add(f"conversation_{case['locomo_sample_id']}", correct, len(evidence_ids), metrics)

        outcome = "correct" if correct else "wrong"
        for name, hits in (
            ("semantic", semantic_hits),
            ("bm25", bm25_hits),
            ("entity", entity_hits),
            ("postpack_leaf", postpack_leaf_hits),
            ("selected_fact_source", selected_fact_source_hits),
            ("prompt_support", prompt_support_hits),
        ):
            channel_hits[name]["evidence_turns"] += len(evidence_ids)
            channel_hits[name]["hits"] += hits
            channel_hits[name][f"{outcome}_turns"] += len(evidence_ids)
            channel_hits[name][f"{outcome}_hits"] += hits

        per_question.append(
            {
                "question_id": question_id,
                "conversation_id": case["locomo_sample_id"],
                "category": int(case["locomo_category"]),
                "query_kind": row.get("query_kind"),
                "correct": correct,
                "question": case["question"],
                "gold_answer": case["answer"],
                "prediction": prediction,
                "judge_reasoning": judgment.get("reasoning"),
                "evidence_count": len(evidence_ids),
                **metrics,
            }
        )

    def summarize(counter: Counter[str]) -> dict[str, Any]:
        questions = counter["questions"]
        evidence_turns = counter["evidence_turns"]
        return {
            **dict(counter),
            "accuracy_pct": pct(counter["correct"], questions),
            "index_fact_recall_pct": pct(counter["index_fact_hits"], evidence_turns),
            "semantic_top28_recall_pct": pct(counter["semantic_hits"], evidence_turns),
            "bm25_top28_recall_pct": pct(counter["bm25_hits"], evidence_turns),
            "entity_top28_recall_pct": pct(counter["entity_hits"], evidence_turns),
            "postpack_leaf_recall_pct": pct(counter["postpack_leaf_hits"], evidence_turns),
            "exact_text_recall_pct": pct(counter["exact_text_hits"], evidence_turns),
            "selected_fact_source_recall_pct": pct(
                counter["selected_fact_source_hits"], evidence_turns
            ),
            "prompt_support_recall_pct": pct(counter["prompt_support_hits"], evidence_turns),
            "abstention_pct": pct(counter["abstentions"], questions),
            "wrong_abstention_pct_of_questions": pct(
                counter["wrong_abstentions"], questions
            ),
        }

    summary = {
        "scope": {
            "judge_questions": len(judgments),
            "categories": [1, 2, 3, 4],
            "category_5_excluded": True,
            "judge_correct": sum(bool(row["correct"]) for row in judgments.values()),
        },
        "groups": {
            key: summarize(value)
            for key, value in sorted(aggregates.items())
        },
        "channel_recall_by_outcome": {
            key: {
                **dict(value),
                "overall_recall_pct": pct(value["hits"], value["evidence_turns"]),
                "correct_recall_pct": pct(value["correct_hits"], value["correct_turns"]),
                "wrong_recall_pct": pct(value["wrong_hits"], value["wrong_turns"]),
            }
            for key, value in sorted(channel_hits.items())
        },
        "relation_usage": {
            key: dict(value) for key, value in sorted(relation_totals.items())
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_question:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    overall = summary["groups"]["overall"]
    lines = [
        "# LoCoMo memory-benchmarks judge gap diagnostics",
        "",
        f"- Accuracy: {overall['correct']}/{overall['questions']} = {overall['accuracy_pct']:.2f}%",
        f"- Indexed atomic-fact coverage of official evidence: {overall['index_fact_recall_pct']:.2f}%",
        f"- Leaf semantic top-28 evidence recall: {overall['semantic_top28_recall_pct']:.2f}%",
        f"- Final raw-leaf evidence recall: {overall['postpack_leaf_recall_pct']:.2f}%",
        f"- Final selected-fact source recall: {overall['selected_fact_source_recall_pct']:.2f}%",
        f"- Combined prompt support recall: {overall['prompt_support_recall_pct']:.2f}%",
        "",
        "## Accuracy by prompt-support completeness",
        "",
        "| Support | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("support_all", "support_partial", "support_none"):
        bucket = summary["groups"].get(key, {})
        lines.append(
            f"| {key.removeprefix('support_')} | {bucket.get('correct', 0)} | "
            f"{bucket.get('questions', 0)} | {bucket.get('accuracy_pct', 0):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Accuracy and support by category",
            "",
            "| Category | Correct | Total | Accuracy | Prompt support recall | Wrong abstentions |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category in (1, 2, 3, 4):
        bucket = summary["groups"][f"category_{category}"]
        lines.append(
            f"| {category} | {bucket['correct']} | {bucket['questions']} | "
            f"{bucket['accuracy_pct']:.2f}% | {bucket['prompt_support_recall_pct']:.2f}% | "
            f"{bucket['wrong_abstentions']} |"
        )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["groups"]["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
