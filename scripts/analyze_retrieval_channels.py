#!/usr/bin/env python3
"""Audit GraphMem retrieval channels and the effective use of graph edges.

This script only reads an existing run.  Gold session ids are used for offline
evaluation, never by retrieval.  It reports:

* semantic/BM25/entity session recall at the configured layer cutoffs;
* unique session hits contributed by each channel;
* whether typed expansion adds a gold session missed by the initial seeds;
* whether expanded nodes survive the final evidence pack;
* which edge relations connect initial seeds to expanded candidates, including
  directed inbound edges that the current forward-only traversal cannot use.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def node_session(node_id: str) -> str:
    parts = node_id.split(":")
    return parts[1] if len(parts) >= 3 else ""


def session_metrics(ids: list[str], gold: set[str], cutoff: int) -> tuple[bool, bool, float]:
    sessions = {node_session(node_id) for node_id in ids[:cutoff]}
    hits = sessions & gold
    return bool(hits), bool(gold) and gold <= sessions, len(hits) / len(gold) if gold else 0.0


def percentage(value: int, total: int) -> float:
    return round(100.0 * value / total, 2) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    variant_dir = args.run_dir / "hierarchical_state_graph_v2"
    retrieval_by_id = {
        row["question_id"]: row
        for row in read_jsonl(variant_dir / "retrieval_results.jsonl")
    }
    answers_by_id = {
        row["question_id"]: row for row in read_jsonl(variant_dir / "answers.jsonl")
    }
    judge_path = args.run_dir / "mem0_judge" / "auto_eval.jsonl"
    judge_by_id = (
        {
            row["question_id"]: bool(row["correct"])
            for row in read_jsonl(judge_path)
        }
        if judge_path.exists()
        else {}
    )
    question_ids = sorted(set(retrieval_by_id) & set(answers_by_id))

    channel_counts: dict[str, Counter[str]] = defaultdict(Counter)
    channel_unique: dict[str, Counter[str]] = defaultdict(Counter)
    channel_effective: dict[str, Counter[str]] = defaultdict(Counter)
    relation_effective: dict[str, Counter[str]] = defaultdict(Counter)
    graph = Counter()
    operand_graph = Counter()
    per_question: dict[str, dict[str, Any]] = {}

    cutoffs = {"card": 6, "fact": 14, "leaf": 28}
    for question_id in question_ids:
        row = retrieval_by_id[question_id]
        trace = row.get("retrieval_trace") or {}
        gold = set(answers_by_id[question_id].get("answer_session_ids") or [])
        correct = judge_by_id.get(question_id)
        outcome = "correct" if correct else "incorrect"
        detail: dict[str, Any] = {
            "question_id": question_id,
            "query_kind": row.get("query_kind"),
            "correct": correct,
            "gold_sessions": sorted(gold),
            "channels": {},
        }
        for layer in ("card", "fact", "leaf"):
            layer_trace = trace.get(f"{layer}_channels") or {}
            cutoff = cutoffs[layer]
            channel_sessions: dict[str, set[str]] = {}
            for channel in ("semantic", "bm25", "entity"):
                ids = layer_trace.get(f"{channel}_rank_ids") or []
                any_hit, all_hit, recall = session_metrics(ids, gold, cutoff)
                key = f"{layer}.{channel}"
                channel_counts[key]["questions"] += 1
                channel_counts[key]["any_hit"] += int(any_hit)
                channel_counts[key]["all_hit"] += int(all_hit)
                channel_counts[key]["recall_milli"] += round(recall * 1000)
                channel_counts[key][f"{outcome}_any_hit"] += int(any_hit)
                channel_sessions[channel] = {node_session(node_id) for node_id in ids[:cutoff]}
                detail["channels"][key] = {
                    "any_hit": any_hit, "all_hit": all_hit, "recall": recall
                }
            for channel, sessions in channel_sessions.items():
                other = set().union(
                    *(values for name, values in channel_sessions.items() if name != channel)
                )
                unique_gold = (sessions & gold) - other
                channel_unique[f"{layer}.{channel}"]["unique_gold_sessions"] += len(unique_gold)
                channel_unique[f"{layer}.{channel}"]["questions_with_unique_hit"] += int(bool(unique_gold))

        initial = set(trace.get("initial_card_ids") or []) | set(trace.get("initial_fact_ids") or [])
        typed_expanded = set(trace.get("typed_expanded_node_ids") or [])
        operand_expanded = set(trace.get("operand_expanded_fact_ids") or [])
        expanded = typed_expanded | operand_expanded
        postpack = set((trace.get("postpack") or {}).get("card_ids") or []) | set(
            (trace.get("postpack") or {}).get("fact_ids") or []
        )
        operator_sources = set(trace.get("operator_operand_fact_ids") or [])
        novel_operator_sources = (expanded - initial) & operator_sources
        novel_operand_operator_sources = (
            operand_expanded - initial
        ) & operator_sources
        initial_sessions = {node_session(node_id) for node_id in initial}
        expanded_sessions = {node_session(node_id) for node_id in expanded}
        retained = expanded & postpack
        retained_sessions = {node_session(node_id) for node_id in retained}
        rescue_candidate = bool((gold - initial_sessions) & expanded_sessions)
        rescue_retained = bool((gold - initial_sessions) & retained_sessions)
        graph["questions"] += 1
        graph["expanded_nodes"] += len(expanded)
        graph["retained_expanded_nodes"] += len(retained)
        graph["questions_with_retained_expansion"] += int(bool(retained))
        graph["gold_rescue_candidates"] += int(rescue_candidate)
        graph["gold_rescues_retained"] += int(rescue_retained)
        graph[f"{outcome}_questions"] += 1
        graph[f"{outcome}_retained_expanded_nodes"] += len(retained)
        graph["novel_operator_source_facts"] += len(novel_operator_sources)
        graph["questions_with_novel_operator_source"] += int(bool(novel_operator_sources))
        operand_retained = operand_expanded & postpack
        operand_operator_sources = operand_expanded & operator_sources
        operand_graph["questions"] += 1
        operand_graph["expanded_facts"] += len(operand_expanded)
        operand_graph["retained_facts"] += len(operand_retained)
        operand_graph["operator_source_facts"] += len(operand_operator_sources)
        operand_graph["novel_operator_source_facts"] += len(
            novel_operand_operator_sources
        )
        operand_graph["questions_with_expansion"] += int(bool(operand_expanded))
        operand_graph["questions_with_retained_expansion"] += int(bool(operand_retained))
        operand_graph["questions_with_operator_source"] += int(bool(operand_operator_sources))
        operand_graph["questions_with_novel_operator_source"] += int(
            bool(novel_operand_operator_sources)
        )
        for channel, counts in (trace.get("channel_contributions") or {}).items():
            for metric in ("shortlist", "prepack", "postpack", "operator_source"):
                channel_effective[channel][metric] += int(counts.get(metric) or 0)
            channel_effective[channel]["questions"] += 1
            channel_effective[channel]["questions_with_operator_source"] += int(
                bool(counts.get("operator_source"))
            )
        for relation, counts in (trace.get("relation_contributions") or {}).items():
            for metric in ("expanded", "postpack", "operator_source"):
                relation_effective[relation][metric] += int(counts.get(metric) or 0)
            relation_effective[relation]["questions"] += 1
            relation_effective[relation]["questions_with_expansion"] += int(
                bool(counts.get("expanded"))
            )
            relation_effective[relation]["questions_with_postpack"] += int(
                bool(counts.get("postpack"))
            )
            relation_effective[relation]["questions_with_operator_source"] += int(
                bool(counts.get("operator_source"))
            )
        detail["graph"] = {
            "initial_node_count": len(initial),
            "typed_expanded_node_count": len(typed_expanded),
            "operand_expanded_fact_count": len(operand_expanded),
            "expanded_node_count": len(expanded),
            "retained_expanded_node_count": len(retained),
            "operand_retained_fact_count": len(operand_retained),
            "operand_operator_source_fact_count": len(operand_operator_sources),
            "novel_graph_operator_source_fact_count": len(novel_operator_sources),
            "novel_operand_operator_source_fact_count": len(
                novel_operand_operator_sources
            ),
            "gold_rescue_candidate": rescue_candidate,
            "gold_rescue_retained": rescue_retained,
            "expanded_ids": sorted(expanded),
            "retained_expanded_ids": sorted(retained),
            "novel_graph_operator_source_ids": sorted(novel_operator_sources),
        }
        per_question[question_id] = detail

    relation_counts = Counter()
    question_relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for edge in read_jsonl(variant_dir / "edges.jsonl"):
        src = edge.get("src", "")
        dst = edge.get("dst", "")
        question_id = src.split(":", 1)[0]
        if question_id not in per_question:
            continue
        trace = retrieval_by_id[question_id].get("retrieval_trace") or {}
        seeds = set(trace.get("initial_card_ids") or []) | set(trace.get("initial_fact_ids") or [])
        expanded = set(trace.get("typed_expanded_node_ids") or [])
        retained = set(per_question[question_id]["graph"]["retained_expanded_ids"])
        relation = edge.get("relation", "unknown")
        directed = bool(edge.get("directed"))
        if src in seeds and dst in expanded:
            relation_counts[f"{relation}.forward_seed_to_expanded"] += 1
            question_relation_counts[question_id][f"{relation}.forward"] += 1
            if dst in retained:
                relation_counts[f"{relation}.forward_retained"] += 1
        if not directed and dst in seeds and src in expanded:
            relation_counts[f"{relation}.undirected_seed_to_expanded"] += 1
            question_relation_counts[question_id][f"{relation}.undirected"] += 1
            if src in retained:
                relation_counts[f"{relation}.undirected_retained"] += 1
        if directed and dst in seeds and src not in seeds:
            # This is a useful incoming relationship that forward-only traversal
            # cannot follow from the selected seed.
            relation_counts[f"{relation}.blocked_inbound_from_seed"] += 1
            if src in expanded:
                relation_counts[f"{relation}.blocked_but_reached_elsewhere"] += 1
            if node_session(src) in set(per_question[question_id]["gold_sessions"]):
                relation_counts[f"{relation}.blocked_inbound_gold_session"] += 1

    channel_summary: dict[str, Any] = {}
    for key, counts in sorted(channel_counts.items()):
        total = counts["questions"]
        channel_summary[key] = {
            "questions": total,
            "any_session_recall_pct": percentage(counts["any_hit"], total),
            "all_session_recall_pct": percentage(counts["all_hit"], total),
            "mean_session_recall_pct": round(counts["recall_milli"] / max(total, 1) / 10.0, 2),
            "correct_any_hits": counts["correct_any_hit"],
            "incorrect_any_hits": counts["incorrect_any_hit"],
            **channel_unique[key],
        }

    result = {
        "run_dir": str(args.run_dir),
        "question_count": len(question_ids),
        "channel_cutoffs": cutoffs,
        "channels": channel_summary,
        "graph_expansion": {
            **graph,
            "expanded_retention_pct": percentage(
                graph["retained_expanded_nodes"], graph["expanded_nodes"]
            ),
            "questions_with_retained_expansion_pct": percentage(
                graph["questions_with_retained_expansion"], graph["questions"]
            ),
            "gold_rescue_retention_pct": percentage(
                graph["gold_rescues_retained"], graph["gold_rescue_candidates"]
            ),
            "questions_with_novel_operator_source_pct": percentage(
                graph["questions_with_novel_operator_source"], graph["questions"]
            ),
        },
        "operand_graph_expansion": {
            **operand_graph,
            "retention_pct": percentage(
                operand_graph["retained_facts"], operand_graph["expanded_facts"]
            ),
            "operator_source_pct": percentage(
                operand_graph["operator_source_facts"], operand_graph["expanded_facts"]
            ),
            "questions_with_expansion_pct": percentage(
                operand_graph["questions_with_expansion"], operand_graph["questions"]
            ),
            "questions_with_novel_operator_source_pct": percentage(
                operand_graph["questions_with_novel_operator_source"], operand_graph["questions"]
            ),
        },
        "channel_effective_usage": {
            key: dict(value) for key, value in sorted(channel_effective.items())
        },
        "relation_effective_usage": {
            key: dict(value) for key, value in sorted(relation_effective.items())
        },
        "edge_path_counts": dict(relation_counts.most_common()),
        "per_question": list(per_question.values()),
    }
    output = args.output or args.run_dir / "analysis" / "retrieval_channel_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_question"}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
