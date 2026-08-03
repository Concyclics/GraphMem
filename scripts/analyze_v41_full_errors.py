#!/usr/bin/env python3
"""Offline V4.1 error analysis; gold annotations never enter online retrieval."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: object) -> str:
    if isinstance(value, (list, tuple)):
        value = " ".join(map(str, value))
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def pct(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def dist(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": pct(values, 0.5),
        "p95": pct(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def suffix(node_id: str) -> str:
    parts = str(node_id).split(":")
    for index, value in enumerate(parts):
        if value.startswith("session_"):
            return ":".join(parts[index:])
    return str(node_id)


def locomo_gold(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    evidence = set(map(str, case.get("locomo_evidence") or []))
    ids: list[str] = []
    texts: list[str] = []
    for session_id, turns in zip(
        case.get("haystack_session_ids") or [],
        case.get("haystack_sessions") or [],
    ):
        for turn_index, turn in enumerate(turns):
            if str(turn.get("dia_id")) in evidence:
                ids.append(f"{session_id}:turn:{turn_index}")
                texts.append(" ".join((
                    str(turn.get("content") or ""),
                    " ".join(map(str, turn.get("media_captions") or [])),
                )).strip())
    return ids, texts


def gold_channel_ranks(
    channels: dict[str, dict[str, int]], gold: set[str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for node_id, ranks in channels.items():
        if suffix(node_id) not in gold:
            continue
        for channel, rank in ranks.items():
            result[channel] = min(result.get(channel, int(rank)), int(rank))
    return dict(sorted(result.items()))


def _trace_node_ids(value: object) -> set[str]:
    """Collect provenance-shaped node IDs from one trace payload."""
    found: set[str] = set()
    if isinstance(value, str):
        if ":turn:" in value:
            found.add(suffix(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and ":turn:" in key:
                found.add(suffix(key))
            found.update(_trace_node_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_trace_node_ids(item))
    return found


def locomo_retrieval_stage(
    trace: dict[str, Any], gold: set[str], packed: set[str],
) -> tuple[str, dict[str, list[str]]]:
    """Locate the earliest online stage containing annotated evidence."""
    candidate_trace = trace.get("v41_candidate_trace") or {}
    stages = {
        "packed": packed,
        "v41_candidate": {
            suffix(value)
            for value in (candidate_trace.get("channels") or {})
            if ":turn:" in str(value)
        },
        "source_closure": _trace_node_ids(
            trace.get("source_span_closure") or {}
        ),
        "fine_retrieval": {
            suffix(value)
            for value in (trace.get("fine_channels") or {})
            if ":turn:" in str(value)
        },
    }
    matches = {
        name: sorted(gold.intersection(values))
        for name, values in stages.items()
    }
    if gold and gold <= stages["packed"]:
        stage = "answer_with_full_evidence"
    elif gold.intersection(stages["packed"]):
        stage = "partial_evidence_packed"
    elif gold and gold <= stages["v41_candidate"]:
        stage = "candidate_found_not_packed"
    elif gold.intersection(stages["v41_candidate"]):
        stage = "partial_candidate_not_packed"
    elif gold and gold <= stages["source_closure"]:
        stage = "source_closure_found_not_selected"
    elif gold.intersection(stages["source_closure"]):
        stage = "partial_source_closure"
    elif gold and gold <= stages["fine_retrieval"]:
        stage = "fine_found_not_promoted"
    elif gold.intersection(stages["fine_retrieval"]):
        stage = "partial_fine_retrieval"
    else:
        stage = "not_found_by_logged_channels"
    return stage, matches


def grouped(rows_map: dict[str, Counter[str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in sorted(rows_map.items()):
        total = value["total"]
        payload[key] = {
            **dict(value),
            "accuracy": value["correct"] / total if total else 0.0,
        }
    return payload


def token_summary(
    stat_rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    benchmark: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "answer_cache_miss_input_tokens",
        "answer_cache_hit_input_tokens",
        "answer_output_tokens",
        "answer_total_tokens",
        "retrieval_latency_sec",
    ):
        result[field] = dist([float(row.get(field) or 0) for row in stat_rows])
    totals = [int(row.get("answer_total_tokens") or 0) for row in stat_rows]
    result["answer_threshold_counts"] = {
        f"over_{limit // 1000}k": sum(value > limit for value in totals)
        for limit in (10_000, 12_000, 13_000, 15_000)
    }
    builds: dict[str, dict[str, int]] = {}
    for row in stat_rows:
        case = cases[row["question_id"]]
        key = (
            str(case["locomo_sample_id"])
            if benchmark == "locomo"
            else str(row["question_id"])
        )
        candidate = {
            "cache_miss_input": int(row.get("build_cache_miss_input_tokens") or 0),
            "cache_hit_input": int(row.get("build_cache_hit_input_tokens") or 0),
            "output": int(row.get("build_output_tokens") or 0),
            "total": int(row.get("build_total_tokens") or 0),
        }
        if candidate["total"] > (builds.get(key) or {}).get("total", -1):
            builds[key] = candidate
    result["build_memory_count"] = len(builds)
    result["build_by_memory"] = builds
    for field in ("cache_miss_input", "cache_hit_input", "output", "total"):
        result[f"build_{field}"] = dist(
            [float(value[field]) for value in builds.values()]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("lme", "locomo"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    case_rows = json.loads(args.data.read_text())
    cases = {str(row["question_id"]): row for row in case_rows}
    answers = {str(row["question_id"]): row for row in rows(args.run_dir / "answers.jsonl")}
    retrievals = {
        str(row["question_id"]): row
        for row in rows(args.run_dir / "retrieval_results.jsonl")
    }
    judges = {
        str(row["question_id"]): row
        for row in rows(args.judge_dir / "auto_eval.jsonl")
    }
    ids = sorted(set(cases) & set(answers) & set(retrievals) & set(judges))
    if len(ids) != len(answers):
        raise ValueError(
            f"incomplete join cases={len(cases)} answers={len(answers)} "
            f"retrieval={len(retrievals)} judge={len(judges)} common={len(ids)}"
        )

    by_type: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_coverage: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_algebra: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_retrieval_stage: defaultdict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    planner_counts: Counter[str] = Counter()
    certificate_counts: Counter[str] = Counter()
    wrong: list[dict[str, Any]] = []

    for question_id in ids:
        case = cases[question_id]
        answer = answers[question_id]
        retrieval = retrievals[question_id]
        judge = judges[question_id]
        trace = retrieval.get("retrieval_trace") or {}
        correct = bool(judge.get("correct"))
        qtype = str(case.get("question_type") or f"category_{case['locomo_category']}")
        augmentation = trace.get("v41_query_augmentation") or {}
        algebra = str(augmentation.get("answer_algebra") or "unknown")
        certificate = trace.get("v41_evidence_certificate") or {}
        complete = bool(certificate.get("complete"))
        planner = bool(trace.get("planner_applied"))
        for mapping, key in ((by_type, qtype), (by_algebra, algebra)):
            mapping[key]["total"] += 1
            mapping[key]["correct" if correct else "wrong"] += 1
        planner_counts[
            ("planner_" if planner else "deterministic_")
            + ("correct" if correct else "wrong")
        ] += 1
        certificate_counts[
            ("complete_" if complete else "incomplete_")
            + ("correct" if correct else "wrong")
        ] += 1

        packed = {
            suffix(value)
            for value in trace.get("packed_source_turn_ids")
            or retrieval.get("leaf_node_ids")
            or []
        }
        original = {
            suffix(value) for value in trace.get("v41_original_source_ids") or []
        }
        additions = {
            suffix(value) for value in trace.get("v41_source_additions") or []
        }
        context = norm(retrieval.get("context_text") or "")
        candidate_channels = (
            (trace.get("v41_candidate_trace") or {}).get("channels") or {}
        )
        gold: set[str] = set()
        ranks: dict[str, int] = {}

        if args.benchmark == "locomo":
            gold_list, gold_texts = locomo_gold(case)
            gold = set(gold_list)
            matched = gold & packed
            matched |= {
                gold_list[index]
                for index, text in enumerate(gold_texts)
                if norm(text) and norm(text) in context
            }
            coverage = (
                "gold_turns_full" if gold and matched == gold
                else "gold_turns_partial" if matched
                else "gold_turns_none"
            )
            reason = (
                "answer_error_with_full_gold"
                if not correct and coverage == "gold_turns_full"
                else "partial_gold_evidence"
                if not correct and coverage == "gold_turns_partial"
                else "retrieval_missing_gold"
                if not correct
                else "correct"
            )
            ranks = gold_channel_ranks(candidate_channels, gold)
            for channel in ranks:
                channels[channel] += 1
            retrieval_stage, stage_matches = locomo_retrieval_stage(
                trace, gold, packed,
            )
            by_retrieval_stage[retrieval_stage]["total"] += 1
            by_retrieval_stage[retrieval_stage][
                "correct" if correct else "wrong"
            ] += 1
        else:
            gold_sessions = set(map(str, case.get("answer_session_ids") or []))
            retrieved_sessions = set(map(str, retrieval.get("retrieved_session_ids") or []))
            session_full = bool(gold_sessions) and gold_sessions <= retrieved_sessions
            answer_present = bool(norm(case.get("answer"))) and norm(case["answer"]) in context
            coverage = (
                "gold_answer_text_present" if answer_present
                else "gold_session_full_text_missing" if session_full
                else "gold_session_missing"
            )
            reason = (
                "answer_error_with_gold_text"
                if not correct and answer_present
                else "fine_retrieval_or_reasoning"
                if not correct and session_full
                else "coarse_retrieval_missing"
                if not correct
                else "correct"
            )
            retrieval_stage, stage_matches = "not_applicable", {}

        by_coverage[coverage]["total"] += 1
        by_coverage[coverage]["correct" if correct else "wrong"] += 1
        if correct:
            continue
        reasons[reason] += 1
        graph_edges = trace.get("v41_typed_expansion") or []
        wrong.append({
            "question_id": question_id,
            "question_type": qtype,
            "question": case.get("question"),
            "gold_answer": case.get("answer"),
            "prediction": answer.get("prediction"),
            "judge_reasoning": judge.get("reasoning") or judge.get("judge_response"),
            "error_class": reason,
            "evidence_coverage": coverage,
            "gold_evidence": (
                case.get("locomo_evidence")
                if args.benchmark == "locomo"
                else case.get("answer_session_ids")
            ),
            "gold_turn_suffixes": sorted(gold),
            "gold_channel_ranks": ranks,
            "retrieval_stage": retrieval_stage,
            "stage_gold_matches": stage_matches,
            "gold_in_original_sources": bool(gold & original),
            "gold_in_source_additions": bool(gold & additions),
            "gold_touched_by_graph": any(
                suffix(edge.get("src", "")) in gold
                or suffix(edge.get("dst", "")) in gold
                for edge in graph_edges
            ),
            "answer_algebra": algebra,
            "domain_hints": augmentation.get("domain_hints") or [],
            "required_roles": augmentation.get("required_roles") or [],
            "planner_required": bool(trace.get("planner_required")),
            "planner_applied": planner,
            "certificate_complete": complete,
            "missing_roles": certificate.get("missing_roles") or [],
            "packed_source_count": len(packed),
            "source_addition_count": len(additions),
            "typed_expansion_count": len(graph_edges),
            "query_tokens": (
                trace.get("v41_query_token_usage") or {}
            ).get("total_tokens"),
        })

    correct_count = sum(bool(judges[value].get("correct")) for value in ids)
    summary = {
        "benchmark": args.benchmark,
        "offline_gold_only": True,
        "questions": len(ids),
        "correct": correct_count,
        "wrong": len(wrong),
        "accuracy": correct_count / len(ids),
        "by_type": grouped(by_type),
        "by_error_class": dict(reasons),
        "by_evidence_coverage": grouped(by_coverage),
        "by_answer_algebra": grouped(by_algebra),
        "by_retrieval_stage": grouped(by_retrieval_stage),
        "planner": dict(planner_counts),
        "certificate": dict(certificate_counts),
        "gold_channel_presence": dict(channels),
        "tokens": token_summary(
            rows(args.run_dir / "question_stats.jsonl"), cases, args.benchmark
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    with (args.output_dir / "wrong_questions.jsonl").open("w") as handle:
        for row in wrong:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        f"# V4.1 {args.benchmark} full error analysis", "",
        f"- Questions: {len(ids)}",
        f"- Correct: {correct_count}",
        f"- Wrong: {len(wrong)}",
        f"- Accuracy: {correct_count / len(ids):.2%}",
        "- Gold annotations are used offline only.", "",
        "## Error classes", "",
        *[f"- {key}: {value}" for key, value in reasons.most_common()],
        "", "## By type", "",
    ]
    for key, value in summary["by_type"].items():
        lines.append(
            f"- {key}: {value.get('correct', 0)}/{value['total']} "
            f"({value['accuracy']:.2%})"
        )
    lines.extend(["", "## Evidence coverage", ""])
    for key, value in summary["by_evidence_coverage"].items():
        lines.append(
            f"- {key}: total={value['total']}, wrong={value.get('wrong', 0)}, "
            f"accuracy={value['accuracy']:.2%}"
        )
    (args.output_dir / "analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "benchmark": args.benchmark,
        "questions": len(ids),
        "correct": correct_count,
        "wrong": len(wrong),
        "accuracy": correct_count / len(ids),
        "error_classes": dict(reasons),
    }))


if __name__ == "__main__":
    main()
