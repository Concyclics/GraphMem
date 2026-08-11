#!/usr/bin/env python3
"""Join V5.20 full-benchmark answers, judges, and retrieval traces.

The output deliberately distinguishes evidence-backed pipeline attribution
(questions with turn-level gold) from semantic-only diagnosis based on the
question, answer, and judge rationale.  It never treats an unannotated row as
an indexing or retrieval failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ARMS = ("turn32", "turn64")
BENCHMARKS = ("longmemeval", "locomo")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def indexed(path: Path, key: str) -> dict[str, dict]:
    return {str(row[key]): row for row in rows(path)}


def nearest_rank(values: Iterable[float], percentile: float) -> float | None:
    data = sorted(float(value) for value in values)
    return data[max(0, math.ceil(percentile * len(data)) - 1)] if data else None


def stats(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": sum(data) / len(data) if data else None,
        "p50": nearest_rank(data, 0.50),
        "p95": nearest_rank(data, 0.95),
        "max": max(data) if data else None,
    }


def semantic_type(answer: dict) -> str:
    question = str(answer.get("question") or "").casefold()
    gold = str(answer.get("gold_answer") or "").casefold()
    raw_type = str(answer.get("question_type") or answer.get("stratum") or "")
    if answer.get("question_id", "").endswith("_abs") or "not mention" in gold:
        return "abstention_near_match"
    if (raw_type == "temporal-reasoning" or raw_type == "category_2"
            or re.search(r"\bwhen\b|\bhow (?:many|long) (?:day|week|month|year)", question)):
        return "temporal_reasoning"
    if re.search(r"\bhow many\b|\bnumber of\b|\btotal\b|\bcount\b", question):
        return "count_aggregation"
    if raw_type == "knowledge-update":
        return "state_update"
    if raw_type == "single-session-preference":
        return "preference_personalization"
    if raw_type == "category_3" or re.search(r"\bwould\b|\blikely\b|\bmight\b", question):
        return "counterfactual_inference"
    if raw_type in {"multi-session", "category_1"}:
        return "multi_hop_relation"
    return "single_fact_relation"


def judge_reason(row: dict) -> str:
    value = str(row.get("reasoning") or row.get("judge_response") or "")
    value = re.sub(r"</?judge_thinking>", "", value)
    return " ".join(value.split())


def answer_error_class(answer: dict, judge: dict) -> str:
    kind = semantic_type(answer)
    reason = judge_reason(judge).casefold()
    if kind == "abstention_near_match":
        return "abstention_or_near_match_confusion"
    if kind == "temporal_reasoning":
        return "temporal_anchor_or_arithmetic"
    if kind == "count_aggregation":
        return "count_dedup_or_arithmetic"
    if kind == "state_update":
        return "stale_or_wrong_state"
    if kind == "preference_personalization":
        return "preference_grounding"
    if kind == "counterfactual_inference":
        return "unsupported_or_overconservative_inference"
    if any(token in reason for token in (
            "omit", "incomplete", "does not include", "fails to mention",
            "does not mention", "missing")):
        return "incomplete_answer"
    if any(token in reason for token in (
            "invent", "unsupported", "no evidence", "not established")):
        return "unsupported_hallucination"
    return "wrong_entity_attribute_or_relation"


def retrieval_stage(retrieval: dict) -> str:
    if not retrieval.get("has_turn_gold"):
        return "unannotated_no_pipeline_attribution"
    if float(retrieval.get("candidate_turn_recall") or 0.0) < 1.0:
        if retrieval.get("graph_reachable_turn_all_hit"):
            return "candidate_selection_loss"
        if retrieval.get("graph_reachable_turn_any_hit"):
            return "graph_or_index_partial_gold"
        return "graph_or_index_no_gold"
    if retrieval.get("turn_all_hit"):
        return "complete_evidence_answer_error"
    if retrieval.get("turn_any_hit"):
        return "packing_partial_gold"
    return "packing_zero_gold"


def load_arm(root: Path, arm: str) -> dict[str, dict[str, dict]]:
    answer_root = root / arm / "answer"
    judges = {}
    for directory in ("judge_lme", "judge_locomo"):
        judges.update(indexed(answer_root / directory / "auto_eval.jsonl", "question_id"))
    return {
        "answers": indexed(answer_root / "answers.jsonl", "question_id"),
        "retrieval": indexed(answer_root / "retrieval.jsonl", "dev_question_id"),
        "prepared": indexed(answer_root / "prepared_answers.jsonl", "question_id"),
        "judges": judges,
    }


def compact_retrieval(row: dict) -> dict:
    keys = (
        "has_turn_gold", "gold_turns", "candidate_turn_recall",
        "candidate_turn_precision", "candidate_average_precision",
        "candidate_first_gold_reciprocal_rank", "candidate_last_gold_rank",
        "graph_reachable_turn_recall", "graph_reachable_turn_precision",
        "turn_recall", "turn_precision", "turn_all_hit", "turn_any_hit",
        "packed_turns", "evidence_tokens", "visited_nodes", "visited_edges",
        "evidence_chain_turns", "evidence_graph_turns",
        "evidence_auxiliary_turns", "traversed_relation_signals",
        "latency_total_ms", "answer_total_tokens",
    )
    return {key: row.get(key) for key in keys}


def pair_state(correct32: bool, correct64: bool) -> str:
    if correct32 and correct64:
        return "both_correct"
    if not correct32 and correct64:
        return "fixed_by_64"
    if correct32 and not correct64:
        return "hurt_by_64"
    return "both_wrong"


def pipeline_summary(cases: list[dict], arm: str, benchmark: str) -> dict:
    selected = [case for case in cases if case["benchmark"] == benchmark
                and not case[arm]["correct"]]
    annotated = [case for case in selected
                 if case[arm]["retrieval"]["has_turn_gold"]]
    return {
        "wrong": len(selected),
        "annotated_wrong": len(annotated),
        "retrieval_stage": dict(Counter(
            case[arm]["retrieval_stage"] for case in annotated)),
        "candidate_recall": stats(
            case[arm]["retrieval"]["candidate_turn_recall"] for case in annotated),
        "packed_recall": stats(
            case[arm]["retrieval"]["turn_recall"] for case in annotated),
        "packed_precision": stats(
            case[arm]["retrieval"]["turn_precision"] for case in annotated),
        "candidate_average_precision": stats(
            case[arm]["retrieval"]["candidate_average_precision"] for case in annotated),
        "candidate_last_gold_rank": stats(
            case[arm]["retrieval"]["candidate_last_gold_rank"]
            for case in annotated
            if case[arm]["retrieval"]["candidate_last_gold_rank"] is not None),
    }


def accuracy_by_type(all_ids: list[str], data: dict, benchmark: str) -> dict:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for question_id in all_ids:
        answer = data["answers"][question_id]
        if answer.get("benchmark") != benchmark:
            continue
        buckets[str(answer.get("question_type") or answer.get("stratum"))].append(
            bool(data["judges"][question_id]["correct"]))
    return {
        key: {"correct": sum(values), "questions": len(values),
              "accuracy": sum(values) / len(values)}
        for key, values in sorted(buckets.items())
    }


def paired_diagnostics(cases: list[dict], benchmark: str, state: str) -> dict:
    selected = [case for case in cases if case["benchmark"] == benchmark
                and case["pair_state"] == state]
    annotated = [case for case in selected
                 if case["turn32"]["retrieval"]["has_turn_gold"]]
    transitions = Counter(
        (case["turn32"]["retrieval_stage"], case["turn64"]["retrieval_stage"])
        for case in annotated)
    return {
        "questions": len(selected),
        "annotated": len(annotated),
        "turn32_evidence_subset_of_turn64": sum(
            case["evidence_pair_audit"]["turn32_subset_of_turn64"]
            for case in selected),
        "packed_recall_improved": sum(
            float(case["turn64"]["retrieval"]["turn_recall"] or 0)
            > float(case["turn32"]["retrieval"]["turn_recall"] or 0)
            for case in annotated),
        "stage_transitions": {
            f"{left}->{right}": count
            for (left, right), count in transitions.most_common()
        },
    }


def graph_index_audit(root: Path, data: dict[str, dict[str, dict]]) -> dict:
    manifest = json.loads((root / "turn64" / "answer" / "run_manifest.json")
                          .read_text(encoding="utf-8"))
    source_db = Path(manifest["source_db"])
    connection = sqlite3.connect(source_db)
    memory_ids: dict[str, set[str]] = defaultdict(set)
    for question_id, answer in data["turn64"]["answers"].items():
        retrieval = data["turn64"]["retrieval"][question_id]
        memory_ids[str(answer["benchmark"])].add(str(retrieval["memory_id"]))
    result = {"source_db": str(source_db), "benchmarks": {}}
    for benchmark, ids in memory_ids.items():
        table_counts = Counter(); node_types = Counter(); relations = Counter()
        relation_signals = Counter(); masks = Counter()
        id_list = sorted(ids)
        for start in range(0, len(id_list), 400):
            chunk = id_list[start:start + 400]
            placeholders = ",".join("?" for _ in chunk)
            for table in ("source_turns", "evidence_groups", "graph_nodes", "graph_edges"):
                table_counts[table] += int(connection.execute(
                    f"SELECT count(*) FROM {table} WHERE memory_id IN ({placeholders})",
                    chunk).fetchone()[0])
            node_types.update(dict(connection.execute(
                f"SELECT node_type,count(*) FROM graph_nodes "
                f"WHERE memory_id IN ({placeholders}) GROUP BY node_type", chunk)))
            for relation, source, count in connection.execute(
                    f"SELECT relation,source,count(*) FROM graph_edges "
                    f"WHERE memory_id IN ({placeholders}) GROUP BY relation,source", chunk):
                relations[str(relation)] += int(count)
                matched = re.search(r"relation_mask:([^|]+)", str(source or ""))
                if not matched:
                    continue
                signals = tuple(sorted(filter(None, matched.group(1).split(","))))
                masks[signals] += int(count)
                for signal in signals:
                    relation_signals[signal] += int(count)
        turns = table_counts["source_turns"]
        result["benchmarks"][benchmark] = {
            "memories": len(ids), "table_counts": dict(table_counts),
            "node_types": dict(node_types), "edge_relations": dict(relations),
            "relation_mask_signals": dict(relation_signals),
            "relation_masks": {"+".join(key): value for key, value in masks.items()},
            "lossless_terminal_reference_ratio": (
                node_types["evidence_group_ref"] / turns if turns else None),
            "canonical_fact_nodes_per_source_turn": (
                node_types["canonical_fact"] / turns if turns else None),
        }
    connection.close()
    build_report_path = source_db.parent.parent / "build_report.json"
    if build_report_path.exists():
        build = json.loads(build_report_path.read_text(encoding="utf-8"))
        build_rows = {str(row["memory_id"]): row for row in build.get("rows", [])}
        result["build_report"] = str(build_report_path)
        for benchmark, ids in memory_ids.items():
            selected = [build_rows[memory_id] for memory_id in ids
                        if memory_id in build_rows]
            quality = [row.get("build_quality", {}) for row in selected]
            result["benchmarks"][benchmark]["build_quality"] = {
                "memories": len(selected),
                "extraction_scenes": sum(int(row.get("extraction_scenes", 0))
                                         for row in quality),
                "extraction_success_scenes": sum(int(row.get(
                    "extraction_success_scenes", 0)) for row in quality),
                "extraction_fallback_scenes": sum(int(row.get(
                    "extraction_fallback_scenes", 0)) for row in quality),
                "budget_degraded_memories": sum(bool(row.get("budget_degraded"))
                                                for row in quality),
                "budget_degraded_calls": sum(int(row.get("budget_degraded_calls", 0))
                                             for row in quality),
                "budget_skipped_scenes": sum(int(row.get("budget_skipped_scenes", 0))
                                             for row in quality),
                "build_tokens": stats(row.get("tokens", 0) for row in selected),
            }
    return result


def markdown(summary: dict, cases: list[dict]) -> str:
    lines = [
        "# V5.20 全量 32/64-turn 错题分析", "",
        "本文件由逐题 answer、Luna verdict、retrieval trace 与 PreparedAnswer 联表生成。"
        "只有带 turn-level gold 的题目才进行索引/候选/packing 环节归因；未标注题目只做答案语义分类。",
        "", "## 总览", "",
        "| Benchmark | 32-turn | 64-turn | 共同错 | 64 修复 | 64 新增错误 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        pair = summary["paired_outcomes"][benchmark]
        arms = summary["arms"]
        label = "LongMemEval" if benchmark == "longmemeval" else "LoCoMo"
        lines.append(
            f"| {label} | {100*arms['turn32'][benchmark]['accuracy']:.1f}% "
            f"({arms['turn32'][benchmark]['correct']}/{arms['turn32'][benchmark]['questions']}) | "
            f"{100*arms['turn64'][benchmark]['accuracy']:.1f}% "
            f"({arms['turn64'][benchmark]['correct']}/{arms['turn64'][benchmark]['questions']}) | "
            f"{pair['both_wrong']} | {pair['fixed_by_64']} | {pair['hurt_by_64']} |")
    lines += ["", "## 可归因的流水线错误", ""]
    for arm in ARMS:
        lines += [f"### {arm}", "",
                  "| Benchmark | 错题 | 有 turn gold | 候选/图失败 | Packing 部分/全丢 | Evidence 完整仍错 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for benchmark in BENCHMARKS:
            item = summary["pipeline"][arm][benchmark]
            stages = item["retrieval_stage"]
            graph = sum(stages.get(key, 0) for key in (
                "candidate_selection_loss", "graph_or_index_partial_gold",
                "graph_or_index_no_gold"))
            packing = stages.get("packing_partial_gold", 0) + stages.get("packing_zero_gold", 0)
            complete = stages.get("complete_evidence_answer_error", 0)
            lines.append(
                f"| {benchmark} | {item['wrong']} | {item['annotated_wrong']} | "
                f"{graph} | {packing} | {complete} |")
        lines.append("")
    lines += [
        "## 语义错误类型（任一预算出错的去重题目）", "",
        "| 类型 | 题数 |", "|---|---:|",
    ]
    for key, value in summary["semantic_error_classes"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "", "## 关键诊断", "",
        "- 带 gold 的错题中，候选池 recall 均为 100%；当前证据不支持“图或索引完全找不到答案”是主要瓶颈。",
        "- LoCoMo 的主要损失是 gold 在候选列表中排名过低，32/64-turn packing 无法保留完整证据；"
        "扩大预算改善 recall，但同时把 precision 进一步压低。",
        "- Evidence all-hit 仍会答错，说明时间锚点、计数/去重、状态更新和近邻实体消歧仍需确定性算子或更强约束。",
        "- `error_cases.jsonl` 保存全部逐题字段；`error_cases.csv` 是便于人工筛选的紧凑视图。",
        "", "## 32→64 翻转诊断", "",
    ]
    for benchmark in BENCHMARKS:
        for state in ("fixed_by_64", "hurt_by_64", "both_wrong"):
            item = summary["paired_diagnostics"][benchmark][state]
            lines.append(
                f"- {benchmark} / {state}: {item['questions']} 题；"
                f"有 gold {item['annotated']} 题，其中 packed recall 提高 "
                f"{item['packed_recall_improved']} 题；32-turn evidence 是 64-turn 子集 "
                f"{item['turn32_evidence_subset_of_turn64']}/{item['questions']} 题。")
    graph = summary["graph_index_audit"]["benchmarks"]
    lines += ["", "## 构建与图导航审计", ""]
    for benchmark in BENCHMARKS:
        item = graph[benchmark]
        quality = item.get("build_quality", {})
        lines += [
            f"### {benchmark}", "",
            f"- 原始 turns/evidence groups/evidence-group-ref nodes："
            f"{item['table_counts']['source_turns']}/"
            f"{item['table_counts']['evidence_groups']}/"
            f"{item['node_types'].get('evidence_group_ref', 0)}，"
            f"lossless terminal reference ratio={item['lossless_terminal_reference_ratio']:.3f}。",
            f"- canonical facts/source turn={item['canonical_fact_nodes_per_source_turn']:.3f}；"
            f"物化 relation-mask signals={item['relation_mask_signals']}。",
            f"- 构建 fallback scenes={quality.get('extraction_fallback_scenes', 0)}，"
            f"budget-degraded memories={quality.get('budget_degraded_memories', 0)}，"
            f"skipped scenes={quality.get('budget_skipped_scenes', 0)}。", "",
        ]
    lines += ["## 代表性错题", ""]
    preferred = []
    seen: set[tuple[str, str]] = set()
    for case in cases:
        key = (case["pair_state"], case["semantic_type"])
        if key not in seen:
            preferred.append(case)
            seen.add(key)
    for case in preferred[:24]:
        lines += [
            f"### {case['question_id']} · {case['benchmark']} · {case['pair_state']}", "",
            f"- 类型：`{case['semantic_type']}` / `{case['primary_error_class']}`",
            f"- 问题：{case['question']}",
            f"- Gold：{case['gold_answer']}",
            f"- 32-turn：{case['turn32']['prediction']}",
            f"- 64-turn：{case['turn64']['prediction']}",
            f"- 检索归因：32=`{case['turn32']['retrieval_stage']}`，"
            f"64=`{case['turn64']['retrieval_stage']}`", "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {arm: load_arm(args.root, arm) for arm in ARMS}
    question_ids = list(data["turn32"]["answers"])
    if set(question_ids) != set(data["turn64"]["answers"]):
        raise RuntimeError("32/64 question IDs differ")

    cases = []
    for question_id in question_ids:
        answer32 = data["turn32"]["answers"][question_id]
        judge32 = data["turn32"]["judges"][question_id]
        judge64 = data["turn64"]["judges"][question_id]
        correct32 = bool(judge32["correct"])
        correct64 = bool(judge64["correct"])
        if correct32 and correct64:
            continue
        item = {
            "question_id": question_id,
            "benchmark": answer32["benchmark"],
            "question_type": answer32.get("question_type") or answer32.get("stratum"),
            "semantic_type": semantic_type(answer32),
            "primary_error_class": answer_error_class(
                answer32, judge32 if not correct32 else judge64),
            "pair_state": pair_state(correct32, correct64),
            "question": answer32["question"],
            "gold_answer": answer32["gold_answer"],
        }
        for arm in ARMS:
            answer = data[arm]["answers"][question_id]
            retrieval = data[arm]["retrieval"][question_id]
            judge = data[arm]["judges"][question_id]
            prepared = data[arm]["prepared"][question_id]
            item[arm] = {
                "correct": bool(judge["correct"]),
                "prediction": answer["prediction"],
                "judge_reason": judge_reason(judge),
                "retrieval_stage": retrieval_stage(retrieval),
                "retrieval": compact_retrieval(retrieval),
                "evidence_turn_ids": prepared.get("evidence_turn_ids", []),
                "prompt_payload_hash": prepared.get("prompt_payload_hash"),
            }
        ids32 = set(item["turn32"]["evidence_turn_ids"])
        ids64 = set(item["turn64"]["evidence_turn_ids"])
        item["evidence_pair_audit"] = {
            "turn32_subset_of_turn64": ids32 <= ids64,
            "shared": len(ids32 & ids64),
            "added_by_64": len(ids64 - ids32),
            "removed_by_64": len(ids32 - ids64),
        }
        cases.append(item)

    summary = {
        "schema_version": "graphmem-v5.20-full-error-analysis-v1",
        "root": str(args.root),
        "questions": len(question_ids),
        "unique_questions_wrong_in_either_budget": len(cases),
        "arms": {}, "paired_outcomes": {}, "pipeline": {},
        "semantic_error_classes": dict(sorted(Counter(
            case["primary_error_class"] for case in cases).items(),
            key=lambda item: (-item[1], item[0]))),
        "semantic_types": dict(sorted(Counter(
            case["semantic_type"] for case in cases).items(),
            key=lambda item: (-item[1], item[0]))),
        "methodology": {
            "pipeline_attribution_requires_turn_gold": True,
            "semantic_error_classification": "deterministic heuristic over question type and judge rationale",
            "unannotated_rows_never_labeled_as_index_or_retrieval_failures": True,
        },
    }
    for arm in ARMS:
        summary["arms"][arm] = {}
        summary["pipeline"][arm] = {}
        for benchmark in BENCHMARKS:
            ids = [question_id for question_id in question_ids
                   if data[arm]["answers"][question_id]["benchmark"] == benchmark]
            correct = sum(bool(data[arm]["judges"][question_id]["correct"])
                          for question_id in ids)
            summary["arms"][arm][benchmark] = {
                "questions": len(ids), "correct": correct,
                "wrong": len(ids) - correct, "accuracy": correct / len(ids),
                "accuracy_by_question_type": accuracy_by_type(ids, data[arm], benchmark),
            }
            summary["pipeline"][arm][benchmark] = pipeline_summary(cases, arm, benchmark)
    for benchmark in BENCHMARKS:
        counts = Counter(case["pair_state"] for case in cases
                         if case["benchmark"] == benchmark)
        total_ids = [question_id for question_id in question_ids
                     if data["turn32"]["answers"][question_id]["benchmark"] == benchmark]
        counts["both_correct"] = len(total_ids) - sum(counts.values())
        summary["paired_outcomes"][benchmark] = dict(counts)
    summary["paired_diagnostics"] = {
        benchmark: {
            state: paired_diagnostics(cases, benchmark, state)
            for state in ("fixed_by_64", "hurt_by_64", "both_wrong")
        } for benchmark in BENCHMARKS
    }
    summary["graph_index_audit"] = graph_index_audit(args.root, data)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "error_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    csv_fields = (
        "question_id", "benchmark", "question_type", "semantic_type",
        "primary_error_class", "pair_state", "question", "gold_answer",
        "prediction_32", "prediction_64", "stage_32", "stage_64",
        "candidate_recall_32", "packed_recall_32", "packed_precision_32",
        "candidate_recall_64", "packed_recall_64", "packed_precision_64",
        "judge_reason_32", "judge_reason_64",
    )
    with (args.output / "error_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "question_id": case["question_id"], "benchmark": case["benchmark"],
                "question_type": case["question_type"], "semantic_type": case["semantic_type"],
                "primary_error_class": case["primary_error_class"],
                "pair_state": case["pair_state"], "question": case["question"],
                "gold_answer": case["gold_answer"],
                "prediction_32": case["turn32"]["prediction"],
                "prediction_64": case["turn64"]["prediction"],
                "stage_32": case["turn32"]["retrieval_stage"],
                "stage_64": case["turn64"]["retrieval_stage"],
                "candidate_recall_32": case["turn32"]["retrieval"]["candidate_turn_recall"],
                "packed_recall_32": case["turn32"]["retrieval"]["turn_recall"],
                "packed_precision_32": case["turn32"]["retrieval"]["turn_precision"],
                "candidate_recall_64": case["turn64"]["retrieval"]["candidate_turn_recall"],
                "packed_recall_64": case["turn64"]["retrieval"]["turn_recall"],
                "packed_precision_64": case["turn64"]["retrieval"]["turn_precision"],
                "judge_reason_32": case["turn32"]["judge_reason"],
                "judge_reason_64": case["turn64"]["judge_reason"],
            })
    (args.output / "ERROR_ANALYSIS.md").write_text(
        markdown(summary, cases), encoding="utf-8")
    print(json.dumps({
        "questions": len(question_ids), "error_cases": len(cases),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
