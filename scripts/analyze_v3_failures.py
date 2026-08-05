#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "my", "of", "on",
    "or", "the", "to", "was", "were", "what", "when", "where", "which", "who",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tokens(value: Any) -> set[str]:
    return {
        token for token in _WORD.findall(str(value).casefold())
        if len(token) > 1 and token not in _STOP
    }


def _support(gold: Any, context: str) -> tuple[float, list[str], list[str]]:
    expected = _tokens(gold)
    present = expected & _tokens(context)
    missing = expected - present
    return (
        len(present) / max(1, len(expected)),
        sorted(present),
        sorted(missing),
    )


def _length(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _locomo_evidence_suffixes(source: dict[str, Any]) -> list[str]:
    suffixes: list[str] = []
    for label in source.get("locomo_evidence") or []:
        match = re.fullmatch(r"D(\d+):(\d+)", str(label).strip())
        if not match:
            continue
        session_number = int(match.group(1))
        turn_index = max(0, int(match.group(2)) - 1)
        suffixes.append(f":session_{session_number}:turn:{turn_index}")
    return list(dict.fromkeys(suffixes))


def _evidence_coverage(
    source: dict[str, Any], retrieval: dict[str, Any]
) -> tuple[int, int, float, bool]:
    expected = _locomo_evidence_suffixes(source)
    if not expected:
        return 0, 0, 0.0, False
    trace = retrieval.get("retrieval_trace") or {}
    selected_ids = {
        str(value)
        for key in ("evidence_leaf_ids", "leaf_node_ids")
        for value in retrieval.get(key) or []
    }
    selected_ids.update(str(value) for value in trace.get("answer_evidence_block_ids") or [])
    found = sum(
        any(node_id.endswith(suffix) for node_id in selected_ids)
        for suffix in expected
    )
    return len(expected), found, found / len(expected), found == len(expected)


def _cause(
    *,
    correct: bool,
    stats: dict[str, Any],
    trace: dict[str, Any],
    support: float,
    prediction: str,
    gold_evidence_count: int,
    gold_evidence_recall: float,
) -> str:
    if correct:
        return "correct"
    if gold_evidence_count:
        if gold_evidence_recall == 0.0:
            return "gold_evidence_postpack_miss"
        if gold_evidence_recall < 1.0:
            return "gold_evidence_partial_pack"
    elif int(stats.get("gold_answer_session_count") or 0) > 0 and not bool(
        stats.get("retrieved_answer_session_all_hit")
    ):
        return "gold_session_retrieval_miss"
    hint = trace.get("catalog_operator_hint") or {}
    if hint.get("complete") and str(hint.get("value", "")).casefold() in prediction.casefold():
        return "complete_operator_wrong"
    if _length(trace.get("graph_rescue_kept_ids")) == 0:
        return "graph_expansion_or_rescue_gap"
    if support >= 0.6:
        return "answer_selection_or_composition_wrong"
    if support <= 0.15:
        return "fact_extraction_or_evidence_gap"
    return "evidence_binding_or_packing_gap"


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "answer_session_all_hit_rate": mean(
            float(row["answer_session_all_hit"]) for row in rows
        ),
        "answer_session_recall_mean": mean(row["answer_session_recall"] for row in rows),
        "gold_lexical_support_mean": mean(row["gold_lexical_support"] for row in rows),
        "gold_evidence_recall_mean": mean(
            row["gold_evidence_recall"] for row in rows if row["gold_evidence_count"]
        ) if any(row["gold_evidence_count"] for row in rows) else None,
        "typed_expansion_steps_mean": mean(row["typed_expansion_steps"] for row in rows),
        "visited_hyperedges_mean": mean(row["visited_hyperedges"] for row in rows),
        "graph_rescue_kept_mean": mean(row["graph_rescue_kept"] for row in rows),
        "closure_complete_rate": mean(float(row["closure_complete"]) for row in rows),
        "answer_tokens_mean": mean(row["answer_total_tokens"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join V3 judge, retrieval, graph-use, and token evidence."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = {
        str(row["question_id"]): row
        for row in json.loads(args.data.read_text(encoding="utf-8"))
    }
    answers = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.run_dir / "answers.jsonl")
    }
    retrievals = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.run_dir / "retrieval_results.jsonl")
    }
    stats = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.run_dir / "question_stats.jsonl")
    }
    judges = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.judge_results)
    }

    audits: list[dict[str, Any]] = []
    pack_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for question_id, judge in judges.items():
        source = data.get(question_id, {})
        answer = answers.get(question_id, {})
        retrieval = retrievals.get(question_id, {})
        question_stats = stats.get(question_id, {})
        trace = retrieval.get("retrieval_trace") or {}
        correct = bool(judge.get("correct"))
        support, present, missing = _support(
            source.get("answer"),
            str(retrieval.get("context_text") or ""),
        )
        evidence_count, evidence_found, evidence_recall, evidence_all_hit = (
            _evidence_coverage(source, retrieval)
        )
        row = {
            "question_id": question_id,
            "question_type": source.get("question_type"),
            "question": source.get("question"),
            "gold_answer": source.get("answer"),
            "prediction": answer.get("prediction"),
            "correct": correct,
            "judge_reason": judge.get("reasoning") or judge.get("judge_response"),
            "answer_session_all_hit": bool(
                question_stats.get("retrieved_answer_session_all_hit")
            ),
            "answer_session_recall": float(
                question_stats.get("retrieved_answer_session_recall") or 0.0
            ),
            "gold_lexical_support": support,
            "gold_evidence_count": evidence_count,
            "gold_evidence_found": evidence_found,
            "gold_evidence_recall": evidence_recall,
            "gold_evidence_all_hit": evidence_all_hit,
            "gold_terms_present": present,
            "gold_terms_missing": missing,
            "typed_expansion_steps": _length(trace.get("expansion_steps")),
            "visited_hyperedges": _length(trace.get("visited_hyperedge_ids")),
            "graph_rescue_candidates": _length(trace.get("graph_rescue_ids")),
            "graph_rescue_kept": _length(trace.get("graph_rescue_kept_ids")),
            "catalog_protected": _length(trace.get("catalog_protected_ids")),
            "catalog_dense_protected": _length(
                trace.get("catalog_dense_protected_ids")
            ),
            "closure_complete": bool(
                (trace.get("closure_certificate") or {}).get("complete")
            ),
            "closure_missing_requirements": list(
                (trace.get("closure_certificate") or {}).get(
                    "missing_requirements"
                ) or []
            ),
            "catalog_operation": (
                (trace.get("catalog_operator_hint") or {}).get("operation")
            ),
            "catalog_operation_complete": bool(
                (trace.get("catalog_operator_hint") or {}).get("complete")
            ),
            "answer_total_tokens": int(
                question_stats.get("answer_total_tokens") or 0
            ),
        }
        row["heuristic_cause"] = _cause(
            correct=correct,
            stats=question_stats,
            trace=trace,
            support=support,
            prediction=str(answer.get("prediction") or ""),
            gold_evidence_count=evidence_count,
            gold_evidence_recall=evidence_recall,
        )
        audits.append(row)
        group = "correct" if correct else "wrong"
        for decision in trace.get("pack_decisions") or []:
            if decision.get("decision") == "keep":
                pack_sources[group][str(decision.get("source") or "unknown")] += 1

    correct_rows = [row for row in audits if row["correct"]]
    wrong_rows = [row for row in audits if not row["correct"]]
    summary = {
        "question_count": len(audits),
        "correct": len(correct_rows),
        "accuracy": len(correct_rows) / max(1, len(audits)),
        "heuristic_cause_counts": dict(Counter(
            row["heuristic_cause"] for row in wrong_rows
        )),
        "correct_metrics": _group_metrics(correct_rows),
        "wrong_metrics": _group_metrics(wrong_rows),
        "pack_kept_sources": {
            group: dict(counter.most_common())
            for group, counter in pack_sources.items()
        },
        "method_note": (
            "Causes are deterministic stage-signal heuristics, not judge labels. "
            "Lexical support is diagnostic only and may undercount paraphrases."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v3_failure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "v3_failure_audit.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "# V3 Failure Audit",
        "",
        f"- Questions: {summary['question_count']}",
        f"- Accuracy: {summary['correct']}/{summary['question_count']} "
        f"({summary['accuracy']:.2%})",
        f"- Heuristic causes: {json.dumps(summary['heuristic_cause_counts'], ensure_ascii=False)}",
        "",
        "| question_id | type | cause | session recall | lexical support | graph kept | operator |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in wrong_rows:
        lines.append(
            f"| {row['question_id']} | {row['question_type']} | "
            f"{row['heuristic_cause']} | {row['answer_session_recall']:.2f} | "
            f"{row['gold_lexical_support']:.2f} | {row['graph_rescue_kept']} | "
            f"{row['catalog_operation'] or ''} |"
        )
    (args.output_dir / "v3_failure_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
