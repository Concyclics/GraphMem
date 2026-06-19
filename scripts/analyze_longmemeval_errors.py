#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LongMemEval official-judge errors with retrieval and budget context."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--question-stats", type=Path, required=True)
    parser.add_argument("--nodes", type=Path)
    parser.add_argument("--generic-ops", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def content_terms(value: Any) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "how",
        "many",
        "much",
        "have",
        "had",
        "did",
        "does",
        "was",
        "were",
        "are",
        "you",
        "your",
        "my",
        "me",
        "i",
        "last",
        "current",
        "currently",
        "total",
        "answer",
        "final",
        "evidence",
        "facts",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]+", normalize_text(value))
        if len(token) > 2 and token not in stop
    }


def gold_lexical_overlap(gold: Any, context: str) -> float | None:
    terms = content_terms(gold)
    if not terms:
        return None
    lowered = normalize_text(context)
    return sum(term in lowered for term in terms) / len(terms)


def operator_gap(question: str, question_type: str) -> str | None:
    q = question.lower()
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", q):
        return "count_operator_gap"
    if re.search(r"\bhow much\b|\btotal\b|\bsum\b|\bcombined\b|\bspent\b|\bcost\b", q):
        return "quantity_operator_gap"
    if re.search(r"\bhow long\b|\bhow many (?:days|weeks|months|years)\b|\bwhen\b|\btime\b|\bago\b", q):
        return "temporal_operator_gap"
    if re.search(r"\bcurrent(?:ly)?\b|\blatest\b|\bnow\b|\bstill\b", q):
        return "current_state_operator_gap"
    if question_type == "single-session-preference":
        return "preference_operator_gap"
    if re.search(r"\bprevious\b|\bearlier\b|\blast time\b|\bdecided\b|\bchosen\b", q):
        return "previous_chat_operator_gap"
    return None


def prediction_abstains(text: str) -> bool:
    return bool(
        re.search(
            r"insufficient|not enough|cannot determine|not mentioned|no evidence|unknown|"
            r"can't determine|do not know|does not mention",
            text,
            flags=re.IGNORECASE,
        )
    )


def classify_cause(
    *,
    correct: bool,
    case: dict[str, Any],
    answer: dict[str, Any],
    retrieval: dict[str, Any],
    stats: dict[str, Any],
    op_row: dict[str, Any] | None,
    context_overlap: float | None,
) -> str:
    if correct:
        return "correct"
    if not retrieval.get("answer_session_all_hit"):
        return "retrieval_session_miss"
    qa_tokens = int(stats.get("answer_prompt_tokens") or 0) + int(
        stats.get("answer_completion_tokens") or 0
    )
    if qa_tokens > 10000:
        return "qa_budget_overrun"
    if op_row is not None:
        return "generic_operator_wrong"
    gold = str(case.get("answer") or "")
    prediction = str(answer.get("prediction") or "")
    if gold.strip() and prediction_abstains(prediction):
        return "over_abstain"
    if context_overlap is not None and context_overlap < 0.5:
        return "retrieval_context_missing_gold_terms"
    gap = operator_gap(str(case.get("question") or ""), str(case.get("question_type") or ""))
    if gap is not None:
        return gap
    return "generation_or_reasoning_error"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
        "wrong_count": sum(not row["correct"] for row in rows),
        "cause_counts": Counter(row["cause"] for row in rows if not row["correct"]),
        "by_type": {},
        "operator_wrong_counts": Counter(
            row.get("generic_memory_op") or "none"
            for row in rows
            if not row["correct"] and row.get("generic_memory_op_applied")
        ),
    }
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["question_type"])].append(row)
    for question_type, type_rows in sorted(by_type.items()):
        wrong_rows = [row for row in type_rows if not row["correct"]]
        summary["by_type"][question_type] = {
            "rows": len(type_rows),
            "correct": sum(row["correct"] for row in type_rows),
            "accuracy": sum(row["correct"] for row in type_rows) / len(type_rows),
            "wrong": len(wrong_rows),
            "answer_session_all_hit_rate": (
                sum(row["answer_session_all_hit"] for row in type_rows) / len(type_rows)
            ),
            "avg_answer_session_recall": (
                sum(float(row.get("answer_session_recall") or 0.0) for row in type_rows)
                / len(type_rows)
            ),
            "qa_over_10k": sum(row["qa_tokens"] > 10000 for row in type_rows),
            "wrong_cause_counts": Counter(row["cause"] for row in wrong_rows),
        }
    return summary


def json_default(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    raise TypeError(type(value))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = {str(row["question_id"]): row for row in read_json_or_jsonl(args.data)}
    answers = {str(row["question_id"]): row for row in read_jsonl(args.answers)}
    evals = {str(row["question_id"]): row for row in read_json_or_jsonl(args.eval_results)}
    retrievals = {str(row["question_id"]): row for row in read_jsonl(args.retrieval_results)}
    question_stats = {str(row["question_id"]): row for row in read_jsonl(args.question_stats)}
    ops = {str(row["question_id"]): row for row in read_jsonl(args.generic_ops)}

    rows: list[dict[str, Any]] = []
    for qid, case in cases.items():
        answer = answers.get(qid, {})
        eval_row = evals.get(qid, {})
        retrieval = retrievals.get(qid, {})
        stats = question_stats.get(qid, {})
        op_row = ops.get(qid)
        correct = bool(eval_row.get("autoeval_label", {}).get("label"))
        context_overlap = gold_lexical_overlap(case.get("answer"), str(retrieval.get("context_text") or ""))
        qa_tokens = int(stats.get("answer_prompt_tokens") or 0) + int(
            stats.get("answer_completion_tokens") or 0
        )
        cause = classify_cause(
            correct=correct,
            case=case,
            answer=answer,
            retrieval=retrieval,
            stats=stats,
            op_row=op_row,
            context_overlap=context_overlap,
        )
        rows.append(
            {
                "question_id": qid,
                "question_type": case.get("question_type"),
                "question": case.get("question"),
                "gold_answer": case.get("answer"),
                "correct": correct,
                "cause": cause,
                "answer_session_all_hit": bool(retrieval.get("answer_session_all_hit")),
                "answer_session_recall": retrieval.get("answer_session_recall"),
                "retrieved_answer_session_count": retrieval.get("retrieved_answer_session_count"),
                "gold_answer_session_count": retrieval.get("gold_answer_session_count"),
                "gold_lexical_context_overlap": context_overlap,
                "qa_tokens": qa_tokens,
                "build_tokens": int(stats.get("build_prompt_tokens") or 0)
                + int(stats.get("build_completion_tokens") or 0),
                "generic_memory_op_applied": bool(
                    answer.get("generic_memory_op_applied") or op_row is not None
                ),
                "generic_memory_op": (
                    answer.get("generic_memory_op")
                    or (op_row or {}).get("operator")
                ),
                "prediction_abstains": prediction_abstains(str(answer.get("prediction") or "")),
                "prediction_preview": " ".join(str(answer.get("prediction") or "").split())[:360],
            }
        )

    summary = summarize(rows)
    (args.output_dir / "longmemeval_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    with (args.output_dir / "longmemeval_error_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
