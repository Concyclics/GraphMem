#!/usr/bin/env python3
"""Stage-level audit for a GraphMem run.

Reproduces the structure of runs/<run>/analysis/{stage_audit.jsonl,stage_summary.json}.
Correctness is taken from an LLM judge (auto_eval.jsonl) when available, which is
more reliable than the original token-overlap heuristic; pass --correctness heuristic
to fall back to token overlap.

Each non-abstention wrong answer is attributed to the earliest failing stage:
  retrieval_miss        -> the gold session was not retrieved
  answer_over_abstain   -> retrieval ok but the model refused to answer
  build_summary_degraded-> retrieval ok, model answered, but summaries had parse/truncation errors
  answer_reasoning      -> everything upstream looked fine, the answer is simply wrong
Abstention questions (id endswith _abs) are 'correct' when the model abstains,
otherwise 'answer_should_abstain'.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ABSTAIN_PATTERNS = [
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "unable to determine",
    "could not be determined",
    "not be determined",
    "not mentioned",
    "no mention",
    "does not mention",
    "doesn't mention",
    "not available",
    "not specified",
    "not provided",
    "not stated",
    "not found",
    "no information",
    "not enough information",
    "insufficient",
    "there is no",
    "there's no",
    "cannot recall",
    "cannot suggest",
    "cannot be found",
    "no specific",
    "no record",
    "do not have",
    "don't have",
    "cannot answer",
    "unable to answer",
]

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "your", "you", "my", "i", "it", "this",
    "that", "with", "as", "by", "from", "about", "each", "way", "per",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Variant dir with answers.jsonl / question_stats.jsonl (and optionally auto_eval.jsonl).",
    )
    parser.add_argument("--data", type=Path, required=True, help="Source dataset json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write stage_audit.jsonl / stage_summary.json. Default: <run-dir>/../analysis",
    )
    parser.add_argument(
        "--correctness",
        choices=["judge", "heuristic"],
        default="judge",
    )
    parser.add_argument(
        "--judge-field",
        choices=["strict_correct", "relaxed_correct"],
        default="strict_correct",
    )
    parser.add_argument(
        "--judge-file",
        type=Path,
        default=None,
        help="Override path to auto_eval.jsonl. Default: <run-dir>/auto_eval.jsonl",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokenize(text: Any) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in _STOPWORDS]


def gold_overlap(gold: str, prediction: str) -> float | None:
    gold_tokens = tokenize(gold)
    if not gold_tokens:
        return None
    pred_tokens = set(tokenize(prediction))
    hit = sum(1 for t in gold_tokens if t in pred_tokens)
    return round(hit / len(gold_tokens), 2)


def detect_abstain(prediction: str) -> bool:
    lowered = prediction.lower()
    return any(pat in lowered for pat in ABSTAIN_PATTERNS)


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir
    output_dir = args.output_dir or (run_dir.parent / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = json.loads(args.data.read_text())
    question_type = {str(r["question_id"]): r.get("question_type") for r in source_rows}

    answers = {r["question_id"]: r for r in load_jsonl(run_dir / "answers.jsonl")}
    stats = {r["question_id"]: r for r in load_jsonl(run_dir / "question_stats.jsonl")}

    judge: dict[str, dict[str, Any]] = {}
    judge_file = args.judge_file or (run_dir / "auto_eval.jsonl")
    if args.correctness == "judge":
        if not judge_file.exists():
            raise SystemExit(
                f"--correctness judge requires {judge_file}; run scripts/evaluate_answers.py first "
                f"or pass --correctness heuristic."
            )
        judge = {r["question_id"]: r for r in load_jsonl(judge_file)}

    audit: list[dict[str, Any]] = []
    for qid, ans in answers.items():
        st = stats.get(qid, {})
        is_abstention = qid.endswith("_abs")
        prediction = ans.get("prediction", "")
        gold = ans.get("gold_answer", "")
        pred_abstains = detect_abstain(prediction)

        all_hit = bool(
            ans.get("retrieved_answer_session_all_hit",
                     st.get("retrieved_answer_session_all_hit", False))
        )
        recall = float(
            ans.get("retrieved_answer_session_recall",
                    st.get("retrieved_answer_session_recall", 0.0)) or 0.0
        )
        parse_err = int(st.get("summary_parse_error_count", 0) or 0)
        trunc = int(st.get("summary_truncation_count", 0) or 0)

        if is_abstention:
            correct = pred_abstains
            overlap = None
        elif args.correctness == "judge":
            correct = bool(judge.get(qid, {}).get(args.judge_field, False))
            overlap = gold_overlap(gold, prediction)
        else:
            overlap = gold_overlap(gold, prediction)
            correct = overlap is not None and overlap >= 0.6

        if is_abstention:
            stage = "correct" if pred_abstains else "answer_should_abstain"
        elif correct:
            stage = "correct"
        elif not all_hit:
            stage = "retrieval_miss"
        elif pred_abstains:
            stage = "answer_over_abstain"
        elif parse_err > 0 or trunc > 0:
            stage = "build_summary_degraded"
        else:
            stage = "answer_reasoning"

        row = {
            "question_id": qid,
            "question_type": question_type.get(qid),
            "is_abstention": is_abstention,
            "question": ans.get("question"),
            "gold_answer": gold,
            "prediction": prediction,
            "correct": bool(correct),
            "gold_overlap": overlap,
            "stage": stage,
            "answer_session_all_hit": all_hit,
            "answer_session_recall": recall,
            "retrieved_answer_session_count": st.get("retrieved_answer_session_count"),
            "gold_answer_session_count": st.get("gold_answer_session_count"),
            "summary_parse_error_count": parse_err,
            "summary_truncation_count": trunc,
            "answer_prompt_tokens": st.get("answer_prompt_tokens"),
            "answer_completion_tokens": st.get("answer_completion_tokens"),
            "session_count": st.get("session_count"),
            "prediction_abstains": pred_abstains,
        }
        if judge:
            jrow = judge.get(qid, {})
            row["judge_strict_correct"] = jrow.get("strict_correct")
            row["judge_relaxed_correct"] = jrow.get("relaxed_correct")
            row["judge_error_type"] = jrow.get("error_type")
        audit.append(row)

    order = {r["question_id"]: i for i, r in enumerate(answers.values())}
    audit.sort(key=lambda r: order[r["question_id"]])

    summary = build_summary(audit, args)
    (output_dir / "stage_audit.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in audit) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"wrote {output_dir / 'stage_audit.jsonl'}")
    print(f"wrote {output_dir / 'stage_summary.json'}")


def build_summary(audit: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    answered = len(audit)
    stage_counts = Counter(r["stage"] for r in audit)
    correct = stage_counts.get("correct", 0)

    non_abs = [r for r in audit if not r["is_abstention"]]
    all_hit = sum(1 for r in non_abs if r["answer_session_all_hit"])
    avg_recall = (
        sum(r["answer_session_recall"] for r in non_abs) / len(non_abs)
        if non_abs else 0.0
    )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in audit:
        by_type[str(r["question_type"])].append(r)

    by_type_summary: dict[str, Any] = {}
    for qtype in sorted(by_type):
        rows = by_type[qtype]
        nb = [r for r in rows if not r["is_abstention"]]
        nb_hit = sum(1 for r in nb if r["answer_session_all_hit"])
        c = sum(1 for r in rows if r["stage"] == "correct")
        by_type_summary[qtype] = {
            "n": len(rows),
            "correct": c,
            "acc": round(c / len(rows), 3) if rows else 0.0,
            "retrieval_all_hit_rate(non_abs)": round(nb_hit / len(nb), 3) if nb else None,
            "avg_recall(non_abs)": (
                round(sum(r["answer_session_recall"] for r in nb) / len(nb), 3)
                if nb else None
            ),
            "stage_counts": dict(Counter(r["stage"] for r in rows)),
        }

    return {
        "run_dir": str(args.run_dir),
        "correctness_source": (
            f"llm_judge:{args.judge_field}" if args.correctness == "judge" else "token_overlap>=0.6"
        ),
        "answered": answered,
        "stage_counts": dict(stage_counts),
        "accuracy": round(correct / answered, 3) if answered else 0.0,
        "retrieval_truth": {
            "non_abstention_questions": len(non_abs),
            "retrieval_all_hit": all_hit,
            "retrieval_all_hit_rate": round(all_hit / len(non_abs), 3) if non_abs else 0.0,
            "retrieval_miss_count": len(non_abs) - all_hit,
            "avg_recall": round(avg_recall, 3),
        },
        "build_truth": {
            "questions_with_parse_error": sum(1 for r in audit if r["summary_parse_error_count"] > 0),
            "questions_with_truncation": sum(1 for r in audit if r["summary_truncation_count"] > 0),
            "total_parse_errors": sum(r["summary_parse_error_count"] for r in audit),
            "total_truncations": sum(r["summary_truncation_count"] for r in audit),
        },
        "by_type": by_type_summary,
    }


if __name__ == "__main__":
    main()
