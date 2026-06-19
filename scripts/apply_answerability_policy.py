#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from answerability_filter_from_retrieval import render_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a conservative answerability policy to saved classifier outputs."
    )
    parser.add_argument("--original-answers", type=Path, required=True)
    parser.add_argument("--answerability-answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="answerability_policy")
    parser.add_argument(
        "--decision-policy",
        choices=["all", "abstain_only"],
        default="abstain_only",
    )
    parser.add_argument(
        "--judge-compat-ref",
        type=Path,
        help="Optional reference file whose question ids should be used in the hypothesis.",
    )
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_qid(qid: str) -> str:
    return qid.removesuffix("_abs")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    originals = {str(row["question_id"]): row for row in read_json_or_jsonl(args.original_answers)}
    answerability = {
        str(row["question_id"]): row for row in read_json_or_jsonl(args.answerability_answers)
    }
    compat_ids: dict[str, str] = {}
    if args.judge_compat_ref:
        for row in read_json_or_jsonl(args.judge_compat_ref):
            qid = str(row["question_id"])
            compat_ids[normalize_qid(qid)] = qid

    output_rows: list[dict[str, Any]] = []
    for qid, original in originals.items():
        classified = answerability.get(qid)
        row = dict(original)
        row["variant"] = args.variant
        if classified is None:
            decision = "missing"
            effective_decision = "keep_original"
            prediction = str(original.get("prediction") or "")
            reason = "missing_answerability_row"
            final_answer = ""
        else:
            decision = str(classified.get("answerability_decision") or "keep")
            reason = str(classified.get("answerability_reason") or "")
            final_answer = str(classified.get("answerability_final_answer") or "")
            if decision in {"keep", "skipped", "missing"}:
                prediction = str(original.get("prediction") or "")
                effective_decision = "keep_original"
            else:
                prediction = render_prediction(
                    str(original.get("prediction") or ""),
                    {
                        "decision": decision,
                        "reason": reason,
                        "final_answer": final_answer,
                    },
                    policy=args.decision_policy,
                )
                effective_decision = (
                    decision
                    if args.decision_policy == "all" or decision == "abstain"
                    else "keep_original"
                )
        row["prediction"] = prediction
        row["answerability_decision"] = decision
        row["answerability_effective_decision"] = effective_decision
        row["answerability_reason"] = reason
        row["answerability_final_answer"] = final_answer
        row["answerability_policy"] = args.decision_policy
        output_rows.append(row)

    order = {qid: index for index, qid in enumerate(originals)}
    output_rows.sort(key=lambda row: order[str(row["question_id"])])
    write_jsonl(args.output_dir / "answers.jsonl", output_rows)

    hypothesis = [
        {
            "question_id": compat_ids.get(str(row["question_id"]), str(row["question_id"])),
            "hypothesis": row.get("prediction", ""),
            "original_question_id": row["question_id"],
            "original_question_type": row.get("question_type"),
        }
        for row in output_rows
    ]
    hypothesis_path = args.output_dir / f"{args.variant}_hypothesis.jsonl"
    write_jsonl(hypothesis_path, hypothesis)

    stats = {
        "question_count": len(output_rows),
        "decision_policy": args.decision_policy,
        "decision_counts": {
            decision: sum(row["answerability_decision"] == decision for row in output_rows)
            for decision in ("keep", "revise", "abstain", "missing")
        },
        "effective_decision_counts": {
            decision: sum(row["answerability_effective_decision"] == decision for row in output_rows)
            for decision in ("keep", "revise", "abstain", "keep_original")
        },
        "hypothesis": str(hypothesis_path),
    }
    (args.output_dir / "answerability_policy_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
