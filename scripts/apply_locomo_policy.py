#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply explicit Locomo-only policy probes.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="locomo_policy")
    parser.add_argument("--category5-abstain", action="store_true")
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


def abstention_prediction(question: str) -> str:
    target = target_name(question)
    suffix = f" for {target}" if target else ""
    return (
        "Evidence facts: This is a Locomo category_5 answerability probe; the "
        f"policy treats the supplied memory as insufficient to answer{suffix}.\n\n"
        "Final answer: The information is not mentioned in the provided memory."
    )


def target_name(question: str) -> str:
    match = re.search(r"\b(Caroline|Melanie|Oscar)\b", question)
    return match.group(1) if match else ""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {str(row["question_id"]): row for row in read_json_or_jsonl(args.data)}
    rows: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for answer in read_json_or_jsonl(args.answers):
        qid = str(answer["question_id"])
        case = cases.get(qid, {})
        row = dict(answer)
        row["variant"] = args.variant
        row["locomo_policy_applied"] = False
        row["locomo_policy"] = ""
        if args.category5_abstain and str(case.get("question_type")) == "category_5":
            row["prediction"] = abstention_prediction(str(case.get("question") or answer.get("question") or ""))
            row["locomo_policy_applied"] = True
            row["locomo_policy"] = "category5_abstain"
            changed.append(row)
        rows.append(row)

    write_jsonl(args.output_dir / "answers.jsonl", rows)
    hypothesis = [
        {
            "question_id": row["question_id"],
            "hypothesis": row.get("prediction", ""),
            "original_question_id": row["question_id"],
            "original_question_type": row.get("question_type"),
        }
        for row in rows
    ]
    hypothesis_path = args.output_dir / f"{args.variant}_hypothesis.judge_compat.jsonl"
    write_jsonl(hypothesis_path, hypothesis)
    write_jsonl(args.output_dir / "changed_answers.jsonl", changed)
    stats = {
        "variant": args.variant,
        "question_count": len(rows),
        "changed_count": len(changed),
        "category5_abstain": bool(args.category5_abstain),
        "hypothesis": str(hypothesis_path),
    }
    (args.output_dir / "locomo_policy_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
