#!/usr/bin/env python3
"""Materialize current Luna-wrong answers not covered by an exact Sol verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question IDs in {path}")
    return result


def prediction_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("prediction") or "").encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    sol: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    for path in sorted(args.search_root.glob("**/auto_eval.jsonl")):
        for row in read_jsonl(path):
            if str(row.get("judge_model")) != "gpt-5.6-sol":
                continue
            key = (str(row.get("question_id")), str(row.get("prediction_sha256")))
            if key[0] and key[1]:
                sol[key] = (row, str(path))

    summary: dict[str, Any] = {"schema_version": "cross-judge-audit-v1"}
    disagreements: list[dict[str, Any]] = []
    for benchmark, answer_name, judge_suffix in (
        ("longmemeval", "answers_longmemeval.jsonl", "lme"),
        ("locomo", "answers_locomo.jsonl", "locomo"),
    ):
        answers = keyed(args.control_root / "answer" / answer_name)
        luna = keyed(
            args.control_root / "judge" / "luna_medium" / judge_suffix
            / "auto_eval.jsonl"
        )
        if set(answers) != set(luna):
            raise ValueError(f"answer/Luna coverage differs for {benchmark}")

        pairs: Counter[tuple[bool, bool]] = Counter()
        exact = 0
        uncovered_wrong: list[dict[str, Any]] = []
        for question_id, answer in answers.items():
            key = (question_id, prediction_sha256(answer))
            matched = sol.get(key)
            if matched is None:
                if not bool(luna[question_id]["correct"]):
                    uncovered_wrong.append(answer)
                continue
            exact += 1
            sol_row, source = matched
            pair = (bool(luna[question_id]["correct"]), bool(sol_row["correct"]))
            pairs[pair] += 1
            if pair[0] != pair[1]:
                disagreements.append(
                    {
                        "benchmark": benchmark,
                        "question_id": question_id,
                        "question": answer.get("question"),
                        "gold_answer": answer.get("gold_answer"),
                        "prediction": answer.get("prediction"),
                        "luna_verdict": luna[question_id],
                        "sol_verdict": sol_row,
                        "sol_source": source,
                    }
                )

        output_name = (
            "uncovered_wrong_longmemeval.jsonl"
            if benchmark == "longmemeval"
            else "uncovered_wrong_locomo.jsonl"
        )
        write_jsonl(args.output_root / output_name, uncovered_wrong)
        summary[benchmark] = {
            "answers": len(answers),
            "luna_correct": sum(bool(row["correct"]) for row in luna.values()),
            "luna_wrong": sum(not bool(row["correct"]) for row in luna.values()),
            "exact_sol_coverage": exact,
            "both_correct": pairs[(True, True)],
            "both_wrong": pairs[(False, False)],
            "luna_only_correct": pairs[(True, False)],
            "sol_only_correct": pairs[(False, True)],
            "uncovered_luna_wrong": len(uncovered_wrong),
            "uncovered_answers": str(args.output_root / output_name),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_root / "exact_disagreements.jsonl", disagreements)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
