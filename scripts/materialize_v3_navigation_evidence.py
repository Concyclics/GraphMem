#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize selected V3 navigation evidence for offline retrieval-"
            "sufficiency judging. Gold fields are diagnostic outputs only."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--navigation-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = {
        str(row["question_id"]): row
        for row in json.loads(args.data.read_text(encoding="utf-8"))
    }
    answers = {str(row["question_id"]): row for row in _jsonl(args.answers)}
    navigation = {
        str(row["question_id"]): row for row in _jsonl(args.navigation_results)
    }
    question_ids = [question_id for question_id in answers if question_id in navigation]
    missing = [question_id for question_id in question_ids if question_id not in data]
    if missing:
        raise ValueError(f"questions missing from source data: {missing[:8]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for question_id in question_ids:
            source = data[question_id]
            row = {
                "question_id": question_id,
                "question_type": source.get("question_type"),
                "question_date": source.get("question_date"),
                "question": source.get("question"),
                "gold_answer": source.get("answer"),
                "prediction": navigation[question_id].get("selected_evidence", ""),
                "diagnostic_only": True,
                "gold_used_at_runtime": False,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"question_count": len(question_ids), "output": str(args.output)}))


if __name__ == "__main__":
    main()
