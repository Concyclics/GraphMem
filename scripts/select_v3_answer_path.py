#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.v3.navigation_policy import navigation_decision  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select V3 baseline or graph-navigation answers by query algebra."
    )
    parser.add_argument("--baseline-answers", type=Path, required=True)
    parser.add_argument("--navigation-answers", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = _read_jsonl(args.baseline_answers)
    baseline = {str(row["question_id"]): row for row in baseline_rows}
    navigation = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.navigation_answers)
    }
    retrieval = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.retrieval_results)
    }
    if set(baseline) != set(navigation) or set(baseline) != set(retrieval):
        raise RuntimeError("baseline, navigation, and retrieval question IDs differ")

    answers: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for row in baseline_rows:
        question_id = str(row["question_id"])
        trace = retrieval[question_id].get("retrieval_trace") or {}
        decision = navigation_decision(trace)
        chosen = navigation[question_id] if decision.use_graph_navigation else row
        output = dict(chosen)
        output["variant"] = "v3_selective_graph_navigation"
        output["selected_answer_path"] = (
            "graph_navigation" if decision.use_graph_navigation else "baseline"
        )
        output["selection_query_kind"] = decision.query_kind
        output["selection_reason"] = decision.reason
        answers.append(output)
        decisions.append(
            {
                "question_id": question_id,
                "selected_answer_path": output["selected_answer_path"],
                "query_kind": decision.query_kind,
                "reason": decision.reason,
                "answer_total_tokens": int(
                    chosen.get("answer_total_tokens")
                    or retrieval[question_id].get("answer_total_tokens")
                    or 0
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("answers.jsonl", answers),
        ("selection_decisions.jsonl", decisions),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "question_count": len(answers),
        "graph_navigation_count": sum(
            row["selected_answer_path"] == "graph_navigation"
            for row in decisions
        ),
        "baseline_count": sum(
            row["selected_answer_path"] == "baseline" for row in decisions
        ),
        "max_recorded_answer_tokens": max(
            (row["answer_total_tokens"] for row in decisions), default=0
        ),
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
