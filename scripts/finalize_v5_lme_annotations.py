#!/usr/bin/env python3
"""Finalize a Git-safe annotation asset from explicit semantic decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.draft):
        grouped[str(row["question_id"])].append(row)
    decisions = {str(row["question_id"]): row for row in read_jsonl(args.decisions)}
    if set(decisions) != set(grouped):
        raise ValueError("every drafted question requires exactly one explicit decision")
    output = []
    for qid in sorted(grouped):
        decision = decisions[qid]
        if decision.get("status") != "accepted":
            raise ValueError(f"question not adjudicated as accepted: {qid}")
        rows = decision.get("replacement_evidence") or grouped[qid]
        second_required = len(rows) > 1 or any(
            row.get("confidence") != "high" or row.get("disagreement") != "none"
            for row in rows
        )
        if second_required and not decision.get("second_reviewer"):
            raise ValueError(f"second reviewer required: {qid}")
        for row in rows:
            output.append({
                "question_id": qid, "session_id": str(row["session_id"]),
                "turn_index": int(row["turn_index"]), "span_start": int(row["span_start"]),
                "span_end": int(row["span_end"]), "support_role": row["support_role"],
                "confidence": row["confidence"], "first_review": "accepted",
                "second_review": "accepted" if second_required else "not_required",
                "adjudication": "changed" if decision.get("changed") else "accepted",
                "first_reviewer": decision["reviewer"],
                "second_reviewer": decision.get("second_reviewer", "not_required"),
                "disagreement": decision.get("disagreement", "none"),
                "annotation_version": "lme-v5-dev100-r1",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"questions": len(grouped), "annotations": len(output)}))


if __name__ == "__main__":
    main()
