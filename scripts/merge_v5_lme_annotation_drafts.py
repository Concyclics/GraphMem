#!/usr/bin/env python3
"""Merge disjoint drafts and create an offline semantic review packet."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--draft", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    expected = {str(case["question_id"]) for case in cases}
    rows: list[dict[str, Any]] = []
    for path in args.draft:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    if set(grouped) != expected:
        raise ValueError(f"draft coverage mismatch: missing={sorted(expected-set(grouped))}, extra={sorted(set(grouped)-expected)}")
    by_id = {str(case["question_id"]): case for case in cases}
    packet = []
    for qid in sorted(grouped):
        case = by_id[qid]
        sessions = dict(zip(map(str, case["haystack_session_ids"]), case["haystack_sessions"]))
        evidence = []
        for row in sorted(grouped[qid], key=lambda item: (item["session_id"], item["turn_index"], item["span_start"])):
            text = str(sessions[row["session_id"]][row["turn_index"]].get("content") or "")
            evidence.append({
                **row,
                "source_role": sessions[row["session_id"]][row["turn_index"]].get("role"),
                "review_excerpt": text[row["span_start"]:row["span_end"]],
            })
        packet.append({
            "question_id": qid, "question_type": case.get("question_type"),
            "question": case["question"], "reference_answer": case.get("answer"),
            "official_gold_sessions": case.get("answer_session_ids"), "evidence": evidence,
            "requires_second_review": (
                len(evidence) > 1 or any(
                    row["confidence"] != "high" or row["disagreement"] != "none"
                    for row in evidence
                )
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for qid in sorted(grouped) for row in sorted(
                    grouped[qid], key=lambda item: (item["session_id"], item["turn_index"], item["span_start"])
                )),
        encoding="utf-8",
    )
    args.review_packet.parent.mkdir(parents=True, exist_ok=True)
    args.review_packet.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(grouped), "annotations": len(rows),
                      "second_review_questions": sum(row["requires_second_review"] for row in packet)}))


if __name__ == "__main__":
    main()
