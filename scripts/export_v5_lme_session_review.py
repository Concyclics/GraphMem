#!/usr/bin/env python3
"""Validate per-session selections and emit flat refs plus an offline review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    reviews = [json.loads(line) for line in args.review.read_text(encoding="utf-8").splitlines() if line.strip()]
    keyed = {(row["question_id"], row["session_id"]): row for row in reviews}
    if len(keyed) != len(reviews):
        raise ValueError("duplicate question/session reviews")
    expected = {
        (str(case["question_id"]), str(session_id))
        for case in cases for session_id in case["answer_session_ids"]
    }
    if set(keyed) != expected:
        raise ValueError(f"session coverage mismatch: missing={len(expected-set(keyed))}, extra={len(set(keyed)-expected)}")
    flat: list[dict[str, Any]] = []
    packet = []
    for case in sorted(cases, key=lambda row: str(row["question_id"])):
        qid = str(case["question_id"])
        sessions = dict(zip(map(str, case["haystack_session_ids"]), case["haystack_sessions"]))
        evidence = []
        for sid in map(str, case["answer_session_ids"]):
            for selected in keyed[(qid, sid)]["evidence"]:
                index = int(selected["turn_index"])
                source = sessions[sid][index]
                if source.get("role") != "user":
                    raise ValueError(f"non-user evidence: {qid}/{sid}/{index}")
                row = {
                    "question_id": qid, "session_id": sid, "turn_index": index,
                    "span_start": int(selected["span_start"]), "span_end": int(selected["span_end"]),
                    "support_role": selected["support_role"], "confidence": selected["confidence"],
                    "disagreement": "none",
                    "annotation_version": "lme-v5-dev100-session-review-draft-r1",
                }
                flat.append(row)
                evidence.append({**row, "source_role": "user",
                                 "review_excerpt": source.get("content", "")})
        packet.append({
            "question_id": qid, "question_type": case.get("question_type"),
            "question": case["question"], "reference_answer": case.get("answer"),
            "official_gold_sessions": list(map(str, case["answer_session_ids"])),
            "evidence": evidence,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in flat), encoding="utf-8")
    args.review_packet.parent.mkdir(parents=True, exist_ok=True)
    args.review_packet.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(packet), "gold_sessions": len(expected), "annotations": len(flat)}))


if __name__ == "__main__":
    main()
