#!/usr/bin/env python3
"""Export one Git-safe deterministic user-turn candidate per official gold session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    rows = []
    for case in packet:
        for session in case["answer_sessions"]:
            turn = session["recommended"]
            rows.append({
                "question_id": case["question_id"], "session_id": session["session_id"],
                "turn_index": turn["turn_index"], "span_start": 0,
                "span_end": len(turn["content"]), "support_role": (
                    "negative_scope" if str(case["question_id"]).endswith("_abs")
                    else "temporal_endpoint" if case["question_type"] == "temporal-reasoning"
                    else "aggregation_member"
                ),
                "confidence": "medium", "candidate_score": turn["score"],
                "candidate_margin": session["candidate_margin"],
                "annotation_version": "lme-v5-dev100-deterministic-draft-r1",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"questions": len({row["question_id"] for row in rows}),
                      "gold_sessions": len(rows)}))


if __name__ == "__main__":
    main()
