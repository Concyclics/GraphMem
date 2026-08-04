#!/usr/bin/env python3
"""Prepare an offline review packet; never consumed by build/query code."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


STOP = {
    "a", "an", "and", "at", "before", "by", "did", "do", "for", "from",
    "have", "how", "i", "in", "is", "it", "many", "much", "my", "of",
    "on", "or", "the", "to", "was", "when", "will",
}


def tokens(value: object) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value).casefold())
        if len(token) > 1 and token not in STOP
    }


def score(question: str, answer: object, content: str) -> float:
    q, a, c = tokens(question), tokens(answer), tokens(content)
    value = 3.0 * len(q & c) + 2.0 * len(a & c)
    normalized_answer = " ".join(str(answer).casefold().split())
    normalized_content = " ".join(content.casefold().split())
    if normalized_answer and normalized_answer in normalized_content:
        value += 12.0
    if re.search(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b", content):
        value += 1.0
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-per-session", type=int, default=3)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    packet: list[dict[str, Any]] = []
    for case in cases:
        answer_sessions = set(map(str, case.get("answer_session_ids") or []))
        sessions = []
        for session_id, session_date, messages in zip(
            case.get("haystack_session_ids") or [],
            case.get("haystack_dates") or [],
            case.get("haystack_sessions") or [],
        ):
            if str(session_id) not in answer_sessions:
                continue
            ranked = []
            for turn_index, message in enumerate(messages):
                content = str(message.get("content") or "")
                ranked.append({
                    "turn_index": turn_index,
                    "role": str(message.get("role") or ""),
                    "speaker": str(message.get("speaker") or ""),
                    "content": content,
                    "score": score(case["question"], case.get("answer"), content),
                })
            ranked.sort(key=lambda row: (-row["score"], row["turn_index"]))
            user_ranked = [row for row in ranked if row["role"] == "user"]
            if not user_ranked:
                raise ValueError(f"gold session has no user turn: {case['question_id']}/{session_id}")
            sessions.append({
                "session_id": str(session_id), "session_date": session_date,
                "recommended": user_ranked[0],
                "user_candidates": user_ranked[: args.candidates_per_session],
                "all_role_candidates": ranked[: args.candidates_per_session],
                "candidate_margin": (
                    user_ranked[0]["score"] - user_ranked[1]["score"]
                    if len(user_ranked) > 1 else math.inf
                ),
            })
        packet.append({
            "question_id": str(case["question_id"]),
            "question_type": case.get("question_type"),
            "question": case.get("question"), "question_date": case.get("question_date"),
            "answer": case.get("answer"), "answer_sessions": sessions,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(packet), "sessions": sum(len(row["answer_sessions"]) for row in packet)}, indent=2))


if __name__ == "__main__":
    main()
