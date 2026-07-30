#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any


_LABEL = re.compile(r"^D(\d+):(\d+)$")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expected_suffixes(row: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for value in row.get("locomo_evidence") or []:
        match = _LABEL.fullmatch(str(value).strip())
        if match:
            result.append(
                f":session_{int(match.group(1))}:turn:{max(0, int(match.group(2)) - 1)}"
            )
    return list(dict.fromkeys(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-only audit of LoCoMo gold turns in V3 navigation closures."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--navigation-results", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = {
        str(row["question_id"]): row
        for row in json.loads(args.data.read_text(encoding="utf-8"))
    }
    navigation = {
        str(row["question_id"]): row for row in _jsonl(args.navigation_results)
    }
    judges = {
        str(row["question_id"]): row for row in _jsonl(args.judge_results)
    }
    audits: list[dict[str, Any]] = []
    for question_id, judge in judges.items():
        expected = _expected_suffixes(data.get(question_id, {}))
        closure = {
            str(value) for value in navigation.get(question_id, {}).get("closure_ids") or []
        }
        candidates = {
            str(value)
            for value in navigation.get(question_id, {}).get("valid_candidate_ids") or []
        }
        found = sum(any(node_id.endswith(suffix) for node_id in closure) for suffix in expected)
        candidate_found = sum(
            any(node_id.endswith(suffix) for node_id in candidates) for suffix in expected
        )
        recall = found / len(expected) if expected else None
        nav = navigation.get(question_id, {})
        audits.append(
            {
                "question_id": question_id,
                "correct": bool(judge.get("correct")),
                "expected_turn_count": len(expected),
                "found_turn_count": found,
                "candidate_turn_count": candidate_found,
                "candidate_gold_turn_recall": (
                    candidate_found / len(expected) if expected else None
                ),
                "gold_turn_recall": recall,
                "gold_turn_all_hit": bool(expected) and found == len(expected),
                "closure_size": len(closure),
                "selected_count": len((nav.get("plan") or {}).get("selected_ids") or []),
                "recovery_applied": bool(nav.get("graph_recovery_applied", True)),
            }
        )

    with_gold = [row for row in audits if row["gold_turn_recall"] is not None]
    groups: dict[str, Any] = {}
    for correct in (True, False):
        rows = [row for row in with_gold if row["correct"] is correct]
        groups["correct" if correct else "wrong"] = {
            "count": len(rows),
            "gold_turn_recall": mean(row["gold_turn_recall"] for row in rows) if rows else None,
            "gold_turn_all_hit_rate": mean(row["gold_turn_all_hit"] for row in rows) if rows else None,
            "closure_size": mean(row["closure_size"] for row in rows) if rows else None,
        }
    summary = {
        "question_count": len(audits),
        "with_gold_evidence_labels": len(with_gold),
        "groups": groups,
        "wrong_recall_buckets": dict(
            Counter(
                "none" if row["gold_turn_recall"] == 0
                else "partial" if row["gold_turn_recall"] < 1
                else "all"
                for row in with_gold
                if not row["correct"]
            )
        ),
        "wrong_stage_buckets": dict(
            Counter(
                "frontier_miss"
                if row["candidate_turn_count"] == 0
                else "selection_or_recovery_miss"
                if row["found_turn_count"] == 0
                else "partial_closure"
                if not row["gold_turn_all_hit"]
                else "answer_error_with_full_gold_turns"
                for row in with_gold
                if not row["correct"]
            )
        ),
        "runtime_contract": "offline diagnostics only; gold labels are never retrieval inputs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": audits}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
