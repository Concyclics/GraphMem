#!/usr/bin/env python3
"""Paired bootstrap comparison of two V5.12/V5.13 retrieval artifacts."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = (
    "all_hit", "recall", "precision", "f1", "gold_hits", "turns",
    "evidence_tokens", "latency_ms", "candidate_all_hit", "candidate_recall",
    "candidate_turns", "graph_only_turns", "graph_only_gold_hits",
    "candidate_turns_before_limit",
    "visited_nodes", "visited_edges",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    source = path / "per_question.jsonl" if path.is_dir() else path
    return {
        str(row["question_id"]): row
        for row in (
            json.loads(line) for line in source.read_text(
                encoding="utf-8").splitlines() if line.strip())}


def ci(values: list[float], resamples: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(42)
    samples = []
    for _ in range(resamples):
        samples.append(statistics.fmean(
            values[rng.randrange(len(values))] for _row in values))
    samples.sort()
    return [samples[int(.025 * len(samples))],
            samples[int(.975 * len(samples))]]


def main() -> None:
    args = parse_args()
    left = read_rows(args.left); right = read_rows(args.right)
    ids = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError("paired runs must contain identical question ids")
    arms = sorted(
        key for key, value in left[ids[0]].items()
        if key.startswith("turn") and isinstance(value, dict)
        and key in right[ids[0]])
    result = {}
    for arm in arms:
        fields = {}
        for field in FIELDS:
            values = [
                float(right[qid][arm][field]) - float(left[qid][arm][field])
                for qid in ids
                if field in left[qid][arm] and field in right[qid][arm]]
            if values:
                fields[field] = {
                    "mean": statistics.fmean(values),
                    "ci95": ci(values, args.resamples),
                }
        result[arm] = {
            "delta": fields,
            "all_hit_transitions": dict(Counter(
                f"{int(left[qid][arm]['all_hit'])}->"
                f"{int(right[qid][arm]['all_hit'])}" for qid in ids)),
        }
    payload = {
        "schema_version": "graphmem-v5.13-paired-retrieval-comparison-v1",
        "left": {"label": args.left_label, "path": str(args.left)},
        "right": {"label": args.right_label, "path": str(args.right)},
        "questions": len(ids),
        "bootstrap": {"seed": 42, "resamples": args.resamples},
        "arms": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
