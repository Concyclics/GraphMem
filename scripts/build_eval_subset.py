#!/usr/bin/env python3
"""Build a fixed, stratified LongMemEval-S evaluation subset.

The sampler keeps type proportions close to the full cleaned benchmark,
mixes simple vs multi-gold-session items inside complex types, and includes
abstention questions at roughly the same rate as the source set.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TYPE_ORDER = [
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/longmemeval_s_cleaned.json"),
    )
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--name",
        default="longmemeval_s_subset50_balanced",
        help="Base filename (without extension) for data + manifest outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    return parser.parse_args()


def _is_abstention(row: dict[str, Any]) -> bool:
    return str(row.get("question_id", "")).endswith("_abs")


def _gold_session_count(row: dict[str, Any]) -> int:
    return len(row.get("answer_session_ids") or [])


def _complexity_bucket(row: dict[str, Any]) -> str:
    gold = _gold_session_count(row)
    if gold <= 1:
        return "single_gold"
    if gold == 2:
        return "gold_2"
    return "gold_3plus"


def allocate_counts(rows: list[dict[str, Any]], target_size: int) -> dict[str, int]:
    type_counts = Counter(row["question_type"] for row in rows)
    total = len(rows)
    raw = {question_type: target_size * count / total for question_type, count in type_counts.items()}
    floors = {question_type: int(value) for question_type, value in raw.items()}
    allocated = sum(floors.values())
    remainder = target_size - allocated
    fractions = sorted(
        ((question_type, raw[question_type] - floors[question_type]) for question_type in raw),
        key=lambda item: (-item[1], TYPE_ORDER.index(item[0]) if item[0] in TYPE_ORDER else 99),
    )
    for index in range(remainder):
        question_type = fractions[index % len(fractions)][0]
        floors[question_type] += 1
    return floors


def pick_from_bucket(
    rng: random.Random,
    items: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    return rng.sample(items, count)


def sample_type_rows(
    rng: random.Random,
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)

    abstain_rows = [row for row in rows if _is_abstention(row)]
    answerable_rows = [row for row in rows if not _is_abstention(row)]

    target_abstain = round(count * len(abstain_rows) / len(rows))
    target_abstain = min(target_abstain, len(abstain_rows), count)
    target_answerable = count - target_abstain

    selected: list[dict[str, Any]] = []
    selected.extend(pick_from_bucket(rng, abstain_rows, target_abstain))

    remaining_answerable = [row for row in answerable_rows if row not in selected]
    complex_types = {"multi-session", "temporal-reasoning", "knowledge-update"}
    question_type = rows[0]["question_type"]

    if question_type in complex_types and target_answerable > 1:
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in remaining_answerable:
            by_bucket[_complexity_bucket(row)].append(row)
        bucket_order = ["gold_3plus", "gold_2", "single_gold"]
        bucket_targets = _split_evenly(target_answerable, [bucket for bucket in bucket_order if by_bucket[bucket]])
        for bucket, bucket_count in bucket_targets.items():
            selected.extend(pick_from_bucket(rng, by_bucket[bucket], bucket_count))
        if len(selected) < count:
            leftovers = [row for row in remaining_answerable if row not in selected]
            selected.extend(pick_from_bucket(rng, leftovers, count - len(selected)))
    else:
        selected.extend(pick_from_bucket(rng, remaining_answerable, target_answerable))

    if len(selected) < count:
        leftovers = [row for row in rows if row not in selected]
        selected.extend(pick_from_bucket(rng, leftovers, count - len(selected)))
    return selected[:count]


def _split_evenly(total: int, buckets: list[str]) -> dict[str, int]:
    if not buckets:
        return {}
    base = total // len(buckets)
    remainder = total % len(buckets)
    out: dict[str, int] = {}
    for index, bucket in enumerate(buckets):
        out[bucket] = base + (1 if index < remainder else 0)
    return out


def build_subset(rows: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    allocations = allocate_counts(rows, target_size)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)

    selected: list[dict[str, Any]] = []
    for question_type in TYPE_ORDER:
        type_rows = by_type.get(question_type, [])
        selected.extend(sample_type_rows(rng, type_rows, allocations.get(question_type, 0)))

    selected.sort(key=lambda row: (TYPE_ORDER.index(row["question_type"]), row["question_id"]))
    return selected


def manifest_for(rows: list[dict[str, Any]], source: Path, seed: int, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "source": str(source),
        "seed": seed,
        "size": len(rows),
        "question_ids": [row["question_id"] for row in rows],
        "type_counts": dict(Counter(row["question_type"] for row in rows)),
        "abstention_count": sum(1 for row in rows if _is_abstention(row)),
        "gold_session_counts": dict(Counter(_gold_session_count(row) for row in rows)),
        "rows": [
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "is_abstention": _is_abstention(row),
                "gold_session_count": _gold_session_count(row),
                "question_date": row.get("question_date"),
            }
            for row in rows
        ],
    }


def main() -> None:
    args = parse_args()
    source_rows = json.loads(args.source.read_text(encoding="utf-8"))
    subset = build_subset(source_rows, args.size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{args.name}.json"
    manifest_path = args.output_dir / f"{args.name}.manifest.json"

    data_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest_for(subset, args.source, args.seed, args.name), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {data_path} ({len(subset)} questions)")
    print(f"wrote {manifest_path}")
    print("type_counts:", dict(Counter(row["question_type"] for row in subset)))
    print("abstention:", sum(1 for row in subset if _is_abstention(row)))


if __name__ == "__main__":
    main()
