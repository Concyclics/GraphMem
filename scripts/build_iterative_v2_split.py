#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if isinstance(row, dict)]


def ids_from(paths: list[Path]) -> set[str]:
    return {
        str(row["question_id"])
        for path in paths
        for row in read_rows(path)
        if row.get("question_id") is not None
    }


def allocation(groups: dict[str, list[dict[str, Any]]], limit: int) -> dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    if total == 0:
        return {key: 0 for key in groups}
    target = min(limit, total)
    raw = {key: target * len(rows) / total for key, rows in groups.items()}
    result = {key: min(len(groups[key]), math.floor(value)) for key, value in raw.items()}
    order = sorted(
        groups,
        key=lambda key: (-(raw[key] - math.floor(raw[key])), -len(groups[key]), key),
    )
    while sum(result.values()) < target:
        for key in order:
            if result[key] < len(groups[key]):
                result[key] += 1
            if sum(result.values()) == target:
                break
    return result


def sample_stratified(
    rows: list[dict[str, Any]], limit: int, rng: random.Random
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("question_type") or "unknown")].append(row)
    for values in groups.values():
        rng.shuffle(values)
    counts = allocation(groups, limit)
    selected = [
        row
        for key in sorted(groups)
        for row in groups[key][: counts[key]]
    ]
    rng.shuffle(selected)
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(ids: set[str]) -> str:
    payload = "\n".join(sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("question_type") or "unknown") for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one leak-resistant GraphMem iteration split."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dev-selection", type=Path, nargs="+", required=True)
    parser.add_argument("--seen-selection", type=Path, nargs="*", default=[])
    parser.add_argument("--error-judgments", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-size", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("--data must contain one JSON array")
    by_id = {str(row["question_id"]): row for row in data}
    if len(by_id) != len(data):
        raise SystemExit("source dataset contains duplicate question_id values")

    prior_dev_ids = ids_from(args.dev_selection)
    prior_seen_ids = prior_dev_ids | ids_from(args.seen_selection)
    judgment_rows = [
        row for path in args.error_judgments for row in read_rows(path)
    ]
    judged_ids = {
        str(row["question_id"])
        for row in judgment_rows
        if row.get("question_id") is not None
    }
    new_error_ids = {
        str(row["question_id"])
        for row in judgment_rows
        if row.get("question_id") is not None and not bool(row.get("correct"))
    }
    dev_ids = prior_dev_ids | new_error_ids
    seen_before = prior_seen_ids | judged_ids

    missing = sorted((dev_ids | seen_before) - set(by_id))
    if missing:
        raise SystemExit(f"selection contains ids absent from source data: {missing[:10]}")
    eligible = [
        row for row in data if str(row["question_id"]) not in seen_before | dev_ids
    ]
    if len(eligible) < args.blind_size:
        raise SystemExit(
            f"only {len(eligible)} unseen questions remain; requested {args.blind_size}"
        )
    blind = sample_stratified(
        eligible, args.blind_size, random.Random(args.seed)
    )
    blind_ids = {str(row["question_id"]) for row in blind}
    if blind_ids & dev_ids:
        raise SystemExit("development/blind overlap detected")

    development = [row for row in data if str(row["question_id"]) in dev_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "development.json").write_text(
        json.dumps(development, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "blind_50.json").write_text(
        json.dumps(blind, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_jsonl(
        args.output_dir / "development_ids.jsonl",
        [{"question_id": str(row["question_id"])} for row in development],
    )
    write_jsonl(
        args.output_dir / "blind_ids.jsonl",
        [{"question_id": str(row["question_id"])} for row in blind],
    )

    manifest = {
        "iteration": args.iteration,
        "seed": args.seed,
        "source_questions": len(data),
        "prior_development_questions": len(prior_dev_ids),
        "new_errors_added_to_development": len(new_error_ids - prior_dev_ids),
        "development_questions": len(development),
        "seen_before_sampling": len(seen_before | dev_ids),
        "eligible_unseen_before_sampling": len(eligible),
        "blind_questions": len(blind),
        "unseen_after_sampling": len(eligible) - len(blind),
        "development_blind_overlap": 0,
        "development_type_counts": type_counts(development),
        "blind_type_counts": type_counts(blind),
        "development_ids_sha256": digest(dev_ids),
        "blind_ids_sha256": digest(blind_ids),
        "protocol": (
            "Previous blind errors join development. Every prior blind question "
            "remains excluded from future blind samples, regardless of verdict."
        ),
        "files": {
            "development": "development.json",
            "blind": "blind_50.json",
            "development_ids": "development_ids.jsonl",
            "blind_ids": "blind_ids.jsonl",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
