#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _round_robin_sample(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    stratum_field: str,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(stratum_field) or "unknown")].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    ordered_groups = sorted(groups)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for key in ordered_groups:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible stratified error plus control replay set."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-count", type=int, default=24)
    parser.add_argument("--control-count", type=int, default=12)
    parser.add_argument("--stratum-field", default="question_type")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    by_id = {str(row["question_id"]): row for row in data}
    judgments = _jsonl(args.judge)
    annotated: list[dict[str, Any]] = []
    for judgment in judgments:
        question_id = str(judgment.get("question_id") or "")
        source = by_id.get(question_id)
        if source is None:
            continue
        annotated.append({
            **source,
            "_replay_correct": bool(judgment.get("correct")),
            "_replay_stratum": str(
                judgment.get(args.stratum_field)
                or source.get(args.stratum_field)
                or "unknown"
            ),
        })
    errors = [row for row in annotated if not row["_replay_correct"]]
    controls = [row for row in annotated if row["_replay_correct"]]
    selected_errors = _round_robin_sample(
        errors,
        limit=args.error_count,
        stratum_field="_replay_stratum",
        seed=args.seed,
    )
    selected_controls = _round_robin_sample(
        controls,
        limit=args.control_count,
        stratum_field="_replay_stratum",
        seed=args.seed + 1,
    )
    selected = [*selected_errors, *selected_controls]
    for row in selected:
        row.pop("_replay_correct", None)
        row.pop("_replay_stratum", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ids = [str(row["question_id"]) for row in selected]
    manifest = {
        "source": str(args.data),
        "judge": str(args.judge),
        "seed": args.seed,
        "stratum_field": args.stratum_field,
        "error_count": len(selected_errors),
        "control_count": len(selected_controls),
        "question_count": len(selected),
        "error_question_ids": [
            str(row["question_id"]) for row in selected_errors
        ],
        "control_question_ids": [
            str(row["question_id"]) for row in selected_controls
        ],
        "question_ids_sha256": hashlib.sha256(
            "\n".join(ids).encode("utf-8")
        ).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
