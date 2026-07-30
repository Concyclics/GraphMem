#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reproducible question subset without changing runtime logic."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-id", action="append", default=[])
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--exclude-data", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    rows = json.loads(args.data.read_text(encoding="utf-8"))
    source_ids = [str(row["question_id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"duplicate question_id values in source data: {args.data}")
    include_ids = set(args.include_id)
    exclude_ids = set(args.exclude_id)
    for path in args.exclude_data:
        exclude_ids.update(
            str(row["question_id"])
            for row in json.loads(path.read_text(encoding="utf-8"))
        )

    eligible = [
        row
        for row in rows
        if str(row["question_id"]) not in exclude_ids
        and (not include_ids or str(row["question_id"]) in include_ids)
    ]
    selected = list(eligible)
    if args.seed is not None:
        random.Random(args.seed).shuffle(selected)
    if args.limit is not None:
        selected = selected[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.data),
        "output": str(args.output),
        "source_question_count": len(rows),
        "eligible_question_count": len(eligible),
        "question_count": len(selected),
        "question_ids": [str(row["question_id"]) for row in selected],
        "excluded_count": len(exclude_ids),
        "excluded_source_question_count": len(set(source_ids) & exclude_ids),
        "remaining_after_selection": len(eligible) - len(selected),
        "exclude_sources": [str(path) for path in args.exclude_data],
        "question_ids_sha256": hashlib.sha256(
            "\n".join(str(row["question_id"]) for row in selected).encode("utf-8")
        ).hexdigest(),
        "seed": args.seed,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
