#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a completed blind batch into a regression dev set while also "
            "materializing its judge failures as a focused tuning set."
        )
    )
    parser.add_argument("--base-dev", type=Path, required=True)
    parser.add_argument("--blind-data", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--errors-only-output", type=Path,
        help="Optionally materialize only the current blind-batch failures.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("all", "failures"),
        default="all",
        help=(
            "Rows promoted into the regression dev set. 'all' is the strict "
            "default; 'failures' preserves the legacy behavior."
        ),
    )
    args = parser.parse_args()

    base_rows = _load_rows(args.base_dev)
    blind_rows = _load_rows(args.blind_data)
    judgments = _load_jsonl(args.judge_results)
    failed_ids = {
        str(row["question_id"])
        for row in judgments
        if row.get("correct") is False
    }
    blind_by_id = {str(row["question_id"]): row for row in blind_rows}
    missing = sorted(failed_ids - blind_by_id.keys())
    if missing:
        raise ValueError(f"judge failures missing from blind data: {missing}")
    if args.errors_only_output is not None:
        error_rows = [
            row for row in blind_rows
            if str(row["question_id"]) in failed_ids
        ]
        args.errors_only_output.parent.mkdir(parents=True, exist_ok=True)
        args.errors_only_output.write_text(
            json.dumps(error_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.errors_only_output.with_suffix(".manifest.json").write_text(
            json.dumps({
                "blind_data": str(args.blind_data),
                "judge_results": str(args.judge_results),
                "question_count": len(error_rows),
                "question_ids": [str(row["question_id"]) for row in error_rows],
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    promoted_ids = (
        set(blind_by_id) if args.merge_mode == "all" else set(failed_ids)
    )
    merged: dict[str, dict[str, Any]] = {
        str(row["question_id"]): row for row in base_rows
    }
    for question_id in sorted(promoted_ids):
        merged.setdefault(question_id, blind_by_id[question_id])
    rows = list(merged.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "base_dev": str(args.base_dev),
        "blind_data": str(args.blind_data),
        "judge_results": str(args.judge_results),
        "merge_mode": args.merge_mode,
        "base_count": len(base_rows),
        "blind_count": len(blind_rows),
        "blind_failure_count": len(failed_ids),
        "promoted_count": len(promoted_ids),
        "merged_count": len(rows),
        "added_question_ids": sorted(promoted_ids - {
            str(row["question_id"]) for row in base_rows
        }),
        "all_question_ids": [str(row["question_id"]) for row in rows],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
