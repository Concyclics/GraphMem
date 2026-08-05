#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUESTION_FILES = {
    "answers.jsonl",
    "question_stats.jsonl",
    "retrieval_results.jsonl",
}
JSONL_FILES = (
    "llm_calls.jsonl",
    "embedding_calls.jsonl",
    "compression_stats.jsonl",
    "nodes.jsonl",
    "state_chains.jsonl",
    "index_diagnostics.jsonl",
    "edges.jsonl",
    "question_stats.jsonl",
    "retrieval_results.jsonl",
    "answers.jsonl",
)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed LoCoMo GraphMem shards.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument(
        "--additional-shard-root",
        type=Path,
        action="append",
        default=[],
        help="Additional disjoint shard root; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="hierarchical_state_graph_v2")
    parser.add_argument("--expected-questions", type=int, default=1986)
    args = parser.parse_args()

    shard_roots = [args.shard_root, *args.additional_shard_root]
    shard_dirs = sorted(
        path / args.variant
        for root in shard_roots
        for path in root.glob("shard_*")
        if (path / args.variant).is_dir()
    )
    if not shard_dirs:
        raise SystemExit(f"No completed shard variant directories under {shard_roots}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for filename in JSONL_FILES:
        rows = [row for directory in shard_dirs for row in _read(directory / filename)]
        if filename in QUESTION_FILES:
            by_question: dict[str, dict[str, Any]] = {}
            for row in rows:
                question_id = str(row["question_id"])
                if question_id in by_question:
                    raise ValueError(f"duplicate {question_id} in {filename}")
                by_question[question_id] = row
            rows = [by_question[key] for key in sorted(by_question)]
            if len(rows) != args.expected_questions:
                raise ValueError(
                    f"{filename}: expected {args.expected_questions} questions, got {len(rows)}"
                )
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[filename] = len(rows)
    print(json.dumps({"shard_roots": len(shard_roots), "shards": len(shard_dirs), "counts": counts}))


if __name__ == "__main__":
    main()
