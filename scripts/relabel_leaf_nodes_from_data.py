#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.data import build_leaf_nodes, load_longmemeval_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite saved leaf node text from the source data while preserving saved "
            "summary nodes. This is useful when older runs omitted speaker/listener labels."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-type", default="all")
    parser.add_argument("--max-questions", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    cases = load_longmemeval_cases(args.data, args.question_type, args.max_questions)
    leaves_by_qid = {
        case.question_id: {leaf.node_id: leaf for leaf in build_leaf_nodes(case)}
        for case in cases
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rewritten = 0
    missing = 0
    total_leaf = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.nodes):
            if row.get("node_type") == "leaf":
                total_leaf += 1
                qid = str(row.get("question_id") or "")
                node_id = str(row.get("node_id") or "")
                replacement = leaves_by_qid.get(qid, {}).get(node_id)
                if replacement is None:
                    missing += 1
                else:
                    embedding = row.get("embedding")
                    row = asdict(replacement)
                    row["node_type"] = "leaf"
                    if embedding is not None:
                        row["embedding"] = embedding
                    rewritten += 1
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "total_leaf_nodes": total_leaf,
                "rewritten_leaf_nodes": rewritten,
                "missing_leaf_nodes": missing,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
