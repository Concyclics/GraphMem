#!/usr/bin/env python3
"""Combine four frozen direct and four V5.73 structured candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer import combine_candidate_families  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    direct = {str(row["question_id"]): row for row in read_jsonl(args.direct)}
    structured = {
        str(row["question_id"]): row for row in read_jsonl(args.structured)}
    if set(direct) != set(structured):
        raise RuntimeError("direct and structured question sets differ")
    if args.expected and len(direct) != args.expected:
        raise RuntimeError(f"expected {args.expected} questions, got {len(direct)}")

    rows = []
    for question_id, direct_row in direct.items():
        structured_row = structured[question_id]
        for field in ("conversation_id", "category", "memory_id"):
            if direct_row.get(field) != structured_row.get(field):
                raise RuntimeError(f"{question_id}: mismatched {field}")
        rows.append({
            "question_id": question_id,
            "conversation_id": direct_row.get("conversation_id"),
            "category": int(direct_row["category"]),
            "memory_id": direct_row.get("memory_id"),
            "family_prompt_payload_hashes": {
                "direct_v563": direct_row.get("prompt_payload_hash"),
                "structured_v573": structured_row.get(
                    "base_prompt_payload_hash"),
            },
            "candidates": list(combine_candidate_families(
                direct_row["candidates"], structured_row["candidates"],
                per_family=args.per_family)),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.73-hybrid-candidates-v1",
        "questions": len(rows), "per_family": args.per_family,
        "families": ["direct_v563", "structured_v573"],
        "direct": str(args.direct),
        "direct_sha256": hashlib.sha256(args.direct.read_bytes()).hexdigest(),
        "structured": str(args.structured),
        "structured_sha256": hashlib.sha256(
            args.structured.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
