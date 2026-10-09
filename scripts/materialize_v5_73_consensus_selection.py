#!/usr/bin/env python3
"""Prefer cross-family consensus, falling back to a frozen verifier choice."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer import cross_family_consensus_choice  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--fallback-selections", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    candidate_rows = read_jsonl(args.candidates)
    candidates = {str(row["question_id"]): row for row in candidate_rows}
    fallback = {str(row["question_id"]): row
                for row in read_jsonl(args.fallback_selections)}
    if set(candidates) != set(fallback):
        raise RuntimeError("candidate and fallback question sets differ")
    if args.expected and len(candidates) != args.expected:
        raise RuntimeError(f"expected {args.expected} questions, got {len(candidates)}")

    selections = []
    answers = []
    changed_answers = []
    for row in candidate_rows:
        question_id = str(row["question_id"])
        choices = list(row["candidates"])
        consensus = cross_family_consensus_choice(choices)
        if consensus is None:
            selected = int(fallback[question_id]["selected_index"])
            mode = "local_verifier_fallback"
        else:
            selected = consensus
            mode = "cross_family_consensus"
        prediction = str(choices[selected]["prediction"])
        selections.append({
            "question_id": question_id, "selected_index": selected,
            "prediction": prediction, "selection_mode": mode,
            "family": choices[selected].get("family"),
        })
        answers.append({
            "question_id": question_id, "prediction": prediction,
            "answer_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "selection_mode": mode,
        })
        if prediction != str(fallback[question_id]["prediction"]):
            changed_answers.append(answers[-1])

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("selections.jsonl", selections),
                       ("answers.jsonl", answers),
                       ("changed_answers.jsonl", changed_answers)):
        (args.output_root / name).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.73-consensus-selection-v1",
        "questions": len(answers),
        "changed_questions": len(changed_answers),
        "selection_modes": dict(sorted(Counter(
            row["selection_mode"] for row in selections).items())),
        "candidates": str(args.candidates),
        "candidates_sha256": hashlib.sha256(
            args.candidates.read_bytes()).hexdigest(),
        "fallback_selections": str(args.fallback_selections),
        "fallback_selections_sha256": hashlib.sha256(
            args.fallback_selections.read_bytes()).hexdigest(),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
