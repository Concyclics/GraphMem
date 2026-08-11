#!/usr/bin/env python3
"""Materialize a paired answer candidate from certified typed executions.

The answer stage deliberately keeps deterministic bypass disabled until its
precision is audited.  This utility substitutes only rows whose frozen
``PreparedAnswer`` trace says ``safe_to_bypass`` and leaves every other answer
byte-identical, so ``paired_judge_delta.py`` can evaluate the proposed bypass
without rerunning retrieval or the answer model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    answers = _rows(args.answers)
    prepared = {
        str(row["question_id"]): row for row in _rows(args.prepared)
    }
    changed: list[str] = []
    by_benchmark: dict[str, int] = {}
    for row in answers:
        question_id = str(row["question_id"])
        frozen = prepared.get(question_id)
        if frozen is None:
            raise ValueError(f"missing PreparedAnswer for {question_id}")
        typed = frozen.get("trace", {}).get("typed_execution") or {}
        prediction = str(frozen.get("draft_text") or "").strip()
        if not typed.get("safe_to_bypass") or not prediction:
            continue
        if prediction == str(row.get("prediction", "")):
            continue
        row["prediction"] = prediction
        row["deterministic_bypass_candidate"] = True
        row["deterministic_bypass_kind"] = typed.get("kind")
        changed.append(question_id)
        benchmark = str(row.get("benchmark") or "unknown")
        by_benchmark[benchmark] = by_benchmark.get(benchmark, 0) + 1

    if len(prepared) != len(answers):
        answer_ids = {str(row["question_id"]) for row in answers}
        extras = sorted(set(prepared) - answer_ids)
        if extras:
            raise ValueError(f"PreparedAnswer has {len(extras)} extra question ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in answers)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-typed-execution-candidate-v1",
        "answers": str(args.answers),
        "prepared": str(args.prepared),
        "questions": len(answers),
        "changed": len(changed),
        "changed_by_benchmark": dict(sorted(by_benchmark.items())),
        "changed_question_ids": changed,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
