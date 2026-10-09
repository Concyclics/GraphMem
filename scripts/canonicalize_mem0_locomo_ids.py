#!/usr/bin/env python3
"""Map archived Mem0 ``convN_qM`` LoCoMo IDs to GraphMem canonical IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        str(row["question_id"]): row
        for row in json.loads(args.data.read_text(encoding="utf-8"))
        if int(row["locomo_category"]) in {1, 2, 3, 4}
    }
    source = read_jsonl(args.answers)
    converted = []
    mapping = []
    for row in source:
        source_id = str(row["question_id"])
        match = re.fullmatch(r"conv(\d+)_q(\d+)", source_id)
        if not match:
            raise RuntimeError(f"unexpected archived LoCoMo ID: {source_id}")
        canonical_id = f"locomo{int(match.group(1)):02d}_{int(match.group(2)):04d}"
        case = cases.get(canonical_id)
        if case is None:
            raise RuntimeError(f"canonical ID is absent from category 1-4 data: {canonical_id}")
        if str(row.get("question")) != str(case.get("question")):
            raise RuntimeError(f"question mismatch for {source_id} -> {canonical_id}")
        if int(row.get("category") or 0) != int(case["locomo_category"]):
            raise RuntimeError(f"category mismatch for {source_id} -> {canonical_id}")
        mapped = dict(row)
        mapped["source_question_id"] = source_id
        mapped["question_id"] = canonical_id
        mapped["stratum"] = f"category_{int(case['locomo_category'])}"
        converted.append(mapped)
        mapping.append({"source_question_id": source_id, "question_id": canonical_id})
    converted.sort(key=lambda row: str(row["question_id"]))
    if len(converted) != 1540 or len({row["question_id"] for row in converted}) != 1540:
        raise RuntimeError(
            f"expected 1540 unique converted answers, got {len(converted)}"
        )
    if set(row["question_id"] for row in converted) != set(cases):
        raise RuntimeError("converted IDs do not exactly cover the canonical LoCoMo set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "answers_locomo.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
        encoding="utf-8",
    )
    mapping_path = args.output_dir / "id_mapping.jsonl"
    mapping_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapping),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "mem0-locomo-canonical-id-view-v1",
        "mapping_rule": "conv{sample_index}_q{question_index} -> locomo{sample_index:02d}_{question_index:04d}",
        "questions": len(converted),
        "question_text_mismatches": 0,
        "category_mismatches": 0,
        "coverage_exact": True,
        "source_answers": str(args.answers),
        "source_answers_sha256": digest(args.answers),
        "canonical_data": str(args.data),
        "canonical_data_sha256": digest(args.data),
        "output": str(output),
        "output_sha256": digest(output),
    }
    (args.output_dir / "mapping_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
