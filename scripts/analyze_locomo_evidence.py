#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit official LoCoMo evidence-turn survival through GraphMem retrieval."
    )
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.raw_data.read_text(encoding="utf-8"))
    cases = {row["question_id"]: row for row in json.loads(args.data.read_text(encoding="utf-8"))}
    retrieval = {
        row["question_id"]: row
        for row in _read_jsonl(args.run_dir / "retrieval_results.jsonl")
    }
    turn_text: dict[tuple[int, str], str] = {}
    for sample_index, sample in enumerate(raw):
        for key, turns in sample["conversation"].items():
            if not re.fullmatch(r"session_\d+", key) or not isinstance(turns, list):
                continue
            for turn in turns:
                turn_text[(sample_index, str(turn["dia_id"]))] = str(turn["text"])

    stages = ("semantic_top28", "bm25_top28", "entity_top28", "prepack", "postpack")
    totals: dict[str, list[int]] = {stage: [0, 0] for stage in stages}
    exact_text = [0, 0]
    exact_text_macro = [0.0, 0]
    category: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {stage: [0, 0] for stage in (*stages, "exact_text")}
    )
    for question_id, case in cases.items():
        row = retrieval[question_id]
        trace = row.get("retrieval_trace") or {}
        channels = trace.get("leaf_channels") or {}
        evidence_ids = []
        for evidence in case.get("locomo_evidence") or []:
            match = re.fullmatch(r"D(\d+):(\d+)", str(evidence), flags=re.IGNORECASE)
            if match:
                evidence_ids.append(
                    f"{question_id}:session_{int(match.group(1))}:leaf:"
                    f"{(int(match.group(2)) - 1) // 2}"
                )
        if not evidence_ids:
            continue
        sets = {
            "semantic_top28": set((channels.get("semantic_rank_ids") or [])[:28]),
            "bm25_top28": set((channels.get("bm25_rank_ids") or [])[:28]),
            "entity_top28": set((channels.get("entity_rank_ids") or [])[:28]),
            "prepack": set((trace.get("prepack") or {}).get("leaf_ids") or []),
            "postpack": set((trace.get("postpack") or {}).get("leaf_ids") or []),
        }
        evidence_labels = list(case.get("locomo_evidence") or [])
        context = str(row.get("context_text", "")).casefold()
        sample_index = int(case["locomo_sample_index"])
        exact_hits = sum(
            bool(
                (text := turn_text.get((sample_index, str(label)), "").strip())
                and text.casefold() in context
            )
            for label in evidence_labels
        )
        cat = int(case["locomo_category"])
        exact_text[0] += exact_hits
        exact_text[1] += len(evidence_labels)
        exact_text_macro[0] += exact_hits / len(evidence_labels) if evidence_labels else 0.0
        exact_text_macro[1] += 1
        category[cat]["exact_text"][0] += exact_hits
        category[cat]["exact_text"][1] += len(evidence_labels)
        for stage, selected in sets.items():
            hits = sum(node_id in selected for node_id in evidence_ids)
            totals[stage][0] += hits
            totals[stage][1] += len(evidence_ids)
            category[cat][stage][0] += hits
            category[cat][stage][1] += len(evidence_ids)

    def ratio(pair: list[int]) -> float:
        return pair[0] / pair[1] if pair[1] else 0.0

    result = {
        "question_count": len(cases),
        "evidence_turn_recall": {
            **{stage: ratio(pair) for stage, pair in totals.items()},
            "exact_text_in_context": ratio(exact_text),
            "exact_text_in_context_macro": (
                exact_text_macro[0] / exact_text_macro[1]
                if exact_text_macro[1]
                else 0.0
            ),
        },
        "by_category": {
            str(cat): {stage: ratio(pair) for stage, pair in values.items()}
            for cat, values in sorted(category.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
