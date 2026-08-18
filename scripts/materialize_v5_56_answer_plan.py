#!/usr/bin/env python3
"""Apply the core V5.56 AnswerPlan compiler to frozen PreparedAnswer rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.answer_plan import apply_answer_plan  # noqa: E402
from graphmem.answer.stage import PreparedAnswer  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packing-model", required=True)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--excerpt-chars", type=int, default=440)
    parser.add_argument("--max-prompt-tokens", type=int, default=12000)
    parser.add_argument(
        "--kind", action="append", dest="kinds",
        choices=("date_difference", "relative_time", "age_projection",
                 "latest_state", "temporal_lookup", "temporal_order"),
        help=("eligible AnswerPlan kind; repeat to select several; defaults "
              "to date_difference and temporal_order"))
    parser.add_argument("--question-ids", type=Path)
    parser.add_argument("--retrieval-all-hit", type=Path)
    args = parser.parse_args()
    if args.question_ids and args.retrieval_all_hit:
        raise ValueError("use only one selector")

    rows = read_jsonl(args.prepared)
    selected_ids: set[str] | None = None
    if args.question_ids:
        selected_ids = {line.strip() for line in args.question_ids.read_text(
            encoding="utf-8").splitlines() if line.strip()}
    elif args.retrieval_all_hit:
        selected_ids = {
            str(row["dev_question_id"])
            for row in read_jsonl(args.retrieval_all_hit)
            if row.get("benchmark") == "longmemeval"
            and bool(row.get("has_turn_gold"))
            and bool(row.get("turn_all_hit"))
        }
    if selected_ids is not None:
        rows = [row for row in rows if str(row["question_id"]) in selected_ids]

    counter = resolve_token_counter(args.packing_model, require_exact=True)
    output_rows: list[dict] = []
    applied = 0
    kinds: dict[str, int] = {}
    token_deltas: list[int] = []
    for source in rows:
        prepared = PreparedAnswer.from_record(source)
        updated = apply_answer_plan(
            prepared, counter, max_candidates=args.max_candidates,
            excerpt_chars=args.excerpt_chars,
            max_prompt_tokens=args.max_prompt_tokens,
            enabled_kinds=tuple(args.kinds or (
                "date_difference", "temporal_order")))
        record = updated.to_record()
        if record["prompt_payload_hash"] != source["prompt_payload_hash"]:
            applied += 1
            plan = record["trace"]["answer_plan"]
            kind = str(plan["kind"])
            kinds[kind] = kinds.get(kind, 0) + 1
            token_deltas.append(
                int(record["packing_prompt_tokens"])
                - int(source["packing_prompt_tokens"]))
        output_rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in output_rows),
        encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.56-answer-plan-materialization-v1",
        "source": str(args.prepared),
        "source_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "questions": len(rows),
        "applied": applied,
        "unchanged": len(rows) - applied,
        "kinds": dict(sorted(kinds.items())),
        "max_candidates": args.max_candidates,
        "excerpt_chars": args.excerpt_chars,
        "max_prompt_tokens": args.max_prompt_tokens,
        "enabled_kinds": list(args.kinds or (
            "date_difference", "temporal_order")),
        "token_delta": {
            "mean": sum(token_deltas) / max(1, len(token_deltas)),
            "max": max(token_deltas, default=0),
        },
        "transform_reads_gold": False,
        "gold_used_only_for_diagnostic_membership": bool(
            args.retrieval_all_hit),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
