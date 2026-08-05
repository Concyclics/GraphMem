#!/usr/bin/env python3
"""Build reproducible hard multi-session/temporal development sets.

The selector only consumes frozen benchmark data, answers, retrieval statistics,
and judge outputs. It never changes source datasets or frozen run artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate question_id encountered")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def lme_selection(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = load_json(args.lme_data)
    data_by_id = by_id(data)
    v41_judge = by_id(load_jsonl(args.lme_v41_judge))
    v2_judge = by_id(load_jsonl(args.lme_v2_judge))
    retrieval = by_id(load_jsonl(args.lme_v41_retrieval))
    stats = by_id(load_jsonl(args.lme_v41_stats))

    selected_meta: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for question_type, label in (
        ("multi-session", "multi_session"),
        ("temporal-reasoning", "temporal"),
    ):
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for row in data:
            if row.get("question_type") != question_type:
                continue
            qid = str(row["question_id"])
            wrong_v41 = not bool(v41_judge[qid]["correct"])
            wrong_v2 = not bool(v2_judge[qid]["correct"])
            recall = float(retrieval[qid].get("answer_session_recall", 0.0))
            query_tokens = int(stats[qid].get("answer_total_tokens", 0))
            wrong_count = int(wrong_v41) + int(wrong_v2)
            rank_key = (-wrong_count, -int(wrong_v41), recall, -query_tokens, qid)
            candidates.append(
                (
                    rank_key,
                    {
                        "benchmark": "longmemeval",
                        "question_id": qid,
                        "dev_type": label,
                        "native_question_type": question_type,
                        "selection_reason": (
                            "judge_error" if wrong_count else "hard_correct_fill"
                        ),
                        "wrong_count": wrong_count,
                        "v41_correct": not wrong_v41,
                        "v2_ledger_correct": not wrong_v2,
                        "v41_answer_session_recall": recall,
                        "v41_query_tokens": query_tokens,
                    },
                )
            )
        chosen = [meta for _, meta in sorted(candidates)[:50]]
        selected_meta.extend(chosen)
        selected_ids.update(meta["question_id"] for meta in chosen)

    selected_rows = [row for row in data if str(row["question_id"]) in selected_ids]
    selected_rows.sort(
        key=lambda row: (
            0 if row["question_type"] == "multi-session" else 1,
            str(row["question_id"]),
        )
    )
    selected_meta.sort(key=lambda row: (row["dev_type"], row["question_id"]))
    if len(selected_rows) != 100 or len(selected_ids) != 100:
        raise ValueError("LongMemEval selection must contain 100 unique questions")
    return selected_rows, selected_meta


def locomo_selection(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = [row for row in load_json(args.locomo_data) if int(row["locomo_category"]) < 5]
    data_by_id = by_id(data)
    judge = by_id(load_jsonl(args.locomo_judge))
    answers = by_id(load_jsonl(args.locomo_answers))

    selected_by_type: dict[str, set[str]] = {}
    for dev_type, category in (("multi_hop", 1), ("temporal", 2)):
        candidates: list[tuple[tuple[Any, ...], str]] = []
        for qid, row in data_by_id.items():
            if int(row["locomo_category"]) != category:
                continue
            answer = answers[qid]
            wrong = not bool(judge[qid]["correct"])
            recall = float(answer.get("retrieved_answer_session_recall", 0.0))
            parse_error = bool(answer.get("navigation_parse_error", False))
            tokens = int(answer.get("answer_total_tokens", 0))
            rank_key = (-int(wrong), recall, -int(parse_error), -tokens, qid)
            candidates.append((rank_key, qid))
        chosen = {qid for _, qid in sorted(candidates)[:50]}
        if len(chosen) != 50:
            raise ValueError(f"LoCoMo {dev_type} selection must contain 50 questions")
        selected_by_type[dev_type] = chosen

    meta: list[dict[str, Any]] = []
    for dev_type, ids in selected_by_type.items():
        for qid in sorted(ids):
            row = data_by_id[qid]
            answer = answers[qid]
            meta.append(
                {
                    "benchmark": "locomo",
                    "question_id": qid,
                    "dev_type": dev_type,
                    "native_question_type": row["question_type"],
                    "locomo_category": row["locomo_category"],
                    "selection_reason": (
                        "judge_error" if not bool(judge[qid]["correct"]) else "hard_correct_fill"
                    ),
                    "v34_correct": bool(judge[qid]["correct"]),
                    "gold_session_count": len(row.get("answer_session_ids", [])),
                    "v34_answer_session_recall": float(
                        answer.get("retrieved_answer_session_recall", 0.0)
                    ),
                    "v34_navigation_operation": answer.get("navigation_operation"),
                    "v34_navigation_parse_error": bool(
                        answer.get("navigation_parse_error", False)
                    ),
                    "v34_answer_tokens": int(answer.get("answer_total_tokens", 0)),
                }
            )

    selected_ids = set().union(*selected_by_type.values())
    selected_rows = [row for row in data if str(row["question_id"]) in selected_ids]
    selected_rows.sort(
        key=lambda row: (
            0 if str(row["question_id"]) in selected_by_type["multi_hop"] else 1,
            str(row["question_id"]),
        )
    )
    if len(selected_rows) != 100 or len(selected_ids) != 100:
        raise ValueError("LoCoMo selection must contain 100 unique questions")
    return selected_rows, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lme-data", type=Path, required=True)
    parser.add_argument("--lme-v41-judge", type=Path, required=True)
    parser.add_argument("--lme-v2-judge", type=Path, required=True)
    parser.add_argument("--lme-v41-retrieval", type=Path, required=True)
    parser.add_argument("--lme-v41-stats", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--locomo-judge", type=Path, required=True)
    parser.add_argument("--locomo-answers", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lme_rows, lme_meta = lme_selection(args)
    locomo_rows, locomo_meta = locomo_selection(args)
    lme_path = args.output_dir / "longmemeval_hard_multisession50_temporal50.json"
    locomo_path = args.output_dir / "locomo_hard_cat1_multihop50_cat2_temporal50.json"
    selection_path = args.output_dir / "selection_200.jsonl"
    write_json(lme_path, lme_rows)
    write_json(locomo_path, locomo_rows)
    write_jsonl(selection_path, lme_meta + locomo_meta)

    manifest = {
        "name": "graphmem_hard_multisession_temporal_dev_200",
        "selection_policy": {
            "longmemeval": (
                "Within each native type, rank by wrong count across frozen V4.1 and "
                "V2-ledger judges, then V4.1 error, lower answer-session recall, and "
                "higher query tokens. Each type has 49 error-union questions; the "
                "50th is explicitly marked hard_correct_fill."
            ),
            "locomo": (
                "Use the benchmark's official category semantics: Category 1 is multi-hop "
                "and Category 2 is temporal. Within each category rank V3.4 judge errors "
                "first, then lower answer-session recall, navigation parse error, higher "
                "answer tokens, and stable question_id. Category 1 has 47 judged errors, "
                "so three rows are explicitly marked hard_correct_fill; Category 2 has "
                "enough errors to select all 50 from judged errors."
            ),
        },
        "counts": {
            "total": 200,
            "longmemeval": len(lme_rows),
            "locomo": len(locomo_rows),
            "longmemeval_by_type": dict(Counter(row["dev_type"] for row in lme_meta)),
            "locomo_by_type": dict(Counter(row["dev_type"] for row in locomo_meta)),
            "selection_reasons": dict(
                Counter(row["selection_reason"] for row in lme_meta + locomo_meta)
            ),
        },
        "files": {
            lme_path.name: sha256(lme_path),
            locomo_path.name: sha256(locomo_path),
            selection_path.name: sha256(selection_path),
        },
        "sources": {key: str(value) for key, value in vars(args).items() if key != "output_dir"},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
