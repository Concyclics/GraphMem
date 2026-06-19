#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Locomo turn-level evidence coverage.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_qid(qid: str) -> str:
    return qid.removesuffix("_abs")


def parse_evidence_ref(ref: str, session_ids: list[str]) -> tuple[str, int] | None:
    match = re.fullmatch(r"D(\d+):(\d+)", str(ref).strip())
    if not match:
        return None
    session_index = int(match.group(1)) - 1
    message_index = int(match.group(2)) - 1
    if session_index < 0 or session_index >= len(session_ids):
        return None
    return session_ids[session_index], message_index


def prediction_abstains(text: str) -> bool:
    return bool(
        re.search(
            r"insufficient|not enough|cannot determine|not mentioned|does not mention|"
            r"no evidence|unknown|can't determine|do not know",
            text,
            flags=re.IGNORECASE,
        )
    )


def leaf_covers_turn(node: dict[str, Any], message_index: int) -> bool:
    start = int(node.get("turn_index") or 0)
    count = int(node.get("message_count") or 1)
    return start <= message_index < start + count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = {str(row["question_id"]): row for row in read_json_or_jsonl(args.data)}
    answers = {str(row["question_id"]): row for row in read_jsonl(args.answers)}
    evals = {
        normalize_qid(str(row["question_id"])): row
        for row in read_json_or_jsonl(args.eval_results)
    }
    retrieval = {str(row["question_id"]): row for row in read_jsonl(args.retrieval_results)}
    nodes_by_qid = defaultdict(dict)
    for row in read_jsonl(args.nodes):
        if row.get("node_type") != "leaf":
            continue
        nodes_by_qid[str(row["question_id"])][str(row["node_id"])] = row

    rows: list[dict[str, Any]] = []
    for qid, case in cases.items():
        answer_row = answers.get(qid, {})
        eval_row = evals.get(qid, {})
        ret_row = retrieval.get(qid, {})
        prediction = str(answer_row.get("prediction") or "")
        gold = str(case.get("answer") or "")
        evidence_refs = [
            parsed
            for item in case.get("locomo_evidence") or []
            if (parsed := parse_evidence_ref(str(item), list(case.get("haystack_session_ids") or [])))
            is not None
        ]
        leaf_nodes = nodes_by_qid.get(qid, {})
        retrieved_leaf_ids = [str(node_id) for node_id in ret_row.get("leaf_node_ids") or []]
        covered_refs = []
        missing_refs = []
        for session_id, message_index in evidence_refs:
            covered = any(
                (node := leaf_nodes.get(node_id))
                and str(node.get("session_id")) == session_id
                and leaf_covers_turn(node, message_index)
                for node_id in retrieved_leaf_ids
            )
            if covered:
                covered_refs.append((session_id, message_index))
            else:
                missing_refs.append((session_id, message_index))

        correct = bool(eval_row.get("autoeval_label", {}).get("label"))
        evidence_count = len(evidence_refs)
        evidence_coverage = len(covered_refs) / evidence_count if evidence_count else None
        empty_gold = not gold.strip()
        abstains = prediction_abstains(prediction)
        session_all_hit = bool(ret_row.get("answer_session_all_hit"))

        if correct:
            cause = "correct"
        elif empty_gold and not abstains:
            cause = "answerability_empty_gold_not_abstained"
        elif not session_all_hit:
            cause = "retrieval_session_miss"
        elif evidence_count and missing_refs:
            cause = "retrieval_turn_miss"
        elif abstains and not empty_gold:
            cause = "over_abstain"
        else:
            cause = "generation_or_reasoning_error"

        rows.append(
            {
                "question_id": qid,
                "question_type": case.get("question_type"),
                "locomo_sample_id": case.get("locomo_sample_id"),
                "question": case.get("question"),
                "gold_answer": case.get("answer"),
                "correct": correct,
                "cause": cause,
                "answer_session_all_hit": session_all_hit,
                "answer_session_recall": ret_row.get("answer_session_recall"),
                "evidence_ref_count": evidence_count,
                "evidence_turn_covered_count": len(covered_refs),
                "evidence_turn_coverage": evidence_coverage,
                "empty_gold": empty_gold,
                "prediction_abstains": abstains,
                "prediction_preview": " ".join(prediction.split())[:300],
            }
        )

    summary: dict[str, Any] = {
        "question_count": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
        "cause_counts": Counter(row["cause"] for row in rows),
        "by_type": {},
    }
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["question_type"])].append(row)
    for question_type, type_rows in sorted(by_type.items()):
        evidence_values = [
            row["evidence_turn_coverage"]
            for row in type_rows
            if row["evidence_turn_coverage"] is not None
        ]
        summary["by_type"][question_type] = {
            "rows": len(type_rows),
            "correct": sum(row["correct"] for row in type_rows),
            "accuracy": sum(row["correct"] for row in type_rows) / len(type_rows),
            "session_all_hit": sum(row["answer_session_all_hit"] for row in type_rows),
            "avg_evidence_turn_coverage": (
                sum(evidence_values) / len(evidence_values) if evidence_values else None
            ),
            "cause_counts": Counter(row["cause"] for row in type_rows),
        }

    def json_default(value: Any) -> Any:
        if isinstance(value, Counter):
            return dict(value)
        raise TypeError(type(value))

    (args.output_dir / "locomo_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    with (args.output_dir / "locomo_error_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
