#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build answer-evidence location index for LoCoMo/LongMemEval."
    )
    parser.add_argument(
        "--locomo-graphmem",
        type=Path,
        default=Path("data/locomo10_graphmem.json"),
        help="Converted LoCoMo GraphMem-format dataset.",
    )
    parser.add_argument(
        "--locomo-raw",
        type=Path,
        default=None,
        help="Original LoCoMo raw dataset with QA evidence refs.",
    )
    parser.add_argument(
        "--longmemeval",
        type=Path,
        default=Path("data/longmemeval_s_cleaned.json"),
        help="LongMemEval cleaned dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evidence_index"),
        help="Output directory for generated evidence index files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _leaf_pair_index_for_message(messages: list[dict[str, Any]], message_index: int) -> int | None:
    index = 0
    pair_index = 0
    while index < len(messages):
        role = str(messages[index].get("role", "")).lower()
        chunk_len = 1
        if (
            role == "user"
            and index + 1 < len(messages)
            and str(messages[index + 1].get("role", "")).lower() == "assistant"
        ):
            chunk_len = 2
        if index <= message_index < index + chunk_len:
            return pair_index
        index += chunk_len
        pair_index += 1
    return None


def _parse_locomo_evidence_ref(ref: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"D(\d+):(\d+)", str(ref).strip())
    if not match:
        return None
    session_idx = int(match.group(1)) - 1
    message_idx = int(match.group(2)) - 1
    if session_idx < 0 or message_idx < 0:
        return None
    return session_idx, message_idx


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", _normalize_text(text)) if len(token) >= 2}


def _is_insufficient_answer(answer: str) -> bool:
    return bool(
        re.search(
            r"insufficient|not enough|cannot determine|unknown|not mentioned|does not mention",
            answer,
            flags=re.IGNORECASE,
        )
    )


def build_locomo_index(
    graphmem_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in graphmem_rows:
        sample_id = row.get("locomo_sample_id")
        if isinstance(sample_id, int):
            by_sample[sample_id].append(row)

    index_rows: list[dict[str, Any]] = []
    exact_count = 0
    missing_evidence_count = 0

    for sample_id, rows in sorted(by_sample.items(), key=lambda item: item[0]):
        if sample_id >= len(raw_rows):
            for row in rows:
                index_rows.append(
                    {
                        "dataset": "locomo",
                        "question_id": row["question_id"],
                        "question_type": row.get("question_type"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                        "answer_session_ids": row.get("answer_session_ids") or [],
                        "evidence_refs": [],
                        "evidence_status": "raw_sample_missing",
                    }
                )
                missing_evidence_count += 1
            continue

        sample = raw_rows[sample_id]
        qa_rows = sample.get("qa") or []
        qa_by_question = defaultdict(list)
        for qa in qa_rows:
            qa_by_question[_normalize_text(str(qa.get("question") or ""))].append(qa)

        for row in rows:
            q_norm = _normalize_text(str(row.get("question") or ""))
            a_norm = _normalize_text(str(row.get("answer") or ""))
            candidates = qa_by_question.get(q_norm, [])
            qa_match = None
            for qa in candidates:
                if _normalize_text(str(qa.get("answer") or "")) == a_norm:
                    qa_match = qa
                    break
            if qa_match is None and candidates:
                qa_match = candidates[0]

            if qa_match is None:
                index_rows.append(
                    {
                        "dataset": "locomo",
                        "question_id": row["question_id"],
                        "question_type": row.get("question_type"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                        "answer_session_ids": row.get("answer_session_ids") or [],
                        "evidence_refs": [],
                        "evidence_status": "qa_match_missing",
                    }
                )
                missing_evidence_count += 1
                continue

            evidence_refs = []
            session_ids = row.get("haystack_session_ids") or []
            sessions = row.get("haystack_sessions") or []
            dates = row.get("haystack_dates") or []

            for ref in qa_match.get("evidence") or []:
                parsed = _parse_locomo_evidence_ref(ref)
                if parsed is None:
                    continue
                session_idx, message_idx = parsed
                if not (0 <= session_idx < len(session_ids)) or not (
                    0 <= session_idx < len(sessions)
                ):
                    continue
                messages = sessions[session_idx] or []
                if not (0 <= message_idx < len(messages)):
                    continue
                session_id = str(session_ids[session_idx])
                session_date = dates[session_idx] if session_idx < len(dates) else None
                pair_index = _leaf_pair_index_for_message(messages, message_idx)
                leaf_id = (
                    f"{row['question_id']}:{session_id}:leaf:{pair_index}"
                    if pair_index is not None
                    else None
                )
                message = messages[message_idx] or {}
                evidence_refs.append(
                    {
                        "source_ref": str(ref),
                        "session_id": session_id,
                        "session_date": session_date,
                        "message_index": message_idx,
                        "leaf_pair_index": pair_index,
                        "leaf_id": leaf_id,
                        "speaker": message.get("speaker"),
                        "role": message.get("role"),
                        "content": str(message.get("content") or ""),
                        "confidence": "exact_from_locomo_evidence",
                    }
                )

            status = "exact" if evidence_refs else "evidence_refs_missing"
            if status == "exact":
                exact_count += 1
            else:
                missing_evidence_count += 1

            index_rows.append(
                {
                    "dataset": "locomo",
                    "question_id": row["question_id"],
                    "question_type": row.get("question_type"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "answer_session_ids": row.get("answer_session_ids") or [],
                    "evidence_refs": evidence_refs,
                    "evidence_status": status,
                }
            )

    summary = {
        "dataset": "locomo",
        "rows": len(index_rows),
        "exact_rows": exact_count,
        "missing_or_unmatched_rows": missing_evidence_count,
    }
    return index_rows, summary


def build_longmemeval_index(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    exact_like = 0
    session_only = 0

    for row in rows:
        question_id = str(row.get("question_id"))
        answer = str(row.get("answer") or "")
        q_tokens = _tokenize(str(row.get("question") or ""))
        answer_tokens = _tokenize(answer)
        is_insufficient = _is_insufficient_answer(answer)
        answer_sessions = {str(value) for value in (row.get("answer_session_ids") or [])}
        session_ids = row.get("haystack_session_ids") or []
        sessions = row.get("haystack_sessions") or []
        dates = row.get("haystack_dates") or []

        evidence_refs = []
        if not is_insufficient and answer_tokens:
            for sidx, (session_id, messages) in enumerate(zip(session_ids, sessions)):
                sid = str(session_id)
                if answer_sessions and sid not in answer_sessions:
                    continue
                for midx, message in enumerate(messages or []):
                    content = str(message.get("content") or "")
                    content_norm = _normalize_text(content)
                    content_tokens = _tokenize(content)
                    pair_index = _leaf_pair_index_for_message(messages or [], midx)
                    leaf_id = (
                        f"{question_id}:{sid}:leaf:{pair_index}" if pair_index is not None else None
                    )

                    confidence = None
                    reason = None
                    if answer and _normalize_text(answer) in content_norm:
                        confidence = "high_exact_substring"
                        reason = "gold answer substring appears in turn"
                    elif answer_tokens and len(answer_tokens & content_tokens) >= max(
                        2, int(0.6 * len(answer_tokens))
                    ):
                        confidence = "medium_token_overlap"
                        reason = "high overlap between gold answer tokens and turn"
                    elif q_tokens and answer_tokens and len((q_tokens | answer_tokens) & content_tokens) >= max(
                        3, int(0.4 * len(q_tokens | answer_tokens))
                    ):
                        confidence = "low_question_answer_overlap"
                        reason = "partial overlap with question+answer tokens"

                    if confidence is None:
                        continue

                    evidence_refs.append(
                        {
                            "source_ref": f"S{sidx + 1}:M{midx + 1}",
                            "session_id": sid,
                            "session_date": dates[sidx] if sidx < len(dates) else None,
                            "message_index": midx,
                            "leaf_pair_index": pair_index,
                            "leaf_id": leaf_id,
                            "speaker": message.get("speaker"),
                            "role": message.get("role"),
                            "content": content,
                            "confidence": confidence,
                            "reason": reason,
                        }
                    )

        evidence_status = "heuristic_turn_match" if evidence_refs else "session_only"
        if evidence_status == "heuristic_turn_match":
            exact_like += 1
        else:
            session_only += 1

        index_rows.append(
            {
                "dataset": "longmemeval",
                "question_id": question_id,
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "answer_session_ids": row.get("answer_session_ids") or [],
                "evidence_refs": evidence_refs,
                "evidence_status": evidence_status,
                "note": (
                    "LongMemEval cleaned has no gold turn-level evidence field; refs are heuristic."
                ),
            }
        )

    summary = {
        "dataset": "longmemeval",
        "rows": len(index_rows),
        "heuristic_turn_match_rows": exact_like,
        "session_only_rows": session_only,
    }
    return index_rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    locomo_graphmem_rows = load_json(args.locomo_graphmem)
    longmemeval_rows = load_json(args.longmemeval)

    locomo_index: list[dict[str, Any]] = []
    locomo_summary: dict[str, Any] = {"skipped": True, "reason": "no --locomo-raw provided"}
    if args.locomo_raw is not None:
        locomo_raw_rows = load_json(args.locomo_raw)
        locomo_index, locomo_summary = build_locomo_index(locomo_graphmem_rows, locomo_raw_rows)
    long_index, long_summary = build_longmemeval_index(longmemeval_rows)

    locomo_out = args.output_dir / "locomo_answer_evidence_index.jsonl"
    long_out = args.output_dir / "longmemeval_answer_evidence_index.jsonl"
    summary_out = args.output_dir / "answer_evidence_index_summary.json"

    write_jsonl(locomo_out, locomo_index)
    write_jsonl(long_out, long_index)

    summary = {
        "locomo": locomo_summary,
        "longmemeval": long_summary,
        "outputs": {
            "locomo_index": str(locomo_out),
            "longmemeval_index": str(long_out),
        },
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
