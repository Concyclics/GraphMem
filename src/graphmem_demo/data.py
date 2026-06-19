from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import LeafNode, QuestionCase


def load_longmemeval_cases(
    path: str | Path,
    question_type: str = "multi-session",
    max_questions: int | None = None,
) -> list[QuestionCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[QuestionCase] = []
    for row in rows:
        if question_type != "all" and row.get("question_type") != question_type:
            continue
        sessions = row.get("haystack_sessions") or []
        session_ids = row.get("haystack_session_ids") or []
        dates = row.get("haystack_dates") or []
        if len(sessions) != len(session_ids) or len(sessions) != len(dates):
            raise ValueError(
                f"LongMemEval row {row.get('question_id')} has misaligned haystack sessions metadata"
            )
        cases.append(
            QuestionCase(
                question_id=str(row["question_id"]),
                question_type=str(row["question_type"]),
                question=str(row["question"]),
                answer=row.get("answer"),
                question_date=row.get("question_date"),
                haystack_sessions=sessions,
                haystack_session_ids=[str(value) for value in session_ids],
                haystack_dates=dates,
                answer_session_ids=[str(value) for value in row.get("answer_session_ids") or []],
                memory_cache_key=(
                    f"locomo:{row['locomo_sample_id']}"
                    if row.get("locomo_sample_id") is not None
                    else None
                ),
            )
        )
        if max_questions is not None and len(cases) >= max_questions:
            break
    return cases


def build_leaf_nodes(case: QuestionCase) -> list[LeafNode]:
    leaves: list[LeafNode] = []
    for session_id, session_date, messages in zip(
        case.haystack_session_ids, case.haystack_dates, case.haystack_sessions
    ):
        for pair_index, (turn_index, chunk) in enumerate(_leaf_chunks(messages)):
            raw_text = _format_messages(chunk)
            leaves.append(
                LeafNode(
                    node_id=f"{case.question_id}:{session_id}:leaf:{pair_index}",
                    question_id=case.question_id,
                    session_id=session_id,
                    session_date=session_date,
                    turn_index=turn_index,
                    raw_text=raw_text,
                    user_text=_format_user_messages(chunk),
                    message_count=len(chunk),
                    retrieval_text=_format_retrieval_messages(chunk, raw_text),
                )
            )
    return leaves


def group_by_session(leaves: Iterable[LeafNode]) -> dict[str, list[LeafNode]]:
    grouped: dict[str, list[LeafNode]] = {}
    for leaf in leaves:
        grouped.setdefault(leaf.session_id, []).append(leaf)
    return grouped


def _leaf_chunks(messages: list[dict[str, Any]]) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    index = 0
    while index < len(messages):
        message = messages[index]
        role = str(message.get("role", "unknown")).lower()
        if role == "user" and index + 1 < len(messages):
            next_message = messages[index + 1]
            if str(next_message.get("role", "")).lower() == "assistant":
                yield index, [message, next_message]
                index += 2
                continue
        yield index, [message]
        index += 1


def _format_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "unknown")).strip().title() or "Unknown"
        speaker = str(message.get("speaker", "")).strip()
        listener = str(message.get("listener", "")).strip()
        if speaker:
            role = f"{role} ({speaker})"
            if listener:
                role = f"{role} -> {listener}"
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_user_messages(messages: list[dict[str, Any]]) -> str:
    user_messages = [
        message for message in messages if str(message.get("role", "")).lower() == "user"
    ]
    return _format_messages(user_messages or messages)


def _format_retrieval_messages(messages: list[dict[str, Any]], raw_text: str) -> str:
    normalized_lines = []
    for message in messages:
        speaker = str(message.get("speaker", "")).strip()
        content = str(message.get("content", "")).strip()
        if not speaker or not content:
            continue
        normalized = _speaker_retrieval_line(speaker, content)
        if normalized:
            normalized_lines.append(normalized)
    if not normalized_lines:
        return raw_text
    return raw_text + "\n\nSpeaker search cues:\n" + "\n".join(normalized_lines)


def speaker_retrieval_text_from_raw(raw_text: str) -> str:
    normalized_lines = []
    for line in raw_text.splitlines():
        match = re.match(r"^[A-Za-z]+ \(([^)]+)\)(?: -> [^:]+)?:\s*(.*)$", line.strip())
        if not match:
            continue
        normalized = _speaker_retrieval_line(match.group(1), match.group(2))
        if normalized:
            normalized_lines.append(normalized)
    if not normalized_lines:
        return raw_text
    return raw_text + "\n\nSpeaker search cues:\n" + "\n".join(normalized_lines)


def _speaker_retrieval_line(speaker: str, content: str) -> str:
    content = re.sub(r"\s+", " ", content).strip()
    if not content:
        return ""
    # Keep the original first-person wording, but anchor it to the explicit
    # speaker name so embedding search can match questions phrased in third person.
    return f"{speaker} said: {content}"
