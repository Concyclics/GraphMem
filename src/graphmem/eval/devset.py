from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..domain import Conversation, Session, SourceTurn, canonical_json, stable_id
from ..storage import SQLiteGraphStore
from .gold_turns import GoldTurnSet


@dataclass(frozen=True, slots=True)
class GoldTurnRef:
    session_id: str
    turn_index: int
    span_start: int | None = None
    span_end: int | None = None
    support_role: str = "source"


@dataclass(frozen=True, slots=True)
class DevQuestion:
    question_id: str
    memory_id: str
    benchmark: str
    stratum: str
    query: str
    gold_sessions: tuple[str, ...]
    gold_turns: tuple[GoldTurnRef, ...]
    raw: dict[str, Any]


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_dev_questions(lme_path: Path, locomo_path: Path, lme_gold: GoldTurnSet) -> list[DevQuestion]:
    lme_rows = json.loads(lme_path.read_text(encoding="utf-8"))
    locomo_rows = json.loads(locomo_path.read_text(encoding="utf-8"))
    result: list[DevQuestion] = []
    for row in lme_rows:
        question_id = str(row["question_id"])
        annotations = lme_gold.for_question(question_id)
        result.append(DevQuestion(
            question_id=question_id, memory_id=question_id, benchmark="longmemeval",
            stratum="lme_multi_session" if row["question_type"] == "multi-session" else "lme_temporal",
            query=str(row["question"]), gold_sessions=tuple(str(x) for x in row["answer_session_ids"]),
            gold_turns=tuple(GoldTurnRef(x.session_id, x.turn_index, x.span_start, x.span_end,
                                         x.support_role) for x in annotations), raw=row,
        ))
    for row in locomo_rows:
        question_id = str(row["question_id"])
        refs = tuple(parse_locomo_evidence(row.get("locomo_evidence", ())))
        result.append(DevQuestion(
            question_id=question_id, memory_id="locomo:" + str(row["locomo_sample_id"]),
            benchmark="locomo",
            stratum="locomo_multihop" if int(row["locomo_category"]) == 1 else "locomo_temporal",
            query=str(row["question"]), gold_sessions=tuple(str(x) for x in row["answer_session_ids"]),
            gold_turns=refs, raw=row,
        ))
    if len(result) != 200 or len({row.question_id for row in result}) != 200:
        raise ValueError("development set must contain exactly 200 unique questions")
    return result


def parse_locomo_evidence(values: Iterable[str]) -> list[GoldTurnRef]:
    seen: set[tuple[str, int]] = set()
    result: list[GoldTurnRef] = []
    for value in values:
        for part in str(value).split(";"):
            day, turn = part.strip().split(":", 1)
            key = (f"session_{int(day[1:])}", int(turn) - 1)
            if key not in seen:
                seen.add(key)
                result.append(GoldTurnRef(*key))
    return result


def calibration40(questions: Sequence[DevQuestion], seed: int = 42) -> list[DevQuestion]:
    strata = ("lme_multi_session", "lme_temporal", "locomo_multihop", "locomo_temporal")
    selected: list[DevQuestion] = []
    for stratum in strata:
        rows = [row for row in questions if row.stratum == stratum]
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row.question_id}".encode()).hexdigest())
        selected.extend(rows[:10])
    return selected


def conversation_holdout_split(questions: Sequence[DevQuestion], seed: int = 42) -> dict[str, list[DevQuestion]]:
    """Leakage-resistant 6/2/2 LoCoMo and 30/10/10-per-stratum LME split."""
    result = {"development": [], "validation": [], "final": []}
    locomo = [row for row in questions if row.benchmark == "locomo"]
    conversation_ids = sorted({row.memory_id for row in locomo}, key=lambda value:
                              hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    assignment = {value: ("development" if index < 6 else "validation" if index < 8 else "final")
                  for index, value in enumerate(conversation_ids)}
    if len(conversation_ids) != 10:
        raise ValueError("LoCoMo holdout protocol requires exactly 10 conversations")
    for row in locomo:
        result[assignment[row.memory_id]].append(row)
    for stratum in ("lme_multi_session", "lme_temporal"):
        rows = sorted((row for row in questions if row.stratum == stratum), key=lambda row:
                      hashlib.sha256(f"{seed}:{row.question_id}".encode()).hexdigest())
        if len(rows) != 50:
            raise ValueError(f"LME holdout protocol requires 50 questions in {stratum}")
        result["development"].extend(rows[:30]); result["validation"].extend(rows[30:40])
        result["final"].extend(rows[40:])
    for rows in result.values():
        rows.sort(key=lambda row: (row.stratum, row.question_id))
    return result


def ingest_questions(store: SQLiteGraphStore, questions: Sequence[DevQuestion]) -> tuple[str, ...]:
    by_memory: dict[str, DevQuestion] = {}
    for question in questions:
        by_memory.setdefault(question.memory_id, question)
    for memory_id, question in by_memory.items():
        row = question.raw
        sessions_payload = row["haystack_sessions"]
        session_ids = [str(x) for x in row.get("haystack_session_ids", ())]
        dates = [str(x) for x in row.get("haystack_dates", ())]
        sessions: list[Session] = []
        turns: list[SourceTurn] = []
        for session_index, payload in enumerate(sessions_payload):
            session_id = session_ids[session_index] if session_index < len(session_ids) else f"session_{session_index + 1}"
            timestamp = dates[session_index] if session_index < len(dates) else None
            session_hash = _sha(payload)
            sessions.append(Session(session_id, memory_id, session_index, timestamp, session_hash))
            for turn_index, item in enumerate(payload):
                text = str(item.get("content", item.get("text", "")))
                speaker = str(item.get("speaker", item.get("role", "unknown")))
                listener = str(item.get("listener", ""))
                role = str(item.get("role", "unknown"))
                turn_id = stable_id("turn", memory_id, session_id, turn_index)
                turns.append(SourceTurn(
                    turn_id, memory_id, session_id, turn_index, speaker, listener, role,
                    timestamp, text, hashlib.sha256(text.encode("utf-8")).hexdigest(),
                ))
        conversation = Conversation(
            memory_id, question.benchmark, str(row.get("locomo_sample_id", question.question_id)),
            _sha(sessions_payload),
        )
        store.ingest_conversation(conversation, sessions, turns)
    return tuple(sorted(by_memory))
