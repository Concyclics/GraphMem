from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GoldTurnAnnotation:
    question_id: str
    session_id: str
    turn_index: int
    span_start: int
    span_end: int
    support_role: str
    confidence: str
    first_review: str
    second_review: str
    adjudication: str
    first_reviewer: str = "unspecified"
    second_reviewer: str = "not_required"
    disagreement: str = "none"
    annotation_version: str = "lme-v5-dev100-r1"

    def __post_init__(self) -> None:
        if self.turn_index < 0 or self.span_start < 0 or self.span_end <= self.span_start:
            raise ValueError("invalid gold turn/span reference")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("invalid annotation confidence")
        if self.first_review != "accepted":
            raise ValueError("final annotation must pass first review")
        if self.second_review not in {"not_required", "accepted", "changed"}:
            raise ValueError("invalid second review status")
        if self.adjudication not in {"accepted", "changed", "excluded"}:
            raise ValueError("invalid adjudication status")
        if self.confidence != "high" and self.second_review == "not_required":
            raise ValueError("non-high-confidence annotation requires second review")
        if self.second_review != "not_required" and self.second_reviewer == "not_required":
            raise ValueError("second review requires a reviewer identifier")


@dataclass(frozen=True, slots=True)
class GoldTurnSet:
    annotations: tuple[GoldTurnAnnotation, ...]

    @property
    def question_ids(self) -> set[str]:
        return {row.question_id for row in self.annotations}

    def for_question(self, question_id: str) -> tuple[GoldTurnAnnotation, ...]:
        return tuple(row for row in self.annotations if row.question_id == question_id)

    def validate_question_coverage(self, expected: Iterable[str]) -> None:
        expected_ids = set(expected)
        missing = expected_ids - self.question_ids
        extra = self.question_ids - expected_ids
        if missing or extra:
            raise ValueError(f"gold turn question mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def load_gold_turns(path: Path) -> GoldTurnSet:
    rows: list[GoldTurnAnnotation] = []
    seen: set[tuple[str, str, int, int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = GoldTurnAnnotation(**json.loads(line))
            key = (row.question_id, row.session_id, row.turn_index, row.span_start, row.span_end)
            if key in seen:
                raise ValueError(f"duplicate gold turn annotation: {key}")
            seen.add(key)
            rows.append(row)
    return GoldTurnSet(tuple(rows))
