"""Full-benchmark question loading: LongMemEval 500 and LoCoMo Cat 1-4.

``devset.py`` asserts exactly 200 unique questions, which is correct for the
frozen development set and wrong for a benchmark run, so this is a separate
loader rather than a relaxation of that one.

Two flags exist because the 200-question path never needed them and the full
set cannot be scored without them:

* ``has_turn_gold`` -- turn annotations cover only 100 of the 500 LongMemEval
  questions.  ``gold <= predicted`` is *vacuously true* when a question has no
  annotated turns, so averaging ``turn_all_hit`` over the full set would report
  a number inflated by 400 undefined rows.
* ``is_abstention`` -- 30 LongMemEval questions have no correct evidence at all;
  the answer is that the memory does not contain it.  Retrieval metrics are
  meaningless on them and the judge needs to score them as abstentions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .devset import DevQuestion, GoldTurnRef
from .gold_turns import GoldTurnSet

#: LoCoMo category 5 is excluded upstream by mem0's own benchmark harness, so
#: including it here would make our number incomparable to every published one.
LOCOMO_CATEGORIES = (1, 2, 3, 4)
LOCOMO_EXPECTED = 1540
LME_EXPECTED = 500


@dataclass(frozen=True, slots=True)
class FullQuestion:
    """A DevQuestion plus the two flags the full set needs."""

    question: DevQuestion
    has_turn_gold: bool
    is_abstention: bool

    def __getattr__(self, name: str) -> Any:
        return getattr(self.question, name)


def parse_locomo_evidence_lenient(values) -> list[GoldTurnRef]:
    """Parse LoCoMo evidence, skipping refs the full set holds but the 200 did not.

    ``devset.parse_locomo_evidence`` raises on anything that is not ``D<n>:<turn>``,
    which is correct for the frozen Cat 1-2 development set.  The full Cat 1-4 set
    holds one bare ``"D"`` in category 4, and four questions carry no evidence at
    all -- crashing over one malformed reference would cost all 1,540 questions.
    """
    seen: set[tuple[str, int]] = set()
    result: list[GoldTurnRef] = []
    for value in values or ():
        for part in str(value).split(";"):
            piece = part.strip()
            if ":" not in piece:
                continue
            day, turn = piece.split(":", 1)
            try:
                key = (f"session_{int(day[1:])}", int(turn) - 1)
            except ValueError:
                continue
            if key not in seen:
                seen.add(key)
                result.append(GoldTurnRef(*key))
    return result


def _lme_stratum(question_type: str) -> str:
    return "lme_" + str(question_type).replace("-", "_")


def load_full_questions(
    lme_path: Path | None = None,
    locomo_path: Path | None = None,
    lme_gold: GoldTurnSet | None = None,
    *,
    lme_types: Sequence[str] | None = None,
    locomo_categories: Sequence[int] = LOCOMO_CATEGORIES,
    expect_lme: int | None = LME_EXPECTED,
    expect_locomo: int | None = LOCOMO_EXPECTED,
) -> list[FullQuestion]:
    result: list[FullQuestion] = []

    if lme_path is not None:
        rows = json.loads(Path(lme_path).read_text(encoding="utf-8"))
        if lme_types:
            rows = [row for row in rows if str(row["question_type"]) in set(lme_types)]
        if expect_lme is not None and not lme_types and len(rows) != expect_lme:
            raise ValueError(f"expected {expect_lme} LongMemEval questions, found {len(rows)}")
        for row in rows:
            question_id = str(row["question_id"])
            annotations = lme_gold.for_question(question_id) if lme_gold else ()
            result.append(FullQuestion(
                question=DevQuestion(
                    question_id=question_id, memory_id=question_id, benchmark="longmemeval",
                    stratum=_lme_stratum(row["question_type"]), query=str(row["question"]),
                    gold_sessions=tuple(str(item) for item in row["answer_session_ids"]),
                    gold_turns=tuple(GoldTurnRef(item.session_id, item.turn_index,
                                                 item.span_start, item.span_end,
                                                 item.support_role) for item in annotations),
                    raw=row),
                has_turn_gold=bool(annotations),
                is_abstention="_abs" in question_id,
            ))

    if locomo_path is not None:
        rows = json.loads(Path(locomo_path).read_text(encoding="utf-8"))
        allowed = set(locomo_categories)
        rows = [row for row in rows if int(row["locomo_category"]) in allowed]
        if expect_locomo is not None and len(rows) != expect_locomo:
            raise ValueError(f"expected {expect_locomo} LoCoMo questions, found {len(rows)}")
        for row in rows:
            refs = tuple(parse_locomo_evidence_lenient(row.get("locomo_evidence", ())))
            result.append(FullQuestion(
                question=DevQuestion(
                    question_id=str(row["question_id"]),
                    memory_id="locomo:" + str(row["locomo_sample_id"]), benchmark="locomo",
                    stratum=f"locomo_cat{int(row['locomo_category'])}",
                    query=str(row["question"]),
                    gold_sessions=tuple(str(item) for item in row["answer_session_ids"]),
                    gold_turns=refs, raw=row),
                # LoCoMo ships evidence for every question, so turn gold is native.
                has_turn_gold=bool(refs),
                is_abstention=False,
            ))

    seen = {row.question.question_id for row in result}
    if len(seen) != len(result):
        raise ValueError("full question set contains duplicate question ids")
    return result


def memory_ids(questions: Sequence[FullQuestion]) -> tuple[str, ...]:
    """Distinct memories the set needs, in a stable order."""
    return tuple(sorted({row.question.memory_id for row in questions}))
