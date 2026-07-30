from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .schema import QueryFrame


_CONTRAST_RE = re.compile(
    r"\b(?:instead\s+of|rather\s+than)\s+(?P<target>[^?.,;]+)",
    re.IGNORECASE,
)
_DEFER_RE = re.compile(
    r"\b(?:(?:put|puts|putting)\s+off|avoid(?:s|ed|ing)?|"
    r"delay(?:s|ed|ing)?|postpone(?:s|d|ing)?|skip(?:s|ped|ping)?)\s+"
    r"(?:doing|resuming|starting|continuing|going\s+to|taking\s+up|to)?\s*"
    r"(?P<target>[^?.,;]+)",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(?:plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|"
    r"decid(?:e|es|ed|ing)|choose|chooses|chose|choosing|"
    r"agree(?:s|d|ing)?|want(?:s|ed|ing)?)\s+"
    r"(?:that\s+[^,.!?]{0,30}\s+)?(?:to\s+)?(?P<action>[^.!?]+)",
    re.IGNORECASE,
)
_STOP = {
    "a", "an", "and", "do", "doing", "did", "of", "on", "or", "rather",
    "the", "than", "to", "together",
}
_WEAK_ACTION = {"it", "that", "this", "them", "there", "something"}


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[\w'-]+", value)
        if token.casefold() not in _STOP
    }


def _action(value: str) -> str | None:
    match = _ACTION_RE.search(value)
    if not match:
        return None
    action = match.group("action").strip(" ,;:-")
    action = re.sub(r"^(?:we|i|they|he|she)\s+", "", action, flags=re.IGNORECASE)
    return action or None


def _rejected_target(question: str) -> tuple[str, str] | None:
    direct = _CONTRAST_RE.search(question)
    if direct:
        return direct.group("target").strip(), "explicit_contrast"
    if re.search(r"\bwhy\b", question, re.IGNORECASE):
        deferred = _DEFER_RE.search(question)
        if deferred:
            return deferred.group("target").strip(), "causal_displacement"
    return None


def contrast_alternative_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Resolve an alternative from the local dialogue relation that states it.

    This operator is lexical only for the discourse relation and action
    modality; it has no activity, person, benchmark, or topic vocabulary.
    """
    rejected = _rejected_target(frame.raw_question)
    if not rejected:
        return None
    rejected_target, relation_kind = rejected
    target_terms = _tokens(rejected_target)
    if not target_terms:
        return None
    participants = {value.casefold() for value in frame.participant_terms}
    by_session: dict[str, list[Any]] = defaultdict(list)
    for turn in turns:
        by_session[str(getattr(turn, "session_id", ""))].append(turn)

    candidates: list[dict[str, Any]] = []
    for session_id, session_turns in by_session.items():
        ordered = sorted(
            session_turns,
            key=lambda turn: int(getattr(turn, "turn_index", 0)),
        )
        for anchor_position, anchor in enumerate(ordered):
            anchor_text = str(getattr(anchor, "text", ""))
            anchor_terms = _tokens(anchor_text)
            coverage = len(target_terms & anchor_terms) / len(target_terms)
            if coverage < 0.75:
                continue
            anchor_speaker = str(
                getattr(anchor, "speaker_key", "")
                or getattr(anchor, "speaker", "")
            ).casefold()
            question_clauses = re.findall(r"[^.!?]*\?", anchor_text)
            target_in_question = any(
                len(target_terms & _tokens(clause)) / len(target_terms) >= 0.75
                for clause in question_clauses
            )
            if relation_kind == "causal_displacement" and (
                not target_in_question
                or (participants and anchor_speaker in participants)
            ):
                continue
            # The alternative can be stated in the same utterance or in the
            # response side of the next adjacency pair.
            anchor_index = int(getattr(anchor, "turn_index", 0))
            for answer_turn in ordered[anchor_position:]:
                answer_index = int(getattr(answer_turn, "turn_index", 0))
                offset = answer_index - anchor_index
                if offset < 0:
                    continue
                if offset > 2:
                    break
                if relation_kind == "causal_displacement" and offset != 1:
                    continue
                speaker_key = str(
                    getattr(answer_turn, "speaker_key", "")
                    or getattr(answer_turn, "speaker", "")
                ).casefold()
                if participants and speaker_key not in participants:
                    continue
                action = _action(str(getattr(answer_turn, "text", "")))
                if not action:
                    continue
                action_terms = _tokens(action)
                if (
                    relation_kind == "causal_displacement"
                    and not (action_terms - _WEAK_ACTION)
                ):
                    continue
                # Repeating the rejected activity is not an alternative.
                if action_terms and action_terms <= target_terms:
                    continue
                candidates.append(
                    {
                        "value": action,
                        "session_id": session_id,
                        "anchor_turn_id": str(getattr(anchor, "node_id", "")),
                        "answer_turn_id": str(
                            getattr(answer_turn, "node_id", "")
                        ),
                        "target_coverage": round(coverage, 6),
                        "adjacency_offset": offset,
                    }
                )
                break
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            row["target_coverage"],
            -row["adjacency_offset"],
            row["answer_turn_id"],
        ),
        reverse=True,
    )
    distinct = {
        re.sub(r"\W+", " ", row["value"].casefold()).strip()
        for row in candidates
    }
    top = candidates[0]
    source_ids = list(
        dict.fromkeys([top["anchor_turn_id"], top["answer_turn_id"]])
    )
    return {
        "operation": "contrast_alternative",
        "relation_kind": relation_kind,
        "rejected_target": rejected_target,
        "value": top["value"],
        "source_turn_ids": source_ids,
        "candidates": candidates[:6],
        "complete": len(distinct) == 1,
        "completion_basis": "bounded_dialogue_adjacency",
    }
