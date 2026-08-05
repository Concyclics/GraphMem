from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from .schema import QueryFrame
from .temporal_normalize import parse_datetime


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_GLUE = {
    "a", "an", "did", "do", "does", "have", "has", "how", "i", "many",
    "my", "of", "on", "or", "the",
}


def _stem(value: str) -> str:
    token = value.casefold().strip("'")
    irregular = {
        "attended": "attend", "bought": "buy", "completed": "complete",
        "did": "do", "gone": "go", "had": "have", "made": "make",
        "worked": "work",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("oes"):
        return token[:-2]
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[A-Za-z0-9']+", value)
        if _stem(token) not in _GLUE
    }


def _query_slots(question: str) -> tuple[set[str], set[str], str] | None:
    match = re.search(
        r"\bhow many\s+(.+?)\s+(?:did|do|does|have|has)\s+i\s+(.+?)[?]?$",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    target = _tokens(match.group(1))
    relation = _tokens(match.group(2))
    target_phrase = re.split(
        r"\s+of\s+", match.group(1), maxsplit=1, flags=re.IGNORECASE
    )[0]
    head_matches = re.findall(r"[A-Za-z][A-Za-z'-]*", target_phrase)
    if not target or not relation or not head_matches:
        return None
    return target, relation, _stem(head_matches[-1])


def latest_cardinality_from_turns(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Read a latest cumulative cardinality state from lossless user turns."""

    if frame.requested_operation != "count":
        return None
    slots = _query_slots(frame.raw_question)
    if slots is None:
        return None
    target_terms, relation_terms, head = slots
    number_pattern = "|".join(_NUMBER_WORDS)
    candidates: list[tuple[datetime, int, str, str]] = []
    for turn in turns:
        transport = str(getattr(turn, "transport_role", "")).casefold()
        speaker = str(getattr(turn, "speaker_key", "")).casefold()
        if transport == "assistant" or (
            transport and transport != "user"
            and speaker not in {"participant 1", "user"}
        ):
            continue
        observed = parse_datetime(str(getattr(turn, "session_date", "")))
        if observed is None:
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+|\n+", str(getattr(turn, "text", ""))
            )
            if sentence.strip()
        ]
        for index, sentence in enumerate(sentences):
            terms = _tokens(sentence)
            local_context = " ".join(
                sentences[max(0, index - 1): index + 1]
            )
            if len(target_terms & _tokens(local_context)) < min(
                2, len(target_terms)
            ):
                continue
            if not (relation_terms & terms):
                continue
            match = re.search(
                rf"\b(?P<number>\d+|{number_pattern})\s+"
                rf"(?:\w+\s+){{0,2}}{re.escape(head)}s?\b",
                sentence,
                re.IGNORECASE,
            )
            if not match:
                continue
            raw = match.group("number").casefold()
            value = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
            candidates.append((
                observed, value, str(getattr(turn, "node_id", "")),
                local_context,
            ))
    if not candidates:
        return None

    # A coordinated complement denotes independent partitions of the requested
    # cardinality (for example, X and Y), not competing values of one state.
    # Close every explicitly named partition before summing; otherwise retain
    # the ordinary latest-state behavior below.
    partition_match = re.search(
        r"\bfor\s+(.+?)(?:[?]|$)", frame.raw_question, re.IGNORECASE
    )
    if partition_match is not None:
        raw_parts = [
            part.strip()
            for part in re.split(r"\s*(?:,|\band\b|\bor\b)\s*", partition_match.group(1), flags=re.IGNORECASE)
            if part.strip()
        ]
        partition_terms = []
        for part in raw_parts:
            terms = [
                token for token in _tokens(part)
                if token not in relation_terms and token != head
            ]
            if not terms:
                terms = list(_tokens(part) - {head})
            if terms:
                partition_terms.append(sorted(terms, key=lambda value: (len(value), value))[-1])
        partition_terms = list(dict.fromkeys(partition_terms))
        if len(partition_terms) >= 2:
            selected = []
            for partition in partition_terms:
                rows = [row for row in candidates if partition in _tokens(row[3])]
                if not rows:
                    selected = []
                    break
                selected.append(max(rows))
            if selected:
                return {
                    "operation": "latest_cardinality_state",
                    "value": sum(row[1] for row in selected),
                    "unit": head,
                    "observed_at": max(row[0] for row in selected).date().isoformat(),
                    "parts": [
                        {
                            "partition": partition,
                            "value": row[1],
                            "source_turn_id": row[2],
                        }
                        for partition, row in zip(partition_terms, selected)
                    ],
                    "source_turn_ids": list(dict.fromkeys(row[2] for row in selected)),
                    "operand_ids": [],
                    "complete": True,
                    "completion_basis": "explicit_partition_latest_cardinality_sum",
                }
    observed, value, source_turn_id, evidence = max(candidates)
    return {
        "operation": "latest_cardinality_state",
        "value": value,
        "unit": head,
        "observed_at": observed.date().isoformat(),
        "source_turn_ids": [source_turn_id],
        "operand_ids": [],
        "evidence": evidence,
        "complete": True,
        "completion_basis": "lossless_latest_subject_relation_cardinality",
    }
