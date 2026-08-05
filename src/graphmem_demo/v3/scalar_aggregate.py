from __future__ import annotations

import re
from typing import Any

from .schema import QueryFrame


_PARTITION_EQUIVALENCE = {
    "graduate": {"graduate", "master", "masters", "postgraduate"},
    "undergraduate": {"bachelor", "bachelors", "undergrad", "undergraduate"},
}
_PARTITION_GENERIC = {"degree", "program", "studie", "study"}


def _tokens(value: str) -> set[str]:
    result: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]*", value):
        token = raw.casefold()
        if token.endswith("'s"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        result.add(token)
    return result


def _partition_terms(value: str) -> set[str]:
    tokens = _tokens(value)
    expanded = set(tokens)
    for canonical, aliases in _PARTITION_EQUIVALENCE.items():
        if tokens & aliases:
            expanded.add(canonical)
            expanded.update(aliases)
    return expanded - _PARTITION_GENERIC


def named_scalar_average_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Average one scalar independently bound to each named partition."""

    question = frame.raw_question
    match = re.search(
        r"\b(?:average|mean)\s+([A-Za-z][A-Za-z0-9 _-]{0,40}?)\s+of\s+"
        r"(?:my\s+|the\s+)?(.+?)[?]?$",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    attribute = match.group(1).strip()
    partition_phrase = match.group(2).strip()
    raw_partitions = [
        value.strip()
        for value in re.split(r"\s+(?:and|or)\s+|,", partition_phrase)
        if value.strip()
    ]
    if len(raw_partitions) < 2 or len(raw_partitions) > 8:
        return None
    attribute_terms = _tokens(attribute)
    if not attribute_terms:
        return None

    candidates: dict[str, list[tuple[float, float, str, str]]] = {
        partition: [] for partition in raw_partitions
    }
    user_turns = [
        turn for turn in turns
        if str(getattr(turn, "transport_role", "")).casefold() != "assistant"
    ]
    anchor_stop = {
        "a", "an", "and", "at", "degree", "for", "from", "i",
        "in", "my", "of", "program", "studie", "study", "the", "with",
    }
    partition_anchors: dict[tuple[str, str], list[set[str]]] = {}
    for partition in raw_partitions:
        terms = _partition_terms(partition)
        for turn in user_turns:
            session_id = str(getattr(turn, "session_id", ""))
            for sentence in re.split(
                r"(?<=[.!?])\s+|\n+", str(getattr(turn, "text", ""))
            ):
                if terms & _partition_terms(sentence):
                    partition_anchors.setdefault((session_id, partition), []).append(
                        _tokens(sentence) - anchor_stop
                    )
    for turn in user_turns:
        if str(getattr(turn, "transport_role", "")).casefold() == "assistant":
            continue
        text = str(getattr(turn, "text", ""))
        if not attribute_terms.issubset(_tokens(text)):
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip()
        ]
        for index, sentence in enumerate(sentences):
            if not attribute_terms.issubset(_tokens(sentence)):
                continue
            scalar = re.search(
                rf"\b{re.escape(attribute)}\b"
                r"(?:\s+(?:of|was|is|:))?\s*"
                r"(?P<value>\d+(?:\.\d+)?)"
                r"(?:\s*(?:out of|/)\s*\d+(?:\.\d+)?)?",
                sentence,
                re.IGNORECASE,
            )
            if scalar is None:
                continue
            value = float(scalar.group("value"))
            context = " ".join(sentences[max(0, index - 1): index + 1])
            context_terms = _partition_terms(context)
            context_anchor_terms = _tokens(context) - anchor_stop
            session_id = str(getattr(turn, "session_id", ""))
            for partition in raw_partitions:
                terms = _partition_terms(partition)
                local_overlap = len(terms & context_terms)
                anchor_overlap = max((
                    len(context_anchor_terms & anchor)
                    for anchor in partition_anchors.get((session_id, partition), [])
                ), default=0)
                if local_overlap <= 0 and anchor_overlap < 2:
                    continue
                binding_score = 100.0 * local_overlap + float(anchor_overlap)
                candidates[partition].append((
                    binding_score, value, str(getattr(turn, "node_id", "")), context,
                ))
    if any(not values for values in candidates.values()):
        return None

    proofs = []
    source_turn_ids = []
    for partition in raw_partitions:
        values = candidates[partition]
        best_score = max(row[0] for row in values)
        best = [row for row in values if row[0] == best_score]
        unique = {round(row[1], 8) for row in best}
        if len(unique) != 1:
            return None
        _score, value, source_turn_id, evidence = best[-1]
        proofs.append({
            "partition": partition,
            "value": value,
            "source_turn_id": source_turn_id,
            "evidence": evidence,
        })
        source_turn_ids.append(source_turn_id)
    value = sum(row["value"] for row in proofs) / len(proofs)
    precision = max(
        len(str(row["value"]).partition(".")[2].rstrip("0"))
        for row in proofs
    )
    return {
        "operation": "named_scalar_average",
        "attribute": attribute,
        "value": round(value, max(2, precision)),
        "proofs": proofs,
        "operand_ids": [],
        "source_turn_ids": list(dict.fromkeys(source_turn_ids)),
        "complete": True,
        "completion_basis": "one_unambiguous_scalar_per_named_partition",
    }
