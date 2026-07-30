from __future__ import annotations

import re
from typing import Any, Callable

from .catalog_schema import EventFrameV3, OperandRecordV3
from .schema import QueryFrame


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_IRREGULAR = {
    "began": "begin",
    "came": "come",
    "found": "find",
    "gone": "go",
    "kept": "keep",
    "left": "leave",
    "met": "meet",
    "ran": "run",
    "sat": "sit",
    "spoke": "speak",
    "took": "take",
    "went": "go",
}
_PREDICATE_FUNCTION = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "in", "is",
    "of", "on", "or", "the", "to", "was", "were", "with",
}
_LOCATION_PREPOSITIONS = {
    "at", "in", "inside", "near", "on", "outside", "under", "within",
}
_CLAUSE_BOUNDARIES = {
    "and", "because", "but", "so", "that", "then", "when", "where", "which", "who",
}
_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
_TEMPORAL_HEADS = _MONTHS | {
    "day", "days", "evening", "evenings", "hour", "hours", "midnight",
    "minute", "minutes", "month", "months", "morning", "mornings", "night",
    "nights", "noon", "spring", "summer", "today", "tomorrow", "week",
    "weekday", "weekdays", "weeks", "weekend", "weekends", "winter", "year",
    "years", "yesterday",
}


def _lemma(token: str) -> str:
    value = token.casefold().strip(chr(39) + chr(34))
    if value in _IRREGULAR:
        return _IRREGULAR[value]
    if len(value) > 5 and value.endswith("ing"):
        stem = value[:-3]
        if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
            stem = stem[:-1]
        return _IRREGULAR.get(stem, stem)
    if len(value) > 4 and value.endswith("ied"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith("ed"):
        stem = value[:-2]
        if stem.endswith("v"):
            stem += "e"
        return _IRREGULAR.get(stem, stem)
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def _lemmas(text: str) -> list[str]:
    return [_lemma(match.group(0)) for match in _WORD_RE.finditer(text)]


def _predicate_terms(predicate: str) -> set[str]:
    return {
        term for term in _lemmas(predicate.replace("_", " "))
        if term not in _PREDICATE_FUNCTION and len(term) > 1
    }


def _is_temporal_phrase(words: list[str]) -> bool:
    if not words:
        return True
    heads = {_lemma(value) for value in words[:3]}
    if heads & _TEMPORAL_HEADS:
        return True
    joined = " ".join(words[:4]).casefold()
    return bool(
        re.search(r"\b(?:last|next|this|every)\b", joined)
        or re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", joined)
        or re.search(r"\b(?:19|20)\d{2}\b", joined)
    )


def _location_after_relation(text: str, predicate_terms: set[str]) -> str | None:
    """Extract a bounded location PP attached to a relation mention."""

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return None
    lemmas = [_lemma(match.group(0)) for match in matches]
    relation_positions = [
        index for index, term in enumerate(lemmas) if term in predicate_terms
    ]
    for relation_index in relation_positions:
        # Keep the attachment local. A place in a later sentence or unrelated
        # clause is a different event even when it shares participants.
        upper = min(len(matches), relation_index + 15)
        for prep_index in range(relation_index + 1, upper):
            prep = lemmas[prep_index]
            if prep in _CLAUSE_BOUNDARIES:
                break
            if prep not in _LOCATION_PREPOSITIONS:
                continue
            phrase_words: list[str] = []
            end_index = prep_index + 1
            while end_index < upper:
                raw = matches[end_index].group(0)
                term = lemmas[end_index]
                if term in _CLAUSE_BOUNDARIES:
                    break
                phrase_words.append(raw)
                end_index += 1
            if _is_temporal_phrase(phrase_words):
                continue
            if not phrase_words:
                continue
            start = matches[prep_index].start()
            end = matches[end_index - 1].end()
            value = re.sub(r"\s+", " ", text[start:end]).strip(" ,.;:-")
            if value:
                return value
    return None


def relation_slot_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    event_frames: list[EventFrameV3],
    turns: list[Any],
    *,
    query_overlap: Callable[[QueryFrame, str], float],
    semantic_similarity: Callable[[OperandRecordV3], float] | None = None,
) -> dict[str, Any] | None:
    """Bind a requested slot to one event and its lossless provenance.

    This is relation- and topic-neutral: rank only typed operands whose
    predicate is explicit in the query, follow their event-frame/source links,
    and extract the value from that local source. Attributes from merely
    adjacent events are never eligible.
    """

    if frame.requested_operation != "location":
        return None
    query_terms = _lemmas(frame.raw_question)
    query_term_set = set(query_terms)
    turn_by_id = {
        str(getattr(turn, "node_id", "")): turn
        for turn in turns
        if getattr(turn, "node_id", None)
    }
    frame_by_id = {item.frame_id: item for item in event_frames}
    ranked: list[tuple[tuple[Any, ...], OperandRecordV3, set[str]]] = []
    for item in operands:
        if item.polarity == "negative" or item.modality in {
            "planned", "possible", "conditional", "hypothetical",
        }:
            continue
        predicate_terms = _predicate_terms(item.predicate_key)
        positions = [
            index for index, term in enumerate(query_terms) if term in predicate_terms
        ]
        if not positions:
            continue
        object_overlap = len(query_term_set & set(_lemmas(item.object_text)))
        subject_overlap = len(query_term_set & set(_lemmas(item.subject_key)))
        linked_frame = frame_by_id.get(item.event_frame_id or "")
        frame_text = linked_frame.retrieval_text if linked_frame is not None else ""
        source_present = sum(
            source_id in turn_by_id for source_id in item.source_turn_ids
        )
        dense = semantic_similarity(item) if semantic_similarity is not None else 0.0
        score = (
            -min(positions),
            object_overlap,
            subject_overlap,
            query_overlap(frame, item.retrieval_text),
            query_overlap(frame, frame_text),
            max(0.0, dense),
            source_present,
            item.confidence,
            item.operand_id,
        )
        ranked.append((score, item, predicate_terms))
    if not ranked:
        return None

    for _score, item, predicate_terms in sorted(ranked, reverse=True):
        linked_frame = frame_by_id.get(item.event_frame_id or "")
        source_ids = list(dict.fromkeys([
            *item.source_turn_ids,
            *(linked_frame.source_turn_ids if linked_frame is not None else []),
        ]))
        candidates: list[tuple[str, str]] = []
        for source_id in source_ids:
            turn = turn_by_id.get(source_id)
            if turn is None:
                continue
            value = _location_after_relation(
                str(getattr(turn, "text", "")), predicate_terms
            )
            if value is not None:
                candidates.append((source_id, value))
        if not candidates:
            continue
        normalized = {
            re.sub(r"\W+", " ", value.casefold()).strip(): value
            for _source_id, value in candidates
        }
        keys = list(normalized)
        compatible = all(
            left in right or right in left
            for index, left in enumerate(keys)
            for right in keys[index + 1:]
        )
        if not compatible:
            continue
        value = max(normalized.values(), key=lambda row: (len(row.split()), len(row)))
        value_key = re.sub(r"\W+", " ", value.casefold()).strip()
        supporting_ids = [
            source_id for source_id, candidate in candidates
            if (
                re.sub(r"\W+", " ", candidate.casefold()).strip() in value_key
                or value_key in re.sub(r"\W+", " ", candidate.casefold()).strip()
            )
        ]
        return {
            "operation": "relation_slot_location",
            "value": value,
            "predicate": item.predicate_key,
            "operand_ids": [item.operand_id],
            "event_frame_ids": [item.event_frame_id] if item.event_frame_id else [],
            "source_turn_ids": list(dict.fromkeys(supporting_ids)),
            "evidence": [
                str(getattr(turn_by_id[source_id], "text", ""))
                for source_id in supporting_ids if source_id in turn_by_id
            ],
            "complete": bool(supporting_ids),
            "completion_basis": "predicate_operand_event_frame_source_slot",
        }
    return None
