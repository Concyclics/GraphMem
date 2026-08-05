from __future__ import annotations

import re
from typing import Any, Callable

from .action_semantics import action_families
from .catalog_schema import EventFrameV3, OperandRecordV3
from .schema import QueryFrame


_MEASURE_WORDS = {
    "amount", "different", "item", "items", "kind", "kinds", "number",
    "piece", "pieces", "sort", "sorts", "type", "types",
}
_FUNCTION_WORDS = {
    "a", "an", "and", "few", "for", "from", "in", "of", "on",
    "or", "out", "past", "the", "this", "to", "with",
}
_IDENTITY_STOP = {
    "a", "an", "and", "for", "from", "in", "into", "last", "my", "new",
    "of", "on", "one", "the", "their", "this", "to", "with",
}
_NONFINAL_MODALITIES = {"planned", "possible", "hypothetical"}


def _stem(value: str) -> str:
    token = value.casefold().strip("'")
    irregular = {
        "bought": "buy", "built": "build", "did": "do", "fixed": "fix",
        "finished": "finish", "got": "get", "made": "make",
        "ordered": "order", "picked": "pick", "sold": "sell",
        "worked": "work",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[A-Za-z0-9']+", value)
        if len(token) > 1
    }


def _query_parts(question: str) -> tuple[set[str], str, str] | None:
    match = re.search(
        r"\bhow many\s+(.+?)\s+"
        r"(?:am|are|did|do|does|had|has|have|is|was|were|will|would)"
        r"\s+i\s+(.+?)[?]?$",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    target_phrase = match.group(1).strip()
    target_terms = _tokens(target_phrase) - _MEASURE_WORDS - _FUNCTION_WORDS
    if not target_terms:
        return None
    action_phrase = re.split(
        r"\b(?:in|during|over|within)\s+(?:the\s+)?(?:past|last|previous)\b",
        match.group(2), maxsplit=1, flags=re.IGNORECASE,
    )[0].strip()
    return target_terms, target_phrase, action_phrase


def _target_head_and_modifiers(target_phrase: str) -> tuple[str, set[str]]:
    ordered = [
        _stem(token)
        for token in re.findall(r"[A-Za-z0-9']+", target_phrase)
        if len(token) > 1
    ]
    ordered = [
        token for token in ordered
        if token not in _MEASURE_WORDS and token not in _FUNCTION_WORDS
    ]
    if not ordered:
        return "", set()
    return ordered[-1], set(ordered[:-1])


def _direct_target_terms(value: str) -> set[str]:
    """Terms in the object itself, excluding purpose/relation complements."""

    direct = re.split(
        r"\b(?:in order to|so that|to protect|for protecting|about|regarding)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _tokens(direct)


def _identity_terms(value: str) -> set[str]:
    result = {
        token for token in _tokens(value)
        if token not in _IDENTITY_STOP and token not in _MEASURE_WORDS
        and not token.isdigit() and token not in {"scale"}
    }
    return result


def _same_entity(left: set[str], right: set[str]) -> bool:
    shared = left & right
    if not shared:
        return False
    if len(shared) >= 2:
        return True
    return len(shared) == min(len(left), len(right)) and min(len(left), len(right)) <= 2


def distinct_action_collection_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    event_frames: list[EventFrameV3],
    turns: list[Any],
    *,
    target_semantic_similarity: Callable[[Any], float],
) -> dict[str, Any] | None:
    """Close an open collection by target type, action union, and identity.

    The implementation is query-algebraic: no domain names are known here.
    Coarse event frames contribute type evidence, operands contribute the
    asserted action/object, and source turns retain pronoun-resolution context.
    """

    if frame.requested_operation != "count":
        return None
    # Rate/frequency questions have a measure phrase (days per week), not a
    # collection whose members are days.  Leave them to recurrence closure.
    if re.search(
        r"\b(?:days?|times?)\s+(?:a|per|each)\s+"
        r"(?:day|week|month|year)\b|\bhow often\b",
        frame.raw_question,
        re.IGNORECASE,
    ):
        return None
    parsed = _query_parts(frame.raw_question)
    if parsed is None:
        return None
    target_terms, target_phrase, action_phrase = parsed
    target_head, target_modifiers = _target_head_and_modifiers(target_phrase)
    requested_families = action_families(action_phrase)
    requested_actions = _tokens(action_phrase) - _FUNCTION_WORDS - {
        "have", "has",
    }
    if not requested_families and not requested_actions:
        return None

    frame_by_id = {item.frame_id: item for item in event_frames}
    turn_by_id = {
        str(getattr(turn, "node_id", "")): turn for turn in turns
    }
    action_rows: list[tuple[float, OperandRecordV3, str, float, float]] = []
    for item in operands:
        if (
            item.polarity == "negative"
            or item.modality in _NONFINAL_MODALITIES
            or item.state_op in {"cancel", "retract"}
        ):
            continue
        if item.subject_key.casefold() not in {
            "participant 1", "participant_1", "speaker a", "user",
        }:
            continue
        predicate_lower = item.predicate_key.casefold()
        if re.search(r"\b(?:described|said|stated|noted|reported)\b", predicate_lower):
            continue
        if (
            re.search(r"\b(?:plan|planning|think|thinking|want|wants|expect|expects|trying)\b", predicate_lower)
            and not re.search(r"\b(?:already|assembled|bought|built|completed|finished|fixed|got|purchased|repaired|sold|started|worked)\b", predicate_lower)
        ):
            continue
        if re.search(r"\bwork(?:s|ed|ing)?\s+well\b", predicate_lower):
            continue
        predicate_families = action_families(item.predicate_key)
        predicate_actions = _tokens(item.predicate_key)
        family_match = bool(requested_families & predicate_families)
        lexical_match = bool(requested_actions & predicate_actions)
        if not family_match and not lexical_match:
            continue

        frame_node = frame_by_id.get(item.event_frame_id or "")
        operand_similarity = max(0.0, target_semantic_similarity(item))
        frame_similarity = (
            max(0.0, target_semantic_similarity(frame_node))
            if frame_node is not None else 0.0
        )
        type_similarity = max(operand_similarity, frame_similarity)
        direct_terms = _direct_target_terms(item.object_text)
        direct_type_match = target_terms.issubset(direct_terms)
        if (
            not bool(target_terms & direct_terms)
            and bool(target_terms & _tokens(item.predicate_key))
            and re.search(r"\b(?:at|during|for|from|in|on|to|with)\s*$", predicate_lower)
        ):
            continue
        source_text = " ".join(
            str(getattr(turn_by_id[source_id], "text", ""))
            for source_id in item.source_turn_ids
            if source_id in turn_by_id
        )
        source_terms = _tokens(source_text)
        source_type_match = target_terms.issubset(source_terms)
        near_sibling_entity = bool(
            target_head
            and target_modifiers
            and target_head not in direct_terms
            and target_head not in source_terms
            and bool(target_modifiers & (direct_terms | source_terms))
        )
        if near_sibling_entity:
            continue
        pronoun_object = bool(re.match(
            r"^\s*(?:it|one|that|this)\b", item.object_text, re.IGNORECASE
        ))
        if not (
            direct_type_match
            or operand_similarity >= 0.65
            or (pronoun_object and frame_similarity >= 0.65)
            or (source_type_match and frame_similarity >= 0.65)
            or (frame_similarity >= 0.65 and operand_similarity >= 0.55)
        ):
            continue

        identity_text = item.object_text
        action_rows.append((
            4.0 * type_similarity
            + 1.0 * float(direct_type_match)
            + 0.35 * float(source_type_match)
            + 0.25 * float(family_match),
            item,
            identity_text,
            type_similarity,
            frame_similarity,
            direct_type_match,
        ))
    if not action_rows:
        return None

    direct_sources = {
        source
        for _score, item, _identity, _type, _frame, direct in action_rows
        if direct
        for source in item.source_turn_ids
    }
    action_rows = [
        row for row in action_rows
        if not (
            not row[5]
            and bool(set(row[1].source_turn_ids) & direct_sources)
            and bool(target_terms & _tokens(row[1].predicate_key))
        )
    ]
    if not action_rows:
        return None

    action_rows.sort(
        key=lambda row: (row[0], row[1].confidence, row[1].operand_id),
        reverse=True,
    )
    clusters: list[dict[str, Any]] = []
    for score, item, identity_text, type_similarity, frame_similarity, _direct in action_rows:
        identity = _identity_terms(identity_text)
        source_ids = set(item.source_turn_ids)
        match = next((
            cluster for cluster in clusters
            if _same_entity(identity, cluster["identity_terms"])
            or (
                source_ids & cluster["source_turn_id_set"]
                and _same_entity(
                    _identity_terms(item.object_text),
                    cluster["direct_identity_terms"],
                )
            )
        ), None)
        row = {
            "operand_id": item.operand_id,
            "predicate": item.predicate_key,
            "object": item.object_text,
            "source_turn_ids": list(item.source_turn_ids),
            "type_similarity": round(float(type_similarity), 6),
            "frame_type_similarity": round(float(frame_similarity), 6),
            "score": round(float(score), 6),
        }
        if match is None:
            clusters.append({
                "identity_terms": set(identity),
                "direct_identity_terms": _identity_terms(item.object_text),
                "source_turn_id_set": set(source_ids),
                "rows": [row],
            })
        else:
            match["identity_terms"].update(identity)
            match["direct_identity_terms"].update(_identity_terms(item.object_text))
            match["source_turn_id_set"].update(source_ids)
            match["rows"].append(row)

    items = [
        max(cluster["rows"], key=lambda row: (row["score"], row["operand_id"]))
        for cluster in clusters
    ]

    represented_sources = {
        source
        for cluster in clusters
        for row in cluster["rows"]
        for source in row["source_turn_ids"]
    }
    uncovered_sources: list[str] = []
    for turn in turns:
        transport = str(getattr(turn, "transport_role", "")).casefold()
        speaker = str(getattr(turn, "speaker_key", "")).casefold()
        if transport == "assistant" or (
            transport and transport != "user"
            and speaker not in {"participant 1", "participant_1", "user"}
        ):
            continue
        node_id = str(getattr(turn, "node_id", ""))
        if not node_id or node_id in represented_sources:
            continue
        text = str(getattr(turn, "text", ""))
        turn_families = action_families(text)
        turn_actions = _tokens(text)
        if not (
            bool(requested_families & turn_families)
            or bool(requested_actions & turn_actions)
        ):
            continue
        turn_terms = _tokens(text)
        turn_identity = _identity_terms(text)
        if any(
            _same_entity(turn_identity, cluster["identity_terms"])
            for cluster in clusters
        ):
            continue
        lexical_type = target_terms.issubset(turn_terms)
        partial_typed = bool(target_terms & turn_terms) and (
            max(0.0, target_semantic_similarity(turn)) >= 0.50
        )
        if lexical_type or partial_typed:
            uncovered_sources.append(node_id)
    uncovered_sources = list(dict.fromkeys(uncovered_sources))

    # Recover an entity explicitly introduced by a grammatical enumeration
    # marker when the fine operand projection omitted that embedded object.
    # This stays target/action driven; the surface entity can be from any domain.
    recovered_items: list[dict[str, Any]] = []
    remaining_uncovered: list[str] = []
    for source_id in uncovered_sources:
        turn = turn_by_id.get(source_id)
        text = str(getattr(turn, "text", "")) if turn is not None else ""
        match = re.search(
            r"\b(?:featuring|including)\s+(?:a|an|the)\s+"
            r"(?P<entity>[^,.;!?]{2,120})",
            text,
            re.IGNORECASE,
        )
        if match is None:
            remaining_uncovered.append(source_id)
            continue
        entity = match.group("entity").strip()
        identity = _identity_terms(entity)
        if not identity or any(
            _same_entity(identity, cluster["identity_terms"])
            for cluster in clusters
        ):
            remaining_uncovered.append(source_id)
            continue
        recovered_items.append({
            "operand_id": f"{source_id}:lossless_embedded_entity",
            "predicate": "lossless embedded action entity",
            "object": entity,
            "source_turn_ids": [source_id],
            "type_similarity": round(
                max(0.0, target_semantic_similarity(turn)), 6
            ),
            "frame_type_similarity": 0.0,
            "score": 0.0,
        })
    items.extend(recovered_items)
    uncovered_sources = remaining_uncovered
    complete = not uncovered_sources
    return {
        "operation": "distinct_action_entity_collection",
        "value": len(items),
        "target_phrase": target_phrase,
        "requested_action_families": sorted(requested_families),
        "requested_action_terms": sorted(requested_actions),
        "items": items,
        "operand_ids": [
            row["operand_id"] for cluster in clusters for row in cluster["rows"]
        ],
        "source_turn_ids": list(dict.fromkeys([
            *(source for cluster in clusters for row in cluster["rows"] for source in row["source_turn_ids"]),
            *(source for row in recovered_items for source in row["source_turn_ids"]),
            *uncovered_sources,
        ])),
        "uncovered_source_turn_ids": uncovered_sources,
        "complete": complete,
        "completion_basis": "global_typed_operand_action_identity_closure",
    }
