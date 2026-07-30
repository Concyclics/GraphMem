from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Callable

from .catalog_schema import OperandRecordV3
from .schema import QueryFrame


_DISCOURSE = re.compile(
    r"\b(?:ask|discuss|explain|provide|recommend|suggest|tip)\w*\b",
    re.IGNORECASE,
)
_ACCEPTED = re.compile(
    r"\b(?:adopt|choose|chose|decide|final|like|love|name|prefer|select|settle|use)\w*\b",
    re.IGNORECASE,
)
_ONLY_PROPOSED = re.compile(
    r"\b(?:brainstorm|propose|suggest)\w*\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_SCALAR_VALUE = re.compile(
    r"(?<!\w)(?:[\$€£]\s*)?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*[:/]\s*\d+(?:\.\d+)?)?"
    r"(?:\s*(?:%|percent|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"degrees?|kg|g|lb|lbs|miles?|km|psi|mph|kph))?(?!\w)",
    re.IGNORECASE,
)
_SCALAR_QUERY = re.compile(
    r"\b(?:age|amount|balance|count|distance|duration|height|length|number|"
    r"percentage|percent|pressure|price|rate|record|score|speed|temperature|"
    r"time|total|value|weight)\b|"
    r"\bhow\s+(?:many|much|long|old|fast|far|heavy|high|tall)\b",
    re.IGNORECASE,
)


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.search(
        r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", value
    )
    if not match:
        return None
    try:
        return datetime(*map(int, match.groups()))
    except ValueError:
        return None


def _scalar_answer_requested(question: str) -> bool:
    lowered = question.casefold()
    if re.search(r"\bhow\s+(?:many|much|long|old|fast|far|heavy|high|tall)\b", lowered):
        return True
    what_match = re.search(r"\bwhat\b", lowered)
    if what_match is None:
        return bool(_SCALAR_QUERY.search(lowered))
    lowered = lowered[what_match.start():]
    # In a what-question, a scalar term in a relative/purpose clause is
    # evidence context, not necessarily the requested answer type.
    answer_head = re.split(
        r"\b(?:because|that|which|who|where|when|so that)\b", lowered, maxsplit=1
    )[0]
    return bool(_SCALAR_QUERY.search(answer_head))



def _embedded_scalar_is_attribute_reference(
    item: OperandRecordV3, value: str,
) -> bool:
    """Recognize a scalar grammatically assigned to an embedded attribute."""
    text = item.object_text.casefold()
    offset = text.rfind(value.casefold())
    if offset <= 0:
        return False
    prefix = text[:offset].rstrip()
    return re.search(
        r"\b(?:is|was|were|equals?|of|at)\s*(?:about|approximately)?$",
        prefix,
    ) is not None
def scalar_attribute_state_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    *,
    semantic_similarity: Callable[[OperandRecordV3], float],
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, object] | None:
    """Resolve a dated scalar-valued attribute without domain-specific labels."""
    if (
        frame.requested_operation not in {"lookup", "latest", "state"}
        or frame.explicit_dates
        or re.search(r"\btotal\b", frame.raw_question, re.IGNORECASE)
        or re.search(
            r"\b(?:what time|when)\s+(?:did|do|does|will)\b",
            frame.raw_question,
            re.IGNORECASE,
        )
        or not _scalar_answer_requested(frame.raw_question)
    ):
        return None
    candidates: list[tuple[float, float, datetime, int, str, OperandRecordV3]] = []
    embedded_candidates: list[
        tuple[float, float, datetime, int, str, OperandRecordV3]
    ] = []
    for item in operands:
        if (
            item.polarity == "negative"
            or item.modality in {"planned", "possible", "hypothetical"}
            or _DISCOURSE.search(item.predicate_key)
        ):
            continue
        values = [
            match.group(0).strip()
            for match in _SCALAR_VALUE.finditer(item.object_text)
        ]
        if not values:
            continue
        relation_text = " ".join((item.predicate_key, item.context_key))
        overlap = query_overlap(frame, relation_text)
        expanded_overlap = query_overlap(
            frame, " ".join((relation_text, item.object_text))
        )
        semantic = max(0.0, semantic_similarity(item))
        sequence_match = re.search(r"(\d+)\Z", item.operand_id)
        sequence = int(sequence_match.group(1)) if sequence_match else -1
        observed = _date(item.event_time or item.observed_at) or datetime.min
        if expanded_overlap > 0 and semantic > 0:
            embedded_candidates.append((
                expanded_overlap, semantic, observed, sequence, values[-1], item,
            ))
        if overlap <= 0 or semantic <= 0:
            continue
        candidates.append((
            overlap, semantic, observed, sequence, values[-1], item,
        ))
    if not candidates:
        return None

    # Select the semantic attribute family before applying temporal update order.
    best_overlap = max(row[0] for row in candidates)
    family = [row for row in candidates if row[0] >= best_overlap * 0.75]
    best_semantic = max(row[1] for row in family)
    family = [row for row in family if row[1] >= best_semantic * 0.82]
    overlap, semantic, observed, sequence, value, item = max(
        family,
        key=lambda row: (row[2], row[3], row[0], row[1], row[5].operand_id),
    )
    unresolved_newer = [
        row for row in embedded_candidates
        if (row[2], row[3]) > (observed, sequence)
        and row[4] != value
        and row[0] >= overlap * 0.75
        and row[1] >= semantic * 0.70
    ]
    resolved_references = [
        row for row in unresolved_newer
        if _embedded_scalar_is_attribute_reference(row[5], row[4])
    ]
    if resolved_references:
        resolved = max(
            resolved_references,
            key=lambda row: (row[2], row[3], row[0], row[1], row[5].operand_id),
        )
        return {
            "operation": "scalar_attribute_state",
            "value": resolved[4],
            "predicate": resolved[5].predicate_key,
            "time": (
                resolved[2].date().isoformat()
                if resolved[2] != datetime.min else None
            ),
            "candidate_values": list(dict.fromkeys([
                value, *[row[4] for row in resolved_references],
            ])),
            "operand_ids": [resolved[5].operand_id],
            "source_turn_ids": list(resolved[5].source_turn_ids),
            "complete": True,
            "semantic_score": resolved[1],
            "query_overlap": resolved[0],
            "completion_basis": "embedded_attribute_scalar_reference",
        }
    if unresolved_newer:
        competing = sorted(
            [
                (overlap, semantic, observed, sequence, value, item),
                *unresolved_newer,
            ],
            key=lambda row: (row[2], row[3], row[5].operand_id),
        )
        return {
            "operation": "scalar_attribute_state_ambiguous",
            "value": None,
            "candidate_values": list(dict.fromkeys(row[4] for row in competing)),
            "operand_ids": [row[5].operand_id for row in competing],
            "source_turn_ids": list(dict.fromkeys(
                source_id
                for row in competing
                for source_id in row[5].source_turn_ids
            )),
            "complete": False,
            "completion_basis": "newer_scalar_reference_requires_lossless_resolution",
        }
    return {
        "operation": "scalar_attribute_state",
        "value": value,
        "predicate": item.predicate_key,
        "time": observed.date().isoformat() if observed != datetime.min else None,
        "candidate_values": list(dict.fromkeys(row[4] for row in family)),
        "operand_ids": [item.operand_id],
        "source_turn_ids": list(item.source_turn_ids),
        "complete": True,
        "semantic_score": semantic,
        "query_overlap": overlap,
    }


def frequency_state_comparison_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    *,
    query_overlap: Callable[[QueryFrame, str], float],
    turns: list[object] | None = None,
) -> dict[str, object] | None:
    """Compare rates after closing each typed state over its lossless source."""
    question = frame.raw_question.casefold()
    direction_match = re.search(
        r"\b(more|less)\s+(?:frequently|often|regularly)\b|"
        r"\b(increased|decreased)\s+(?:frequency|rate)\b",
        question,
    )
    if direction_match is None:
        return None
    number_words = {**_NUMBER_WORDS, "once": 1, "twice": 2}
    source_turns = {
        str(getattr(turn, "node_id", "")): turn for turn in (turns or [])
    }
    self_reference = bool(re.search(r"\b(?:i|me|my|mine|myself)\b", question))
    candidates = []
    for item in operands:
        if item.polarity == "negative" or item.modality in {"planned", "possible", "hypothetical"}:
            continue
        text = " ".join((item.predicate_key, item.object_text, item.context_key))
        all_provenance_turns = [
            source_turns[source_id]
            for source_id in item.source_turn_ids
            if source_id in source_turns
        ]
        provenance_turns = all_provenance_turns
        if self_reference:
            provenance_turns = [
                turn for turn in provenance_turns
                if str(getattr(turn, "transport_role", "")).casefold() == "user"
                or str(getattr(turn, "speaker_key", "")).casefold()
                in {"participant 1", "participant_1", "user"}
            ]
            if all_provenance_turns and not provenance_turns:
                continue
        source_text = " ".join(
            str(getattr(turn, "text", "")) for turn in provenance_turns
        )
        expanded_text = " ".join((text, source_text))
        rate = None
        match = re.search(
            r"\b(\d+|once|twice|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?:time|times|day|days)\s+(?:a|per|each)\s+week\b",
            expanded_text.casefold(),
        )
        if match:
            rate = int(match.group(1)) if match.group(1).isdigit() else number_words[match.group(1)]
        else:
            weekdays = set(re.findall(
                r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                expanded_text.casefold(),
            ))
            if len(weekdays) >= 2:
                rate = len(weekdays)
            elif item.recurrence_days:
                rate = len(set(item.recurrence_days))
        overlap = query_overlap(frame, expanded_text)
        if rate is None or overlap <= 0:
            continue
        candidates.append((
            _date(item.event_time or item.observed_at) or datetime.min,
            rate,
            overlap,
            item,
        ))
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda row: (row[0], row[2], row[3].operand_id))
    older = candidates[-2]
    newer = candidates[-1]
    asks_more = bool(re.search(r"\b(?:more|increased)\b", question))
    value = newer[1] > older[1] if asks_more else newer[1] < older[1]
    return {
        "operation": "frequency_state_comparison",
        "value": "yes" if value else "no",
        "previous_rate_per_week": older[1],
        "current_rate_per_week": newer[1],
        "operand_ids": [older[3].operand_id, newer[3].operand_id],
        "source_turn_ids": list(dict.fromkeys([
            *older[3].source_turn_ids, *newer[3].source_turn_ids,
        ])),
        "proofs": [
            {
                "rate_per_week": older[1],
                "operand_id": older[3].operand_id,
                "source_turn_ids": older[3].source_turn_ids,
            },
            {
                "rate_per_week": newer[1],
                "operand_id": newer[3].operand_id,
                "source_turn_ids": newer[3].source_turn_ids,
            },
        ],
        "complete": True,
        "completion_basis": "typed_state_plus_lossless_source_comparison",
    }


def final_choice_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    *,
    semantic_similarity: Callable[[OperandRecordV3], float],
    query_overlap: Callable[[QueryFrame, str], float] | None = None,
) -> dict[str, object] | None:
    """Return an adopted value after binding the requested relation family."""
    if frame.requested_operation not in {"lookup", "latest"} or not re.search(
        r"\b(?:chose|chosen|decid\w*|finally|name|named|settled)\b",
        frame.raw_question.casefold(),
    ):
        return None
    candidates = []
    name_query = bool(re.search(r"\b(?:name|named)\b", frame.raw_question.casefold()))
    ignored_name_anchors = {
        "a", "an", "and", "by", "choose", "chosen", "chose", "decide",
        "decided", "did", "do", "does", "finally", "for",
        "from", "had", "has", "have", "her", "his", "is", "my", "name",
        "named", "of", "or", "our", "settled", "that", "the", "their",
        "to", "was", "what", "which", "who", "with", "it", "its", "this", "these", "those", "them",
    }
    participant_anchors = set(frame.participant_terms)
    name_relation_anchors = {
        token for token in re.findall(r"[a-z0-9]+", frame.raw_question.casefold())
        if len(token) >= 3
        and token not in ignored_name_anchors
        and token not in participant_anchors
    }
    for item in operands:
        predicate = item.predicate_key.casefold()
        relation_text = " ".join((item.predicate_key, item.object_text, item.context_key))
        relation_tokens = set(re.findall(r"[a-z0-9]+", relation_text.casefold()))
        relation_overlap = (
            query_overlap(
                frame,
                relation_text,
            )
            if query_overlap is not None
            else 1.0
        )
        if relation_overlap <= 0:
            continue
        if name_query and name_relation_anchors and not (
            name_relation_anchors & relation_tokens
        ):
            continue
        if name_query and not re.search(r"\b(?:call|name)\w*\b", predicate):
            # The selected name may be encoded in the object while the
            # predicate carries possession or adoption.
            object_names_value = bool(re.search(
                r"\bnam(?:e|ed|es)\b", item.object_text, re.IGNORECASE
            ))
            preference_names_value = bool(
                re.search(r"\b(?:like|love|prefer)\w*\b", predicate)
                and re.search(r"\bnames?\b", item.object_text, re.IGNORECASE)
            )
            if not (object_names_value or preference_names_value):
                continue
        if (
            item.polarity == "negative"
            or item.modality in {"planned", "possible", "hypothetical"}
            or _DISCOURSE.search(predicate)
            or not _ACCEPTED.search(relation_text)
        ):
            continue
        accepted = 1
        if re.search(r"\b(?:adopt|choose|chose|decide|final|select|settle)\w*\b", predicate):
            accepted = 4
        elif re.search(r"\b(?:like|love|prefer)\w*\b", predicate):
            accepted = 3
        elif re.search(r"\b(?:name|use|call)\w*\b", relation_text.casefold()):
            accepted = 2
        if _ONLY_PROPOSED.search(predicate):
            accepted = 0
        sequence_match = re.search(r"(\d+)\Z", item.operand_id)
        sequence = int(sequence_match.group(1)) if sequence_match else -1
        candidates.append((
            relation_overlap, accepted,
            _date(item.observed_at or item.event_time) or datetime.min,
            sequence,
            max(0.0, semantic_similarity(item)),
            item.confidence,
            item,
        ))
    if not candidates:
        return None
    _overlap, accepted, _date_value, _sequence, _semantic, _confidence, item = max(
        candidates, key=lambda row: (*row[:6], row[6].operand_id)
    )
    if accepted < 2:
        return None
    value = item.object_text.strip().strip(chr(39) + chr(34))
    if name_query:
        name_match = re.search(r"\bnam(?:e|ed|es)\s+(.+?)\Z", value, re.IGNORECASE)
        if name_match is not None:
            value = name_match.group(1).strip().strip(chr(39) + chr(34))
    return {
        "operation": "final_choice_state",
        "value": value,
        "predicate": item.predicate_key,
        "operand_ids": [item.operand_id],
        "source_turn_ids": list(item.source_turn_ids),
        "complete": True,
    }


def _continuous_start(item: OperandRecordV3, observed: datetime) -> datetime:
    text = " ".join((
        item.predicate_key, item.object_text, item.context_key, item.retrieval_text
    )).casefold()
    if not re.search(r"\b(?:has|have|had)\s+been\b|\b(?:past|for)\b", text):
        return observed
    match = re.search(
        r"\b(?:for\s+(?:the\s+)?|past\s+)(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month|year)s?\b", text,
    )
    if match is None:
        return observed
    amount = int(match.group(1)) if match.group(1).isdigit() else _NUMBER_WORDS[match.group(1)]
    days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[match.group(2)]
    return observed - timedelta(days=days)


def earliest_alternative_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
) -> dict[str, object] | None:
    """Choose the earlier of two explicitly named alternatives."""
    if frame.requested_operation != "earliest":
        return None
    match = re.search(
        r",\s*(?:the\s+)?(.+?)\s+or\s+(?:the\s+)?(.+?)[?]?$",
        frame.raw_question.casefold(),
    )
    if not match:
        return None
    relation_prefix = frame.raw_question.casefold().split(",", 1)[0]
    relation_ignored = {
        "a", "an", "did", "do", "does", "event", "first", "happen",
        "happened", "has", "have", "had", "is", "occur", "occurred",
        "the", "was", "were", "what", "which", "who",
    }
    relation_terms = {
        value for value in re.findall(r"[a-z0-9]+", relation_prefix)
        if value not in relation_ignored
    }
    ignored = {
        "a", "an", "did", "event", "first", "for", "i", "in", "local",
        "money", "participate", "raise", "the", "to",
    }
    phrases = []
    for raw in match.groups():
        terms = {
            value for value in re.findall(r"[a-z0-9]+", raw)
            if value not in ignored
        }
        phrases.append((raw.strip(), terms))
    bound = []
    for raw, terms in phrases:
        candidates = []
        for item in operands:
            observed = _date(item.event_time) or _date(item.observed_at)
            if observed is not None:
                observed = _continuous_start(item, observed)
            if observed is None or item.modality in {"planned", "possible", "hypothetical"}:
                continue
            item_terms = set(re.findall(
                r"[a-z0-9]+",
                (
                    item.subject_key + " " + item.predicate_key + " "
                    + item.object_text + " " + item.context_key
                ).casefold(),
            ))
            overlap = len(terms & item_terms)
            if overlap <= 0:
                continue
            relation_overlap = len(relation_terms & item_terms)
            if relation_terms and relation_overlap <= 0:
                continue
            candidates.append((overlap, relation_overlap, observed, item))
        if not candidates:
            return {
                "operation": "named_alternative_incomplete",
                "value": "insufficient evidence for every named alternative",
                "missing_alternatives": [raw],
                "operand_ids": [],
                "source_turn_ids": [],
                "complete": True,
            }
        overlap, _relation_overlap, observed, item = max(
            candidates, key=lambda row: (row[0], row[1], row[2], row[3].operand_id)
        )
        bound.append((observed, raw, overlap, item))
    observed, raw, _overlap, item = min(bound, key=lambda row: (row[0], row[1]))
    return {
        "operation": "earliest_named_alternative",
        "value": raw,
        "date": observed.date().isoformat(),
        "operand_ids": [item.operand_id],
        "source_turn_ids": list(item.source_turn_ids),
        "complete": True,
    }


def ordered_event_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    *,
    semantic_similarity: Callable[[OperandRecordV3], float],
    object_semantic_similarity: Callable[[OperandRecordV3], float] | None = None,
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, object] | None:
    """Order a bounded, query-semantic event collection by its event dates."""
    if frame.requested_operation != "ordering":
        return None
    requested = next((
        value for word, value in _NUMBER_WORDS.items()
        if re.search(
            rf"\b{word}\s+(?!(?:days?|weeks?|months?|years?)\b)[a-z]",
            frame.raw_question.casefold(),
        )
    ), None)
    if requested is None:
        match = re.search(
            r"\b(\d+)\s+(?!(?:days?|weeks?|months?|years?)\b)[a-z]",
            frame.raw_question.casefold(),
        )
        requested = int(match.group(1)) if match else None
    open_collection = (
        requested is None
        and bool(re.search(
            r"\b(?:past|last|recent|all|every)\b",
            frame.raw_question.casefold(),
        ))
    )
    if (requested is None and not open_collection) or (requested is not None and (requested <= 1 or requested > 10)):
        return None
    limit = requested or 10
    def normalized_terms(value: str) -> set[str]:
        values = set()
        for raw in re.findall(r"[a-z0-9]+", value.casefold()):
            token = raw
            if len(token) > 5 and token.endswith("ical"):
                token = token[:-2]
            elif len(token) > 4 and token.endswith("ed"):
                token = token[:-2]
            elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            values.add(token)
        return values
    action_families = [
        ({"attend"}, r"\battend\w*\b"),
        ({"complet", "finish"}, r"\b(?:complet|finish)\w*\b"),
        ({"join", "participat"}, r"\b(?:join|participat|take part)\w*\b"),
        ({"visit"}, r"\bvisit\w*\b"),
        ({"meet"}, r"\b(?:meet|met)\b"),
    ]
    question_action_family = next((
        family for family, pattern in action_families
        if re.search(pattern, frame.raw_question.casefold())
    ), set())
    ignored_targets = {
        "all", "and", "earliest", "event", "from", "last", "latest",
        "month", "order", "past", "recent", "start", "starting", "the",
        "two", "week", "year", *question_action_family,
    }
    target_terms = normalized_terms(frame.raw_question) - ignored_targets
    candidates = []
    object_semantic_scores: dict[str, float] = {}
    lexical_type_matches: set[str] = set()
    projected_type_matches: set[str] = set()
    projected_type_mismatches: set[str] = set()
    for item in operands:
        observed = _date(item.event_time) or _date(item.observed_at)
        if (
            observed is None
            or item.polarity == "negative"
            or item.modality in {"planned", "possible", "hypothetical"}
            or _DISCOURSE.search(item.predicate_key)
        ):
            continue
        predicate_terms = normalized_terms(item.predicate_key)
        if question_action_family and not (predicate_terms & question_action_family):
            continue
        semantic = max(0.0, (object_semantic_similarity or semantic_similarity)(item))
        object_semantic_scores[item.operand_id] = semantic
        object_terms = normalized_terms(item.object_text)
        if target_terms & object_terms:
            lexical_type_matches.add(item.operand_id)
        projected_types = normalized_terms(" ".join(item.event_type_keys))
        if projected_types:
            if target_terms & projected_types:
                projected_type_matches.add(item.operand_id)
            else:
                projected_type_mismatches.add(item.operand_id)
        overlap = query_overlap(frame, item.retrieval_text)
        action = int(bool(re.search(
            r"\b(?:attend|complete|join|participat|take part)\w*\b",
            item.predicate_key,
            flags=re.IGNORECASE,
        )))
        if not action:
            continue
        candidates.append((
            3.0 * semantic + overlap + action,
            observed,
            item,
        ))
    candidates.sort(key=lambda row: (row[0], row[1], row[2].operand_id), reverse=True)
    if open_collection and candidates:
        best_semantic = max(object_semantic_scores.values(), default=0.0)
        semantic_floor = max(0.48, best_semantic * 0.72)
        candidates = [
            row for row in candidates
            if (
                row[2].operand_id in lexical_type_matches
                or row[2].operand_id in projected_type_matches
                or (
                    row[2].operand_id not in projected_type_mismatches
                    and object_semantic_scores.get(row[2].operand_id, 0.0)
                    >= semantic_floor
                )
            )
        ]
    clusters: list[tuple[set[str], tuple[float, datetime, OperandRecordV3]]] = []
    used_source_turns: set[str] = set()
    generic_event_terms = {
        "activity", "concert", "event", "live", "music", "musical", "show",
        "the", "at", "in", "with",
    }
    for row in candidates:
        source_turns = set(row[2].source_turn_ids)
        if source_turns & used_source_turns:
            continue
        identity = {
            value for value in re.findall(r"[a-z0-9]+", row[2].object_key.casefold())
            if value not in generic_event_terms
        }
        duplicate_index = next((
            index for index, (old_identity, _old_row) in enumerate(clusters)
            if identity and old_identity
            and len(identity & old_identity) >= min(2, len(identity), len(old_identity))
        ), None)
        if duplicate_index is None:
            clusters.append((identity, row))
        else:
            old_identity, old_row = clusters[duplicate_index]
            # Retrospective repeats are not new occurrences; keep the earliest
            # grounded observation for chronological collection ordering.
            if row[1] < old_row[1]:
                clusters[duplicate_index] = (old_identity | identity, row)
            else:
                clusters[duplicate_index] = (old_identity | identity, old_row)
        used_source_turns.update(source_turns)
        if requested is not None and len(clusters) >= limit:
            break
    if len(clusters) < (requested or 2):
        return None
    distinct = {str(index): row for index, (_identity, row) in enumerate(clusters[:limit])}
    ordered = sorted(distinct.values(), key=lambda row: (row[1], row[2].operand_id))
    return {
        "operation": "ordered_event_collection",
        "values": [
            {
                "value": item.object_text,
                "predicate": item.predicate_key,
                "date": observed.date().isoformat(),
                "operand_id": item.operand_id,
            }
            for _score, observed, item in ordered
        ],
        "source_turn_ids": list(dict.fromkeys(
            source for _score, _date_value, item in ordered
            for source in item.source_turn_ids
        )),
        "complete": True,
    }
