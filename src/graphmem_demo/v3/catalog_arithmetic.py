from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Any, Callable

from .action_semantics import has_completed_participation
from .catalog_schema import EventFrameV3, OperandRecordV3
from .schema import QueryFrame


_IRREGULAR = {
    "bought": "buy", "rode": "ride", "spent": "spend", "took": "take",
    "went": "go", "met": "meet", "used": "use", "paid": "pay",
    "led": "lead", "leading": "lead", "purchased": "purchase",
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_MONEY_UNITS = {
    "usd", "dollar", "dollars", "eur", "euro", "euros",
    "gbp", "pound", "pounds", "cad", "aud", "yen",
    "$", "€", "£", "¥",
}
_GENERIC = {
    "all", "amount", "at", "different", "each", "expense", "expenses", "for",
    "from", "have", "how", "in", "item", "items", "many", "money", "much",
    "of", "on", "relat", "related", "since", "start", "the", "this", "time", "times",
    "total", "type", "types", "event", "events", "year",
}
_ACTION_WORDS = {
    "acquire", "add", "attend", "buy", "cost", "do", "download", "go", "inherit", "install", "meet", "pay",
    "pick", "purchase", "replace", "return", "ride", "spend", "take", "use",
}


def _token(value: str) -> str:
    value = value.casefold().strip("'\"")
    if value.endswith("'s"):
        value = value[:-2]
    if value in _IRREGULAR:
        return _IRREGULAR[value]
    if len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return _IRREGULAR.get(value, value)


def _tokens(value: str) -> set[str]:
    return {
        _token(item)
        for item in re.findall(r"[\w']+", value.replace("_", " ").replace("-", " "))
        if len(item) > 1
    }


def _is_money(item: OperandRecordV3) -> bool:
    unit = item.unit.casefold().strip()
    return (
        item.quantity is not None
        and (
            unit in _MONEY_UNITS
            or any(symbol in unit for symbol in ("$", "€", "£", "¥"))
        )
    )


def _asserted(item: OperandRecordV3) -> bool:
    return (
        item.polarity != "negative"
        and item.modality not in {"planned", "possible", "hypothetical"}
        and item.state_op not in {"remove", "cancel", "retract"}
    )


def _entity_terms(value: str) -> set[str]:
    return {
        token for token in _tokens(value)
        if token not in _GENERIC and not token.isdigit()
    }


def _number_in_text(value: str) -> int | None:
    match = re.search(r"\b(\d+)\b", value)
    if match:
        return int(match.group(1))
    lowered = value.casefold()
    for word, number in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return number
    return None


def _time_key(item: OperandRecordV3) -> tuple[int, int, int, int, int, int]:
    text = item.observed_at or item.event_time or ""
    match = re.search(
        r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"
        r"(?:[^0-9]{0,20}(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
        text,
    )
    if not match:
        return (0, 0, 0, 0, 0, 0)
    year, month, day, hour, minute, second = match.groups()
    return (
        int(year), int(month), int(day), int(hour or 0),
        int(minute or 0), int(second or 0),
    )


def _dedupe_money(
    rows: list[OperandRecordV3],
    operands: list[OperandRecordV3],
) -> list[OperandRecordV3]:
    """Collapse repeated reports using the local event context, not amount alone."""
    contextual_terms: dict[str, set[str]] = {}
    direct_terms: dict[str, set[str]] = {
        item.operand_id: _entity_terms(item.predicate_key + " " + item.object_text)
        for item in rows
    }
    for item in rows:
        source_ids = set(item.source_turn_ids)
        contextual_terms[item.operand_id] = set().union(*(
            _entity_terms(peer.predicate_key + " " + peer.object_text)
            for peer in operands
            if (
                source_ids & set(peer.source_turn_ids)
                or (
                    item.event_frame_id
                    and peer.event_frame_id == item.event_frame_id
                )
            )
        ))
    kept: list[OperandRecordV3] = []
    for item in sorted(rows, key=lambda row: row.confidence, reverse=True):
        terms = contextual_terms[item.operand_id]
        duplicate = False
        for old in kept:
            if item.quantity != old.quantity:
                continue
            old_terms = contextual_terms[old.operand_id]
            overlap = len(terms & old_terms) / max(1, min(len(terms), len(old_terms)))
            if (
                overlap >= 0.6
                or len(direct_terms[item.operand_id] & old_terms) >= 2
                or len(direct_terms[old.operand_id] & terms) >= 2
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def per_item_amount(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    question = frame.raw_question.casefold()
    if not (
        re.search(r"\bhow much\b", question)
        and re.search(r"\b(?:each|per)\b", question)
    ):
        return None
    money = [
        item for item in operands
        if _asserted(item) and _is_money(item) and query_overlap(frame, item.retrieval_text) > 0
    ]
    counts = [
        item for item in operands
        if _asserted(item) and item.quantity is not None and item.quantity > 0
        and not _is_money(item) and query_overlap(frame, item.retrieval_text) > 0
    ]
    candidates = []
    query_entities = _entity_terms(frame.raw_question)
    for total in money:
        total_terms = _entity_terms(total.retrieval_text)
        for count in counts:
            count_terms = _entity_terms(count.retrieval_text)
            shared = total_terms & count_terms & query_entities
            if not shared:
                continue
            score = (
                3 * len(shared)
                + query_overlap(frame, total.retrieval_text)
                + query_overlap(frame, count.retrieval_text)
                + int(bool(set(total.session_ids) & set(count.session_ids)))
            )
            candidates.append((score, total, count))
    if not candidates:
        return None
    _score, total, count = max(candidates, key=lambda row: row[0])
    value = float(total.quantity) / float(count.quantity)
    return {
        "operation": "per_item_amount",
        "value": value,
        "unit": total.unit,
        "total": total.quantity,
        "count": count.quantity,
        "operand_ids": [total.operand_id, count.operand_id],
        "source_turn_ids": list(dict.fromkeys(
            [*total.source_turn_ids, *count.source_turn_ids]
        )),
        "complete": True,
    }


def total_money(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    question = frame.raw_question.casefold()
    if not (
        re.search(r"\b(?:how much|total|sum)\b", question)
        and re.search(
            r"\b(?:money|cost|spend|spent|pay|paid|price|quote|expense)\b",
            question,
        )
    ):
        return None
    money = [item for item in operands if _asserted(item) and _is_money(item)]
    delta_query = bool(re.search(
        r"\b(?:how much more|how much less|difference|extra|additional)\b",
        question,
    ))
    if delta_query:
        query_terms = _entity_terms(frame.raw_question) - _ACTION_WORDS
        scoped = [
            item for item in money
            if query_overlap(frame, item.retrieval_text) > 0
            and bool(query_terms & _entity_terms(item.retrieval_text))
        ]
        baselines = [
            item for item in scoped
            if re.search(
                r"\b(?:initial|original|quote|quoted|estimate|estimated)\w*\b",
                (item.predicate_key + " " + item.object_text + " " + item.context_key).casefold(),
            )
        ]
        updates = [
            item for item in scoped
            if re.search(
                r"\b(?:actual|correct|corrected|final|revised|updated)\w*\b",
                (item.predicate_key + " " + item.object_text + " " + item.context_key).casefold(),
            )
        ]
        pairs = [
            (baseline, update)
            for baseline in baselines for update in updates
            if baseline.operand_id != update.operand_id
            and (
                _entity_terms(
                    baseline.predicate_key + " " + baseline.object_text + " "
                    + baseline.context_key
                )
                & _entity_terms(
                    update.predicate_key + " " + update.object_text + " "
                    + update.context_key
                )
                & query_terms
            )
        ]
        if pairs:
            baseline, update = max(
                pairs,
                key=lambda pair: (
                    query_overlap(frame, pair[0].retrieval_text)
                    + query_overlap(frame, pair[1].retrieval_text),
                    _time_key(pair[1]),
                ),
            )
            return {
                "operation": "money_difference",
                "value": abs(float(update.quantity) - float(baseline.quantity)),
                "unit": update.unit or baseline.unit,
                "baseline": baseline.quantity,
                "updated": update.quantity,
                "operand_ids": [baseline.operand_id, update.operand_id],
                "source_turn_ids": list(dict.fromkeys([
                    *baseline.source_turn_ids, *update.source_turn_ids,
                ])),
                "complete": True,
                "completion_basis": "entity_bound_initial_updated_money_pair",
            }
    revenue_query = bool(re.search(
        r"\b(?:earn|earned|earning|fundrais\w*|income|rais(?:e|ed|ing)|"
        r"revenue|proceeds|selling|sold)\b",
        question,
    ))
    if revenue_query:
        sale_query = bool(re.search(r"\b(?:sell|selling|sold|market)\b", question))
        direct_relation = (
            r"\b(?:earn|profit|proceeds|revenue)\w*\b"
            if sale_query else
            r"\b(?:earn|fundrais|income|profit|proceeds|rais|revenue)\w*\b"
        )
        direct = [
            item for item in money
            if re.search(direct_relation, item.predicate_key.casefold())
            and (
                not sale_query
                or bool(
                    _tokens(item.context_key + " " + item.retrieval_text)
                    & {"market", "sale", "sell", "sold", "product"}
                )
            )
        ]
        derived = []
        by_source: dict[str, list[OperandRecordV3]] = defaultdict(list)
        for candidate in operands:
            for source_id in candidate.source_turn_ids:
                by_source[source_id].append(candidate)
        for source_id, rows in by_source.items():
            counts = [
                item for item in rows
                if _asserted(item) and item.quantity is not None and not _is_money(item)
                and re.search(r"\b(?:sell|sold)\w*\b", item.predicate_key.casefold())
            ]
            prices = [
                item for item in rows
                if _asserted(item) and _is_money(item)
                and (
                    re.search(r"\b(?:each|per)\b", item.object_text.casefold())
                    or re.search(r"\bprice\b", item.predicate_key.casefold())
                )
            ]
            for count in counts:
                for price in prices:
                    derived.append((source_id, count, price))
        if direct or derived:
            direct = _dedupe_money(direct, operands)
            value = sum(float(item.quantity or 0) for item in direct) + sum(
                float(count.quantity or 0) * float(price.quantity or 0)
                for _source, count, price in derived
            )
            ids = [item.operand_id for item in direct]
            ids.extend(
                operand.operand_id
                for _source, count, price in derived for operand in (count, price)
            )
            sources = [source for item in direct for source in item.source_turn_ids]
            sources.extend(source for source, _count, _price in derived)
            return {
                "operation": "aggregate_revenue_total",
                "value": value,
                "unit": next((item.unit for item in direct if item.unit), "USD"),
                "operand_ids": list(dict.fromkeys(ids)),
                "source_turn_ids": list(dict.fromkeys(sources)),
                "complete": True,
            }
    explicit_total = bool(re.search(
        r"\b(?:total|sum|in total|altogether|combined)\b", question
    ))
    if not explicit_total:
        target_terms = _entity_terms(frame.raw_question) - _ACTION_WORDS
        transaction_rows = [
            item for item in money
            if re.search(
                r"\b(?:buy|bought|cost|paid|pay|purchase|spend|spent)\w*\b",
                (item.predicate_key + " " + item.object_text).casefold(),
            )
        ]
        scored_rows = [
            (
                len(target_terms & _entity_terms(
                    item.predicate_key + " " + item.object_text + " " + item.context_key
                )),
                query_overlap(frame, item.retrieval_text),
                item,
            )
            for item in transaction_rows
        ]
        best_binding = max((row[0] for row in scored_rows), default=0)
        entity_bound = _dedupe_money(
            [row[2] for row in scored_rows if row[0] == best_binding and best_binding > 0],
            operands,
        )
        if len(entity_bound) == 1:
            item = entity_bound[0]
            return {
                "operation": "money_amount",
                "value": float(item.quantity),
                "unit": item.unit,
                "operand_ids": [item.operand_id],
                "source_turn_ids": list(item.source_turn_ids),
                "complete": True,
                "completion_basis": "single_explicit_entity_bound_money_value",
            }

    anchors = [
        item for item in money if query_overlap(frame, item.retrieval_text) > 0
    ]
    if not anchors:
        return None
    query_terms = _entity_terms(frame.raw_question)
    relation_terms = set().union(*(
        _tokens(item.predicate_key) & query_terms for item in operands
    ))
    target_terms = query_terms - _ACTION_WORDS
    contextual_anchors = [
        item for item in operands
        if _entity_terms(item.retrieval_text) & target_terms
    ]
    anchor_sessions = {
        value for item in [*anchors, *contextual_anchors] for value in item.session_ids
    }
    anchor_turns = {value for item in anchors for value in item.source_turn_ids}
    query_entities = _entity_terms(frame.raw_question)
    scoped = [
        item for item in money
        if (
            _entity_terms(item.retrieval_text) & target_terms
            or set(item.session_ids) & anchor_sessions
            or set(item.source_turn_ids) & anchor_turns
        )
        and re.search(
            r"\b(?:buy|bought|cost|expense|install|paid|pay|purchase|repair|replace|spend|spent)\b",
            (item.predicate_key + " " + item.object_text).casefold(),
        )
    ]
    scoped = _dedupe_money(scoped, operands)
    if not scoped:
        return None
    if len(scoped) == 1 and not re.search(
        r"\b(?:total|sum|in total|altogether)\b", question
    ):
        item = scoped[0]
        return {
            "operation": "money_amount",
            "value": float(item.quantity),
            "unit": item.unit,
            "operand_ids": [item.operand_id],
            "source_turn_ids": list(item.source_turn_ids),
            "complete": True,
            "completion_basis": "single_entity_bound_asserted_money_value",
        }
    if len(scoped) < 2:
        return None
    return {
        "operation": "total_money",
        "value": sum(float(item.quantity) for item in scoped),
        "unit": next((item.unit for item in scoped if item.unit), ""),
        "items": [
            {
                "operand_id": item.operand_id,
                "description": item.object_text,
                "amount": item.quantity,
            }
            for item in scoped
        ],
        "operand_ids": [item.operand_id for item in scoped],
        "source_turn_ids": list(dict.fromkeys(
            value for item in scoped for value in item.source_turn_ids
        )),
        "complete": True,
    }


def _event_instance_key(item: OperandRecordV3) -> str:
    """Canonicalize the instance-bearing part of an event operand."""
    text = item.object_text.casefold()
    location = re.search(
        r"\b(?:at|in|to)\s+([a-z0-9][a-z0-9' -]{1,80}?)(?:\s+place)?(?:[,.;]|$)",
        text,
    )
    if location:
        identity = location.group(1)
    else:
        identity = item.object_key or " ".join(sorted(_entity_terms(item.object_text)))
    identity = re.sub(r"'s\b", "", identity)
    identity = re.sub(r"\s+", " ", identity).strip()
    time = (item.event_time or "").casefold().strip()
    return f"{identity}|{time}"


def event_occurrence_count(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    event_frames: list[EventFrameV3],
    query_overlap: Callable[[QueryFrame, str], float],
    turns: list[Any] | None = None,
) -> dict[str, Any] | None:
    if frame.requested_operation != "count":
        return None
    question = frame.raw_question.casefold()
    # A scalar amount request is never an event-occurrence count. If its
    # quantity projection is missing, preserve uncertainty for source closure.
    if re.search(r"\bhow much\b", question):
        return None
    if re.search(r"\b(?:plan|planned|planning|intend|intended|schedule|scheduled|agree|agreed)\b", question):
        return None
    if re.search(r"\bor\b", question) and re.search(r"\b(?:current|currently|now)\b", question):
        return None
    if re.search(
        r"\bhow many\s+(?:different\s+)?(?:types?|hours?|minutes?|"
        r"days?|weeks?|months?|years?|dollars?|euros?|pounds?)\b",
        question,
    ) or re.search(r"\bhow often\b", question):
        return None
    question_terms = _tokens(frame.raw_question)
    query_actions = question_terms & _ACTION_WORDS
    target_heads = _count_target_heads(frame.raw_question)
    target_terms = _count_target_terms(frame.raw_question)
    target_modifiers = target_terms - target_heads
    qualifier_match = re.search(
        r"\b(?:due\s+to|because\s+of|because)\s+([^?.,;]+)",
        frame.raw_question,
        re.IGNORECASE,
    )
    qualifier_terms = (
        _entity_terms(qualifier_match.group(1))
        if qualifier_match else set()
    )
    turn_by_id = {
        str(getattr(turn, "node_id", "")): turn
        for turn in (turns or [])
    }
    candidates = []
    frames = {item.frame_id: item for item in event_frames}
    for item in operands:
        if not _asserted(item):
            continue
        # An intent relation is not a completed occurrence even when an
        # extractor conservatively labels the record as asserted.
        if re.search(
            r"\b(?:plan|planning|intend|schedule|hope|expect)\w*\b",
            item.predicate_key.casefold(),
        ):
            continue
        predicate_terms = _tokens(item.predicate_key)
        frame_terms: set[str] = set()
        if item.event_frame_id and item.event_frame_id in frames:
            frame_terms = _tokens(frames[item.event_frame_id].label)
        combined_terms = (
            predicate_terms | _tokens(item.object_text)
            | _tokens(item.context_key) | frame_terms
        )
        meaningful_terms = question_terms - _GENERIC
        action_match = (predicate_terms | frame_terms) & (query_actions or meaningful_terms)
        participation_actions = {
            "attend", "go", "participate", "take", "visit", "volunteer",
        }
        if query_actions & {"attend", "go", "participate", "take", "visit"}:
            action_match = predicate_terms & participation_actions
        provenance_acquisition = bool(
            query_actions & {"acquire", "inherit"}
            and predicate_terms & {"have", "item", "own"}
            and item.context_key.strip()
            and "from" in question_terms
        )
        if provenance_acquisition:
            action_match = {"acquire"}
        frame_member_terms: set[str] = set()
        if item.event_frame_id:
            frame_member_terms = set().union(*(
                _tokens(
                    member.predicate_key + " " + member.object_text + " "
                    + member.context_key
                )
                for member in operands
                if member.event_frame_id == item.event_frame_id
            ))
        direct_terms = (
            predicate_terms | _tokens(item.object_text)
            | _tokens(item.context_key) | frame_member_terms
        )
        if target_heads and not (target_heads & direct_terms):
            continue
        if target_modifiers and not target_modifiers.issubset(direct_terms):
            continue
        if item.event_frame_id and item.event_frame_id in frames:
            # Secondary operands can mention the requested words while belonging
            # to an unrelated coarse event. The frame must support the action and target.
            if query_actions and not (
                (frame_terms | predicate_terms) & query_actions
            ):
                continue
        typed_frequency = (
            item.quantity is not None
            and bool(_tokens(item.unit) & {"time", "occurrence"})
        )
        if (
            not action_match
            or (len(combined_terms & meaningful_terms) < 2 and not typed_frequency)
        ):
            continue
        if qualifier_terms and turn_by_id:
            source_terms = set().union(*(
                _tokens(str(getattr(turn_by_id[source_id], "text", "")))
                for source_id in item.source_turn_ids
                if source_id in turn_by_id
            ))
            if (
                source_terms
                and len(source_terms & qualifier_terms)
                < max(1, (len(qualifier_terms) + 1) // 2)
            ):
                continue
        contribution = 1
        if item.quantity is not None and (
            "time" in _tokens(item.unit) or "occurrence" in _tokens(item.unit)
        ):
            contribution = int(item.quantity)
        else:
            contribution_text = " ".join((
                item.predicate_key,
                item.object_text,
                frames[item.event_frame_id].label
                if item.event_frame_id and item.event_frame_id in frames else "",
            ))
            stated_values = [
                number for word, number in _NUMBER_WORDS.items()
                if re.search(rf"\b{word}\b", contribution_text.casefold())
            ]
            stated = max(stated_values, default=None)
            if stated is not None:
                contribution = stated
            elif "," in item.object_text:
                contribution = len([
                    value for value in item.object_text.split(",") if value.strip()
                ])
        candidates.append((item, max(1, contribution)))
    if not candidates:
        return None

    by_scope: dict[tuple[str, ...], list[tuple[OperandRecordV3, int]]] = defaultdict(list)
    for item, value in candidates:
        # Source provenance collapses direct and frame projections of one utterance.
        scope = tuple(sorted(item.source_turn_ids)) or (f"operand:{item.operand_id}",)
        by_scope[scope].append((item, value))

    groups = []
    for _scope, rows in by_scope.items():
        source_ids = tuple(dict.fromkeys(
            source for item, _value in rows for source in item.source_turn_ids
        ))
        if any(value > 1 for _item, value in rows):
            value = max(value for _item, value in rows)
        else:
            framed = {
                item.event_frame_id
                for item, _value in rows
                if item.event_frame_id
            }
            unframed = {
                _event_instance_key(item)
                for item, _value in rows
                if not item.event_frame_id
            }
            # One turn may project several relations from one event.  The
            # frame, not the number of predicates, is its occurrence identity.
            value = len(framed) + len(unframed)
        frame_ids = {item.event_frame_id for item, _value in rows if item.event_frame_id}
        participant_keys = {
            token
            for frame_id in frame_ids if frame_id in frames
            for key in frames[frame_id].participant_keys
            for token in _tokens(key)
            if token not in {
                "participant", "cousin", "friend", "roommate", "college",
                "partner", "husband", "wife",
            } and not token.isdigit()
        }
        participant_keys.update({
            _token(name)
            for item, _value in rows
            for name in re.findall(
                r"\b(?:cousin|friend|roommate|partner|husband|wife)\s+([a-z][\w'-]+)",
                (item.predicate_key + " " + item.object_text).casefold(),
            )
            if _token(name) not in meaningful_terms and name not in {"event", "state"}
        })
        groups.append({
            "value": value,
            "rows": rows,
            "source_turn_ids": list(source_ids),
            "participant_keys": participant_keys,
            "time_key": max((_time_key(item) for item, _value in rows), default=(0, 0, 0, 0, 0, 0)),
            "entity_quantity": any(
                item.quantity is not None
                and not ({"time", "occurrence"} & _tokens(item.unit))
                for item, _value in rows
            ),
            "cumulative": any(
                re.search(
                    r"\b(?:already|so far|to date|in total|altogether|up to now)\b",
                    (
                        item.object_text + " " + item.retrieval_text + " "
                        + " ".join(
                            str(getattr(turn_by_id[source_id], "text", ""))
                            for source_id in item.source_turn_ids
                            if source_id in turn_by_id
                        )
                    ).casefold(),
                )
                and value > 1
                for item, _value in rows
            ),
        })

    scalar_groups = [group for group in groups if group["entity_quantity"]]
    if scalar_groups:
        latest_scalar = max(scalar_groups, key=lambda group: group["time_key"])
        scalar_units = set().union(*(
            _tokens(item.unit) for item, _value in latest_scalar["rows"]
        ))
        scalar_predicates = set().union(*(
            _tokens(item.predicate_key) for item, _value in latest_scalar["rows"]
        ))
        groups = [
            group for group in groups
            if group is latest_scalar or not any(
                scalar_units & _tokens(item.object_text + " " + item.unit)
                and scalar_predicates & _tokens(item.predicate_key)
                for item, _value in group["rows"]
            )
        ]

    cumulative = [group for group in groups if group["cumulative"]]
    if cumulative:
        latest = max(cumulative, key=lambda group: group["time_key"])
        latest_terms = set().union(*(
            _entity_terms(item.predicate_key + " " + item.object_text)
            for item, _value in latest["rows"]
        )) - set(_NUMBER_WORDS)
        groups = [
            group for group in groups
            if group is latest or not (latest_terms & (set().union(*(
                _entity_terms(item.predicate_key + " " + item.object_text)
                for item, _value in group["rows"]
            )) - set(_NUMBER_WORDS)))
        ]

    merged = []
    for group in groups:
        match = next((
            old for old in merged
            if group["participant_keys"] and old["participant_keys"]
            and group["participant_keys"] & old["participant_keys"]
        ), None)
        if match is None:
            merged.append(group)
        else:
            match["value"] = max(match["value"], group["value"])
            match["rows"].extend(group["rows"])
            match["source_turn_ids"].extend(group["source_turn_ids"])
            match["participant_keys"].update(group["participant_keys"])
    represented_turns = {
        source for group in merged for source in group["source_turn_ids"]
    }
    missing_turns = []
    if turns:
        meaningful = set(frame.content_terms) - _GENERIC - _ACTION_WORDS - {
            "participate", "visit", "volunteer",
        }
        for turn in turns:
            if turn.node_id in represented_turns:
                continue
            raw = str(getattr(turn, "text", ""))
            lowered = raw.casefold()
            raw_terms = _tokens(raw)
            speaker_key = str(getattr(turn, "speaker_key", "")).casefold()
            if re.search(r"\b(?:i|me|my)\b", question) and (
                "participant 2" in speaker_key or "assistant" in speaker_key
            ):
                continue
            if not has_completed_participation(lowered):
                continue
            if target_heads and not (target_heads & raw_terms):
                continue
            if qualifier_terms and (
                len(qualifier_terms & raw_terms)
                < max(1, (len(qualifier_terms) + 1) // 2)
            ):
                continue
            if len(meaningful & raw_terms) < 1:
                continue
            missing_turns.append(turn.node_id)
    return {
        "operation": "event_occurrence_count",
        "value": sum(group["value"] for group in merged),
        "groups": [
            {
                "count": group["value"],
                "operand_ids": [item.operand_id for item, _value in group["rows"]],
                "source_turn_ids": list(dict.fromkeys(group["source_turn_ids"])),
            }
            for group in merged
        ],
        "operand_ids": [
            item.operand_id for group in merged for item, _value in group["rows"]
        ],
        "source_turn_ids": list(dict.fromkeys(
            value for group in merged for value in group["source_turn_ids"]
        )),
        "complete": not missing_turns,
        "value_lower_bound": sum(group["value"] for group in merged),
        "candidate_value": sum(group["value"] for group in merged) + len(missing_turns),
        "missing_source_turn_ids": missing_turns[:8],
    }


def ratio_percent(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    event_frames: list[EventFrameV3],
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    """Resolve an explicitly requested A-as-a-percentage-of-B ratio."""
    question = frame.raw_question.casefold()
    match = re.search(
        r"\b(?:what|how much) percentage of (.+?) (?:is|are|does|would) (.+?)[?]?$",
        question,
    )
    if not match:
        return None
    denominator_terms = _entity_terms(match.group(1))
    numerator_terms = _entity_terms(match.group(2))
    role_terms = {"amount", "cost", "price", "total", "value"}
    denominator_anchors = denominator_terms - role_terms
    numerator_anchors = numerator_terms - role_terms
    frames = {item.frame_id: item for item in event_frames}
    numeric = [
        item for item in operands
        if _asserted(item) and item.quantity is not None and item.quantity > 0
        and query_overlap(frame, item.retrieval_text) > 0
    ]
    candidates = []
    for denominator in numeric:
        denominator_text = _entity_terms(
            denominator.predicate_key + " " + denominator.object_text + " " + denominator.unit
        )
        if denominator.event_frame_id and denominator.event_frame_id in frames:
            denominator_text |= _entity_terms(frames[denominator.event_frame_id].label)
        denominator_match = len(denominator_terms & denominator_text)
        if denominator_anchors and not (denominator_anchors & denominator_text):
            continue
        if denominator_match == 0:
            continue
        for numerator in numeric:
            if numerator.operand_id == denominator.operand_id:
                continue
            if (
                denominator.unit and numerator.unit
                and denominator.unit.casefold() != numerator.unit.casefold()
            ):
                continue
            numerator_text = _entity_terms(
                numerator.predicate_key + " " + numerator.object_text + " " + numerator.unit
            )
            if numerator.event_frame_id and numerator.event_frame_id in frames:
                numerator_text |= _entity_terms(frames[numerator.event_frame_id].label)
            numerator_match = len(numerator_terms & numerator_text)
            if numerator_anchors and not (numerator_anchors & numerator_text):
                continue
            if numerator_match == 0:
                continue
            score = (
                4 * denominator_match + 4 * numerator_match
                + query_overlap(frame, denominator.retrieval_text)
                + query_overlap(frame, numerator.retrieval_text)
            )
            candidates.append((score, numerator, denominator))
    if not candidates:
        return None
    _score, numerator, denominator = max(candidates, key=lambda row: row[0])
    value = 100.0 * float(numerator.quantity) / float(denominator.quantity)
    return {
        "operation": "ratio_percent",
        "value": round(value, 6),
        "numerator": numerator.quantity,
        "denominator": denominator.quantity,
        "operand_ids": [numerator.operand_id, denominator.operand_id],
        "source_turn_ids": list(dict.fromkeys([
            *numerator.source_turn_ids, *denominator.source_turn_ids,
        ])),
        "complete": True,
    }

def _initial_window_days(question: str) -> int | None:
    match = re.search(
        r"\b(?:first|initial)\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month|year)s?\b",
        question.casefold(),
    )
    if not match:
        return None
    amount = int(match.group(1)) if match.group(1).isdigit() else _NUMBER_WORDS[match.group(1)]
    scale = {"day": 1, "week": 7, "month": 30, "year": 365}[match.group(2)]
    return amount * scale


def _mentioned_window_days(text: str) -> int | None:
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month|year)s?\b",
        text.casefold(),
    )
    if not match:
        return None
    amount = int(match.group(1)) if match.group(1).isdigit() else _NUMBER_WORDS[match.group(1)]
    return amount * {"day": 1, "week": 7, "month": 30, "year": 365}[match.group(2)]


def _calendar_day(item: OperandRecordV3, *, prefer_event: bool = False) -> int | None:
    values = (
        (item.event_time, item.observed_at)
        if prefer_event else (item.observed_at, item.event_time)
    )
    for value in values:
        key = _time_key(OperandRecordV3(
            operand_id="date",
            question_id="date",
            subject_key="",
            predicate_key="",
            object_key="",
            object_text="",
            observed_at=value,
        ))
        if key != (0, 0, 0, 0, 0, 0):
            return date(*key[:3]).toordinal()
    return None



def _count_target_terms(question: str) -> set[str]:
    match = re.search(
        r"\bhow many\s+(.+?)\s+"
        r"(?:am|are|did|do|does|had|has|have|is|was|were|will|would)\b",
        question.casefold(),
    )
    return _entity_terms(match.group(1)) if match else set()


def _count_target_heads(question: str) -> set[str]:
    """Return grammatical head alternatives for a count target phrase."""
    match = re.search(
        r"\bhow many\s+(.+?)\s+"
        r"(?:am|are|did|do|does|had|has|have|is|was|were|will|would)\b",
        question.casefold(),
    )
    if not match:
        return set()
    heads: set[str] = set()
    for alternative in re.split(r"\bor\b", match.group(1)):
        ordered = [_token(value) for value in re.findall(r"[\w'-]+", alternative)]
        if not ordered:
            continue
        head = ordered[-1]
        variants = {head}
        if len(head) > 2 and head.endswith("s") and not head.endswith("ss"):
            variants.add(head[:-1])
        heads.update(
            value for value in variants
            if value not in _GENERIC and not value.isdigit()
        )
    return heads


def scalar_snapshot(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    """Return a directly asserted count snapshot for the queried entity."""
    if frame.requested_operation != "count":
        return None
    question_terms = _entity_terms(frame.raw_question)
    target_terms = _count_target_terms(frame.raw_question)
    if not target_terms:
        return None
    target_heads = _count_target_heads(frame.raw_question)
    target_qualifiers = target_terms - target_heads
    initial_window = _initial_window_days(frame.raw_question)
    starts = [
        item for item in operands
        if _asserted(item)
        and bool(_tokens(item.predicate_key) & {"begin", "collect", "collection", "start"})
        and _calendar_day(item, prefer_event=True) is not None
    ]
    candidates = []
    for item in operands:
        if (
            not _asserted(item) or item.quantity is None
            or item.quantity < 0 or _is_money(item)
            or bool({"time", "occurrence"} & _tokens(item.unit))
        ):
            continue
        item_terms = _entity_terms(
            item.predicate_key + " " + item.object_text + " " + item.unit
        )
        # The head noun establishes the measurement unit, not the exact entity.
        # Ground every multiword-target modifier before declaring completion.
        if target_qualifiers and not target_qualifiers.issubset(item_terms):
            continue
        target_unit_match = target_terms & _tokens(item.unit)
        if item.unit and not target_unit_match:
            continue
        if not item.unit and not (target_terms & item_terms):
            continue
        window_match = 0
        if initial_window is not None:
            end_day = _calendar_day(item)
            described_window = _mentioned_window_days(
                item.predicate_key + " " + item.object_text + " " + (item.event_time or "")
            )
            if described_window is not None and abs(described_window - initial_window) <= max(7, int(initial_window * 0.2)):
                window_match = 1
            for start in starts:
                if window_match:
                    break
                start_day = _calendar_day(start, prefer_event=True)
                same_source = bool(set(item.source_turn_ids) & set(start.source_turn_ids))
                if (
                    same_source and end_day is not None and start_day is not None
                    and abs((end_day - start_day) - initial_window) <= max(7, int(initial_window * 0.2))
                ):
                    window_match = 1
                    break
            if not window_match:
                continue
        entity_match = len(question_terms & item_terms)
        unit_match = len(question_terms & _tokens(item.unit))
        if entity_match == 0 or (unit_match == 0 and entity_match < 2):
            continue
        relation_bonus = int(bool(
            _tokens(item.predicate_key)
            & {
                "count", "have", "include", "own", "total",
                "collect", "collection", "participate",
            }
        ))
        temporal_bonus = len(
            {"first", "initial", "month", "week", "year", "current", "currently"}
            & question_terms & item_terms
        )
        score = (
            5 * unit_match + 3 * entity_match + 2 * relation_bonus + 3 * temporal_bonus
            + 12 * window_match
            + query_overlap(frame, item.retrieval_text) + item.confidence
        )
        candidates.append((score, _time_key(item), item))
    if not candidates:
        return None
    _score, base_time, item = max(
        candidates, key=lambda row: (row[1], row[0], row[2].operand_id)
    )
    update_actions = {
        "acquire", "add", "buy", "download", "inherit", "purchase", "receive"
    }
    removal_actions = {"donate", "lose", "remove", "sell"}
    deltas: list[OperandRecordV3] = []
    for candidate in operands:
        if initial_window is not None:
            continue
        if candidate.operand_id == item.operand_id:
            continue
        if candidate.modality in {"planned", "possible", "hypothetical"}:
            continue
        if _time_key(candidate) <= base_time:
            continue
        candidate_terms = _entity_terms(
            candidate.predicate_key + " " + candidate.object_text + " "
            + candidate.context_key + " " + candidate.retrieval_text
        )
        action_terms = _tokens(candidate.predicate_key)
        is_removal = bool(
            action_terms & removal_actions
            or candidate.state_op in {"remove", "cancel", "retract"}
            or candidate.polarity == "negative"
        )
        if not is_removal and not (action_terms & update_actions):
            continue
        if target_heads and not (target_heads & candidate_terms):
            continue
        required_overlap = 1 if len(target_terms) <= 1 else 2
        if len(target_terms & candidate_terms) < required_overlap:
            continue
        deltas.append(candidate)
    value = float(item.quantity or 0) + sum(
        (-1.0 if (
            _tokens(candidate.predicate_key) & removal_actions
            or candidate.state_op in {"remove", "cancel", "retract"}
            or candidate.polarity == "negative"
        ) else 1.0)
        * (float(candidate.quantity) if candidate.quantity is not None else 1.0)
        for candidate in deltas
    )
    return {
        "operation": "scalar_snapshot",
        "value": value,
        "unit": item.unit,
        "operand_ids": [item.operand_id, *[value.operand_id for value in deltas]],
        "source_turn_ids": list(dict.fromkeys([
            *item.source_turn_ids,
            *[source for value in deltas for source in value.source_turn_ids],
        ])),
        "delta_operand_ids": [value.operand_id for value in deltas],
        "complete": True,
    }


def dimensional_quantity_total(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
) -> dict[str, Any] | None:
    """Sum a closed set of same-dimension acquisitions without topic labels."""
    question = frame.raw_question.casefold()
    if not re.search(r"\b(?:total|combined|aggregate|sum|altogether|in total)\b", question):
        return None
    dimensions = {
        "mass": ({"weight", "weigh"}, {"pound", "lb", "lbs", "kg", "kilogram", "gram", "g"}),
        "distance": ({"distance", "length"}, {"mile", "miles", "km", "kilometer", "meter", "metre"}),
        "duration": ({"duration", "time"}, {"second", "minute", "hour", "day", "week", "month", "year"}),
    }
    selected_dimension = next((
        (name, units) for name, (query_words, units) in dimensions.items()
        if query_words & _tokens(question)
    ), None)
    if selected_dimension is None:
        return None
    dimension, units = selected_dimension
    query_actions = _tokens(question) & _ACTION_WORDS
    acquisition_actions = {"acquire", "buy", "get", "purchase", "receive"}
    if query_actions & acquisition_actions:
        query_actions |= acquisition_actions
    candidates = []
    for item in operands:
        if not _asserted(item) or item.quantity is None:
            continue
        unit_terms = _tokens(item.unit)
        if not (unit_terms & units):
            continue
        predicate_terms = _tokens(item.predicate_key)
        if predicate_terms & {"cost", "pay", "price", "spend"}:
            continue
        if re.search(r"\$|\b(?:USD|EUR|GBP)\b", item.object_text, re.IGNORECASE):
            continue
        if query_actions and not (predicate_terms & query_actions):
            continue
        candidates.append(item)
    by_subject: dict[str, list[OperandRecordV3]] = defaultdict(list)
    for item in candidates:
        by_subject[item.subject_key].append(item)
    candidates = max(
        by_subject.values(),
        key=lambda rows: (len(rows), max((_time_key(item) for item in rows), default=(0, 0, 0, 0, 0, 0))),
        default=[],
    )
    if len(candidates) < 2:
        return None
    canonical_units = {
        "mass": {"pound": 1.0, "lb": 1.0, "lbs": 1.0, "kg": 2.2046226218, "kilogram": 2.2046226218, "gram": 0.0022046226, "g": 0.0022046226},
        "distance": {"mile": 1.0, "miles": 1.0, "km": 0.621371, "kilometer": 0.621371, "meter": 0.000621371, "metre": 0.000621371},
        "duration": {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0, "week": 604800.0, "month": 2592000.0, "year": 31536000.0},
    }[dimension]
    def normalized(item: OperandRecordV3) -> float:
        unit = next(value for value in _tokens(item.unit) if value in canonical_units)
        return float(item.quantity or 0) * canonical_units[unit]
    return {
        "operation": "dimensional_quantity_total",
        "value": sum(normalized(item) for item in candidates),
        "unit": next((item.unit for item in candidates if item.unit), ""),
        "dimension": dimension,
        "operand_ids": [item.operand_id for item in candidates],
        "source_turn_ids": list(dict.fromkeys(
            source for item in candidates for source in item.source_turn_ids
        )),
        "complete": True,
    }


def partitioned_scalar_total(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    query_overlap: Callable[[QueryFrame, str], float],
    turns: list[Any] | None = None,
) -> dict[str, Any] | None:
    question = frame.raw_question.casefold()
    if not re.search(r"\b(?:total\s+(?:number|count|amount)|in\s+total|altogether)\b", question):
        return None
    query_terms = _tokens(question) - _GENERIC
    numeric = [
        item for item in operands
        if _asserted(item) and item.quantity is not None and not _is_money(item)
    ]
    if not numeric:
        return None
    # One lossless statement may yield both an entity projection ("finished X")
    # and a scalar projection ("played N hours in X"). They represent one
    # quantity mention when provenance, value, unit, and cross object/context
    # identity agree; independent equal-valued items in the same turn remain separate.
    deduped_numeric: list[OperandRecordV3] = []
    for item in numeric:
        item_sources = set(item.source_turn_ids)
        duplicate_index = next((
            index for index, old in enumerate(deduped_numeric)
            if item_sources and item_sources == set(old.source_turn_ids)
            and item.quantity == old.quantity
            and _tokens(item.unit) == _tokens(old.unit)
            and (
                _entity_terms(item.object_text) & _entity_terms(old.context_key)
                or _entity_terms(old.object_text) & _entity_terms(item.context_key)
            )
        ), None)
        if duplicate_index is None:
            deduped_numeric.append(item)
            continue
        old = deduped_numeric[duplicate_index]
        item_is_scalar = bool(re.fullmatch(
            r"\s*[+-]?(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")(?:\s+\w+)?\s*",
            item.object_text.casefold(),
        ))
        old_is_scalar = bool(re.fullmatch(
            r"\s*[+-]?(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")(?:\s+\w+)?\s*",
            old.object_text.casefold(),
        ))
        if (old_is_scalar, item.confidence, item.operand_id) > (
            item_is_scalar, old.confidence, old.operand_id
        ):
            deduped_numeric[duplicate_index] = item
    numeric = deduped_numeric
    unit_terms = set().union(*(_tokens(item.unit) for item in numeric if item.unit))
    semantic_targets = query_terms - unit_terms - {"count", "number"}
    turn_text_by_session: dict[str, list[str]] = defaultdict(list)
    for turn in turns or []:
        turn_text_by_session[getattr(turn, "session_id", "")].append(
            str(getattr(turn, "text", ""))
        )
    peers_by_session: dict[str, list[OperandRecordV3]] = defaultdict(list)
    peers_by_frame: dict[str, list[OperandRecordV3]] = defaultdict(list)
    for peer in operands:
        for session_id in peer.session_ids:
            peers_by_session[session_id].append(peer)
        if peer.event_frame_id:
            peers_by_frame[peer.event_frame_id].append(peer)
    scoped_numeric = []
    for item in numeric:
        if item.unit and not (_tokens(item.unit) & query_terms):
            continue
        peers = [
            peer for session_id in item.session_ids
            for peer in peers_by_session.get(session_id, [])
        ]
        enriched_terms = _tokens(" ".join(
            [item.retrieval_text, *[peer.retrieval_text for peer in peers], *[text for session_id in item.session_ids for text in turn_text_by_session.get(session_id, [])]]
        ))
        if semantic_targets and not (semantic_targets & set(enriched_terms)):
            continue
        scoped_numeric.append(item)
    numeric = scoped_numeric

    def context(item: OperandRecordV3) -> str | None:
        if item.event_frame_id:
            return f"frame:{item.event_frame_id}"
        scalar_frame_peer = next((
            peer for session_id in item.session_ids
            for peer in peers_by_session.get(session_id, [])
            if peer.event_frame_id and peer.quantity == item.quantity and peer.unit == item.unit
        ), None)
        if scalar_frame_peer is not None:
            return f"frame:{scalar_frame_peer.event_frame_id}"
        object_terms = _entity_terms(item.object_text) - query_terms
        is_scalar_object = bool(re.fullmatch(
            r"\s*[+-]?(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")\s*",
            item.object_text.casefold(),
        ))
        if not is_scalar_object and object_terms:
            return " ".join(sorted(object_terms))
        frame_peers = (
            peers_by_frame.get(item.event_frame_id, [])
            if item.event_frame_id else []
        )
        session_peers = [
            peer for session_id in item.session_ids
            for peer in peers_by_session.get(session_id, [])
        ]
        peers = [
            peer for peer in [*frame_peers, *session_peers]
            if peer.operand_id != item.operand_id
            and peer.subject_key == item.subject_key
            and _entity_terms(peer.object_text)
            and not re.fullmatch(r"\s*\d+(?:\.\d+)?(?:\s+\w+)?\s*", peer.object_text)
        ]
        if not peers:
            return None
        peer = max(peers, key=lambda value: (
            int(bool(item.event_frame_id and value.event_frame_id == item.event_frame_id)),
            len(semantic_targets & _tokens(value.retrieval_text)),
            query_overlap(frame, value.retrieval_text), value.confidence, value.operand_id
        ))
        terms = _entity_terms(peer.object_text) - query_terms
        return " ".join(sorted(terms)) if terms else None

    latest: dict[str, OperandRecordV3] = {}
    for item in numeric:
        bucket = context(item)
        if not bucket:
            continue
        old = latest.get(bucket)
        if old is None or (_time_key(item), item.confidence, item.operand_id) > (
            _time_key(old), old.confidence, old.operand_id
        ):
            latest[bucket] = item
    if len(latest) < 2:
        return None
    parts = [
        {"context": bucket, "value": float(item.quantity or 0), "operand_id": item.operand_id}
        for bucket, item in sorted(latest.items())
    ]
    return {
        "operation": "partitioned_scalar_total",
        "value": sum(part["value"] for part in parts),
        "unit": next((item.unit for item in latest.values() if item.unit), ""),
        "parts": parts,
        "operand_ids": [item.operand_id for item in latest.values()],
        "source_turn_ids": list(dict.fromkeys(
            source for item in latest.values() for source in item.source_turn_ids
        )),
        "complete": True,
    }


def arithmetic_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    event_frames: list[EventFrameV3],
    query_overlap: Callable[[QueryFrame, str], float],
    turns: list[Any] | None = None,
) -> dict[str, Any] | None:
    return (
        ratio_percent(frame, operands, event_frames, query_overlap)
        or per_item_amount(frame, operands, query_overlap)
        or total_money(frame, operands, query_overlap)
        or dimensional_quantity_total(frame, operands)
        or partitioned_scalar_total(frame, operands, query_overlap, turns)
        or scalar_snapshot(frame, operands, query_overlap)
        or event_occurrence_count(frame, operands, event_frames, query_overlap, turns)
    )
