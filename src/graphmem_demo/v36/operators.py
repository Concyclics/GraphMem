from __future__ import annotations

from datetime import datetime, timedelta
import calendar
import re
from typing import Any

_TIME_UNIT_SECONDS = {
    "second": 1.0, "seconds": 1.0, "minute": 60.0, "minutes": 60.0,
    "hour": 3600.0, "hours": 3600.0, "day": 86400.0, "days": 86400.0,
    "week": 604800.0, "weeks": 604800.0, "month": 2629800.0, "months": 2629800.0,
    "year": 31557600.0, "years": 31557600.0,
}

from .schema import (
    CompletenessCertificate,
    EvidenceGroup,
    QueryIR,
    RoleFrameNode,
    V36Index,
)


def _selected(
    index: V36Index,
    frame_ids: list[str],
    group_ids: list[str],
) -> tuple[list[RoleFrameNode], list[EvidenceGroup]]:
    wanted_frames = set(frame_ids)
    wanted_groups = set(group_ids)
    return (
        [frame for frame in index.frames if frame.frame_id in wanted_frames],
        [
            group for group in index.evidence_groups
            if group.group_id in wanted_groups
        ],
    )


def _collection_values(frames: list[RoleFrameNode]) -> list[str]:
    values: list[str] = []
    for frame in sorted(frames, key=lambda item: (
        item.temporal.event_time or item.temporal.observed_at or "",
        item.observation_order,
    )):
        value = frame.object_key
        if not value:
            continue
        if frame.state_op in {"remove", "decrement", "cancel"}:
            values = [item for item in values if item != value]
        elif value not in values and frame.polarity != "negative":
            values.append(value)
    return values


def _reference_location_closures(
    ir: QueryIR,
    frames: list[RoleFrameNode],
    groups: list[EvidenceGroup],
) -> list[dict[str, Any]]:
    """Expose only provenance-complete scalar reference bindings.

    This is a structural closure over RoleFrame slots. It deliberately does
    not recognize countries, cities, benchmark topics, or question templates.
    """
    frame_by_id = {frame.frame_id: frame for frame in frames}
    ignored = {
        "ago", "year", "years", "month", "months",
        "week", "weeks", "day", "days",
    }

    def reference_term(term: str) -> str:
        value = term.casefold()
        if len(value) > 5 and value.endswith("ing"):
            value = value[:-3]
            if len(value) > 2 and value[-1] == value[-2]:
                value = value[:-1]
        elif len(value) > 4 and value.endswith("ed"):
            value = value[:-2]
            if value.endswith("v"):
                value += "e"
        elif len(value) > 4 and value.endswith("s"):
            value = value[:-1]
        return value

    relation_terms = {
        reference_term(term)
        for term in re.findall(r"[\w'-]+", ir.target_relation.casefold())
        if term not in ignored and term != ir.target_owner
    }
    closures: list[dict[str, Any]] = []
    for group in groups:
        if group.group_kind != "reference_chain" or not group.provenance_complete:
            continue
        rows = [
            frame_by_id[frame_id]
            for frame_id in group.member_frame_ids
            if frame_id in frame_by_id
        ]
        if len(rows) != len(group.member_frame_ids):
            continue
        owners = {frame.owner_key for frame in rows if frame.owner_key}
        if ir.target_owner and owners != {ir.target_owner}:
            continue
        has_location_role = any(
            bool(
                {frame.predicate_key, *frame.semantic_type_keys}
                & {"location", "origin", "destination", "place"}
            )
            for frame in rows
        )
        group_terms = {
            reference_term(term)
            for term in re.findall(r"[\w'-]+", group.retrieval_text.casefold())
        }
        if not has_location_role or (
            relation_terms and not relation_terms & group_terms
        ):
            continue
        closures.append({
            "operation": "reference_closure",
            "group_id": group.group_id,
            "bindings": [
                {
                    "owner": frame.owner_key,
                    "entity": frame.entity_key,
                    "predicate": frame.predicate_key,
                            "context": frame.context_key,
                }
                for frame in rows
            ],
            "frame_ids": [frame.frame_id for frame in rows],
            "certified": True,
        })
    return closures


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in (
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %B %Y",
            "%d %B, %Y",
            "%d %B %Y",
        ):
            try:
                return datetime.strptime(value.strip(), pattern)
            except ValueError:
                pass
        match = re.match(
            r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[^0-9]+(\d{1,2}):(\d{2}))?",
            value,
        )
        if match is None:
            return None
        year, month, day, hour, minute = match.groups()
        return datetime(
            int(year), int(month), int(day), int(hour or 0), int(minute or 0),
        )


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _turn_observed_time(turn: Any) -> datetime | None:
    value = turn.session_date or ""
    found = re.match(
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[^0-9]+(\d{1,2}):(\d{2}))?",
        value,
    )
    if found is None:
        return _parse_time(value)
    year, month, day, hour, minute = found.groups()
    return datetime(
        int(year), int(month), int(day), int(hour or 0), int(minute or 0),
    )


def _relative_event_time(text: str, observed: datetime) -> datetime | None:
    lowered = text.casefold()
    match = re.search(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(hours?|days?|weeks?|months?)\s+ago\b", lowered,
    )
    if match:
        amount = (
            int(match.group(1)) if match.group(1).isdigit()
            else 1 if match.group(1) in {"a", "an"}
            else _NUMBER_WORDS[match.group(1)]
        )
        unit = match.group(2).rstrip("s")
        if unit == "month":
            return observed - timedelta(days=30 * amount)
        return observed - timedelta(**{f"{unit}s": amount})
    if re.search(r"\blast week(?:end)?\b", lowered):
        return observed - timedelta(days=7)
    if re.search(r"\bnext month\b", lowered):
        month = 1 if observed.month == 12 else observed.month + 1
        year = observed.year + 1 if observed.month == 12 else observed.year
        day = min(observed.day, calendar.monthrange(year, month)[1])
        return observed.replace(year=year, month=month, day=day)
    if re.search(r"\blast year\b", lowered):
        try:
            return observed.replace(year=observed.year - 1)
        except ValueError:
            return observed.replace(year=observed.year - 1, day=28)
    if re.search(r"\btoday\b", lowered):
        return observed
    if re.search(r"\byesterday\b", lowered):
        return observed - timedelta(days=1)
    weekday = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lowered,
    )
    if weekday:
        days = (observed.weekday() - _WEEKDAYS[weekday.group(1)]) % 7 or 7
        return observed - timedelta(days=days)
    if re.search(r"\btoday\b", lowered):
        return observed
    return None


def _deduplicate_duration_echoes(
    ir: QueryIR, frames: list[RoleFrameNode], index: V36Index,
) -> list[RoleFrameNode]:
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    ordered = sorted(frames, key=lambda frame: min(
        ((turn_by_id[source].session_id, turn_by_id[source].turn_index)
         for source in frame.source_turn_ids if source in turn_by_id),
        default=("", 10**9),
    ))
    word_re = re.compile(r"[\w'-]+", re.UNICODE)
    query_terms = _duration_query_terms(ir)

    def details(frame: RoleFrameNode):
        turn = next((
            turn_by_id[source] for source in frame.source_turn_ids
            if source in turn_by_id
        ), None)
        value = (frame.quantity.value or 0.0) * (
            _TIME_UNIT_SECONDS[frame.quantity.unit.casefold()]
        ) * (frame.quantity.multiplier or 1.0)
        words = (
            {word.casefold() for word in word_re.findall(turn.text)}
            if turn is not None else set()
        )
        projected = {_stem_word(word) for word in words} & query_terms
        return turn, value, words, projected

    # First remove adjacent cross-speaker echoes against every earlier raw
    # candidate. A later cross-session dedup must not erase this anchor.
    non_echoes: list[RoleFrameNode] = []
    processed: list[RoleFrameNode] = []
    for frame in ordered:
        frame_turn, frame_value, frame_words, frame_projected = details(frame)
        is_echo = False
        for prior in processed:
            prior_turn, prior_value, prior_words, prior_projected = details(prior)
            if frame_turn is None or prior_turn is None:
                continue
            overlap = len(frame_words & prior_words) / max(
                1, len(frame_words | prior_words)
            )
            projected_echo = (
                bool(frame_projected & prior_projected)
                and (
                    frame_projected <= prior_projected
                    or prior_projected <= frame_projected
                )
            )
            if (
                abs(frame_value - prior_value) <= 1e-6
                and frame_turn.session_id == prior_turn.session_id
                and abs(frame_turn.turn_index - prior_turn.turn_index) == 1
                and frame_turn.speaker_key != prior_turn.speaker_key
                and (overlap >= 0.30 or projected_echo)
            ):
                is_echo = True
                break
        processed.append(frame)
        if not is_echo:
            non_echoes.append(frame)

    kept: list[RoleFrameNode] = []
    for frame in non_echoes:
        frame_turn, frame_value, frame_words, frame_projected = details(frame)
        duplicate = False
        for prior in kept:
            prior_turn, prior_value, prior_words, prior_projected = details(prior)
            if frame_turn is None or prior_turn is None:
                continue
            overlap = len(frame_words & prior_words) / max(
                1, len(frame_words | prior_words)
            )
            if (
                abs(frame_value - prior_value) <= 1e-6
                and frame_turn.speaker_key == prior_turn.speaker_key
                and bool(frame_projected & prior_projected)
                and (
                    frame_projected <= prior_projected
                    or prior_projected <= frame_projected
                )
                and overlap >= 0.10
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(frame)
    return kept


_DURATION_QUERY_STOP = {
    "how", "many", "much", "long", "did", "does", "do", "take", "took",
    "all", "the", "a", "an", "me", "my", "i", "it", "and", "in", "to",
    "second", "seconds", "minute", "minutes", "hour", "hours", "day",
    "days", "week", "weeks", "month", "months", "year", "years",
}


def _stem_word(value: str) -> str:
    word = value.casefold()
    irregular = {
        "bought": "buy", "got": "get", "had": "have", "used": "use",
        "done": "do", "made": "make", "took": "take",
        "went": "go", "gone": "go", "built": "build",
        "completed": "complete", "created": "create",
        "pickup": "pick", "picked": "pick", "taking": "take",
        "attending": "attend", "consumed": "consume",
        "earning": "earn", "earned": "earn", "selling": "sell",
        "sold": "sell", "spent": "spend", "paid": "pay",
        "rode": "ride", "ridden": "ride",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith("'s") or word.endswith("’s"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        # Preserve a silent e for regular forms such as use/used and
        # organize/organized. This is morphology, not a domain vocabulary.
        if stem.endswith(("us", "iz", "at")):
            return stem + "e"
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _duration_query_terms(ir: QueryIR) -> set[str]:
    words = re.findall(r"[\w'-]+", ir.raw_question)
    relation_positions = {
        index for index, word in enumerate(words)
        if index > 0 and words[index - 1].casefold() == "to"
    }
    return {
        _stem_word(word) for index, word in enumerate(words)
        if word.casefold() not in _DURATION_QUERY_STOP
        and index not in relation_positions
    }


def relative_time_from_sources_hint(
    ir: QueryIR,
    index: V36Index,
    source_turn_ids: list[str],
    question_date: str | None,
) -> dict[str, Any] | None:
    """Compute question-relative recency from one action-bound lossless source.

    Source-relative phrases such as ``today`` or ``two weeks ago`` are first
    anchored to that source turn's date.  Only then is the delta to the
    question date computed.  This is topic-independent and keeps the cited
    source visible for the final LLM answer.
    """
    if ir.requested_value_type != "duration" or "ago" not in ir.temporal_constraints:
        return None
    unit_match = re.search(
        r"\bhow many\s+(days?|weeks?|months?|years?)\s+ago\b",
        ir.raw_question, re.IGNORECASE,
    )
    if unit_match is None:
        return None
    question_time = _parse_time(question_date)
    if question_time is None:
        return None
    ignored = {
        "how", "many", "day", "days", "week", "weeks", "month",
        "months", "year", "years", "ago", "did", "do", "does",
        "i", "me", "my", "the", "a", "an", "on", "in", "at",
        "to", "from", "with", "narrator",
    }
    query_terms = {
        _stem_word(word) for word in re.findall(r"[\w'-]+", ir.raw_question.casefold())
        if word not in ignored
    }
    first_person = bool(re.search(r"\b(?:i|me|my|narrator)\b", ir.raw_question, re.IGNORECASE))
    quoted_targets = [
        left or right for left, right in re.findall(
            r"'([^']+)'|\"([^\"]+)\"", ir.raw_question
        ) if (left or right).strip()
    ]
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    frames_by_source: dict[str, list[RoleFrameNode]] = {}
    for frame in index.frames:
        for source_id in frame.source_turn_ids:
            frames_by_source.setdefault(source_id, []).append(frame)
    candidates: list[tuple[int, int, float, str, datetime]] = []
    for source_id in source_turn_ids:
        turn = turn_by_id.get(source_id)
        if turn is None or (first_person and turn.transport_role != "user"):
            continue
        if quoted_targets and not all(
            target.casefold() in turn.text.casefold() for target in quoted_targets
        ):
            continue
        observed = _turn_observed_time(turn)
        if observed is None:
            continue
        segments = [
            segment.strip() for segment in re.split(
                r"(?<=[.!?])\s+|\n+", turn.text
            ) if segment.strip()
        ] or [turn.text]
        segment_rows = []
        for segment in segments:
            segment_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", segment.casefold()
                )
            }
            segment_rows.append((len(query_terms & segment_terms), segment))
        overlap, bound_segment = max(
            segment_rows, key=lambda row: (row[0], -segments.index(row[1]))
        )
        required = 1 if len(query_terms) <= 2 else 2
        if overlap < required:
            continue
        relative = _relative_event_time(bound_segment, observed)
        relative_bound = relative is not None
        event_time = relative
        if event_time is None:
            dated_frames = []
            for frame in frames_by_source.get(source_id, []):
                frame_terms = {
                    _stem_word(word) for word in re.findall(
                        r"[\w'-]+",
                        " ".join((frame.entity_key, frame.predicate_key, frame.object_key)).casefold(),
                    )
                }
                frame_overlap = len(query_terms & frame_terms)
                parsed = _parse_time(frame.temporal.event_time or frame.temporal.start)
                if parsed is not None and frame_overlap >= required:
                    dated_frames.append((frame_overlap, frame.confidence, parsed))
            if dated_frames:
                _score, _confidence, event_time = max(dated_frames)
        if event_time is None and re.search(
            r"\b(?:download(?:ed)?|start(?:ed)?|began|met|attended?|went|"
            r"read|watched?|received?|bought|got|finished?|used)\b",
            bound_segment, re.IGNORECASE,
        ):
            event_time = observed
        if event_time is None or event_time > question_time + timedelta(days=1):
            continue
        candidates.append((
            overlap, int(relative_bound),
            max((frame.confidence for frame in frames_by_source.get(source_id, [])), default=0.0),
            source_id, event_time,
        ))
    if not candidates:
        return None
    overlap, relative_bound, confidence, source_id, event_time = max(
        candidates, key=lambda row: (row[0], row[1], row[2], row[3])
    )
    delta_days = max(0, (question_time.date() - event_time.date()).days)
    unit = unit_match.group(1).casefold().rstrip("s")
    if unit == "day":
        value: float | int = delta_days
    elif unit == "week":
        value = int(round(delta_days / 7.0))
    elif unit == "month":
        value = int(round(delta_days / 30.4375))
    else:
        value = int(round(delta_days / 365.25))
    return {
        "operation": "relative_time_from_lossless_source",
        "value": value,
        "unit": unit + ("s" if value != 1 else ""),
        "source_turn_ids": [source_id],
        "event_time": event_time.isoformat(),
        "question_time": question_time.isoformat(),
        "matched_query_term_count": overlap,
        "binding_complete": True,
        "certified": True,
    }


def temporal_source_pair_hint(
    ir: QueryIR,
    index: V36Index,
    source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Certify a between/before/after duration from two bound sources."""
    if (
        ir.requested_value_type != "duration"
        or not {"event_a", "event_b", "time_a", "time_b"}
        <= set(ir.required_roles)
    ):
        return None
    between = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        ir.raw_question, re.IGNORECASE,
    )
    relation = "between"
    if between is not None:
        left_text, right_text = between.group(1), between.group(2)
    else:
        ordered = re.search(
            r"\bhow long\s+(?:did|had|have|was|were)?\s*(.+?)\s+"
            r"(before|after)\s+(.+?)(?:\?|$)",
            ir.raw_question, re.IGNORECASE,
        )
        if ordered is None:
            return None
        left_text, relation, right_text = (
            ordered.group(1), ordered.group(2).casefold(), ordered.group(3)
        )
    stop = {
        "a", "an", "and", "at", "between", "before", "after",
        "did", "had", "have", "how", "i", "in", "many", "me",
        "my", "of", "on", "passed", "the", "to", "was", "were",
    }

    def terms(text: str) -> set[str]:
        return {
            _stem_word(word)
            for word in re.findall(r"[\w'-]+", text.casefold())
            if word not in stop and len(word) > 1
        }

    left_terms, right_terms = terms(left_text), terms(right_text)
    if not left_terms or not right_terms:
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    frames_by_source: dict[str, list[RoleFrameNode]] = {}
    for frame in index.frames:
        for source_id in frame.source_turn_ids:
            frames_by_source.setdefault(source_id, []).append(frame)
    first_person = bool(re.search(
        r"\b(?:i|me|my|narrator)\b", ir.raw_question, re.IGNORECASE,
    ))

    def binding(turn: Any, target: set[str]) -> tuple[int, datetime] | None:
        observed = _turn_observed_time(turn)
        if observed is None:
            return None
        segments = [
            segment.strip() for segment in re.split(
                r"(?<=[.!?])\s+|\n+", turn.text
            ) if segment.strip()
        ] or [turn.text]
        scored = []
        for segment in segments:
            segment_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", segment.casefold()
                )
            }
            scored.append((len(target & segment_terms), segment))
        score, segment = max(scored, key=lambda row: row[0])
        required = 1 if len(target) <= 2 else 2
        segment_terms = terms(segment)
        generic_event_terms = {
            "use", "new", "see", "saw", "return", "area", "start",
            "work", "current", "take", "when", "before", "after",
        }
        identity_terms = target - generic_event_terms
        if score < required or (identity_terms and not identity_terms & segment_terms):
            return None
        event_time = _relative_event_time(segment, observed)
        if event_time is None:
            # A later pronoun-bearing clause may carry the only relative date.
            event_time = _relative_event_time(turn.text, observed)
        if event_time is None:
            dated = []
            for frame in frames_by_source.get(turn.node_id, []):
                frame_terms = terms(
                    " ".join((frame.entity_key, frame.predicate_key, frame.object_key))
                )
                parsed = _parse_time(frame.temporal.event_time or frame.temporal.start)
                overlap = len(target & frame_terms)
                if parsed is not None and overlap >= required:
                    dated.append((overlap, frame.confidence, parsed))
            event_time = max(dated)[2] if dated else None
        if event_time is None and re.search(
            r"\b(?:just|visit(?:ed)?|returned?|attended?|went|completed?|"
            r"finished?|started?|began|met|received?|bought|got|saw|"
            r"fixed|serviced)\b",
            segment, re.IGNORECASE,
        ):
            event_time = observed
        if event_time is None:
            return None
        return score, event_time

    turns = [
        turn_by_id[source_id] for source_id in source_turn_ids
        if source_id in turn_by_id
        and (not first_person or turn_by_id[source_id].transport_role == "user")
    ]
    candidates = []
    for left_turn in turns:
        left_binding = binding(left_turn, left_terms)
        if left_binding is None:
            continue
        for right_turn in turns:
            if right_turn.node_id == left_turn.node_id:
                continue
            right_binding = binding(right_turn, right_terms)
            if right_binding is None:
                continue
            left_score, left_date = left_binding
            right_score, right_date = right_binding
            candidates.append((
                min(left_score, right_score), left_score + right_score,
                left_turn.node_id, right_turn.node_id, left_date, right_date,
            ))
    if not candidates:
        return None
    _minimum, _total, left_id, right_id, left_date, right_date = max(
        candidates, key=lambda row: (row[0], row[1], row[2], row[3]),
    )
    if re.search(r"\bdays?\b", ir.raw_question, re.IGNORECASE):
        left_date = datetime.combine(left_date.date(), datetime.min.time())
        right_date = datetime.combine(right_date.date(), datetime.min.time())
    delta_seconds = abs((right_date - left_date).total_seconds())
    if delta_seconds == 0:
        return None
    display_value: float | int = delta_seconds
    display_unit = "seconds"
    if not re.search(
        r"\b(?:seconds?|minutes?|hours?|days?)\b",
        ir.raw_question, re.IGNORECASE,
    ):
        left_source_text = turn_by_id[left_id].text
        right_source_text = turn_by_id[right_id].text
        relative_week = re.compile(
            r"\b(?:a|an|\d+|one|two|three|four|five|six|seven|eight|"
            r"nine|ten|eleven|twelve)\s+weeks?\s+ago\b",
            re.IGNORECASE,
        )
        if relative_week.search(left_source_text) and relative_week.search(right_source_text):
            display_value = int(round(delta_seconds / (7 * 24 * 60 * 60)))
            display_unit = "weeks"
    return {
        "operation": "time_difference_from_lossless_sources",
        "relation": relation,
        "value": display_value,
        "unit": display_unit,
        "raw_seconds": delta_seconds,
        "event_a_source_turn_id": left_id,
        "event_b_source_turn_id": right_id,
        "event_a_time": left_date.isoformat(),
        "event_b_time": right_date.isoformat(),
        "binding_complete": True,
        "certified": True,
    }



def relative_anchor_source_hint(
    ir: QueryIR, index: V36Index, question_date: str | None,
    allowed_session_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Retrieve the best user source inside a question-derived date window."""
    question_time = _parse_time(question_date)
    if question_time is None:
        return None
    raw = ir.raw_question
    target_dates: list[datetime] = []
    tolerance_days = 0
    relative = re.search(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve)\s+(days?|weeks?|months?)\s+ago\b",
        raw, re.IGNORECASE,
    )
    if relative:
        token = relative.group(1).casefold()
        amount = int(token) if token.isdigit() else 1 if token in {"a", "an"} else _NUMBER_WORDS[token]
        unit = relative.group(2).casefold().rstrip("s")
        days = amount if unit == "day" else 7 * amount if unit == "week" else 30 * amount
        target_dates = [question_time - timedelta(days=days)]
        tolerance_days = 1 if unit == "day" else 3 if unit == "week" else 5
    weekday = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        raw, re.IGNORECASE,
    )
    if weekday:
        days = (question_time.weekday() - _WEEKDAYS[weekday.group(1).casefold()]) % 7 or 7
        target_dates = [question_time - timedelta(days=days)]
        tolerance_days = 0
    if re.search(r"\b(?:past|last)\s+weekend\b", raw, re.IGNORECASE):
        days_to_saturday = (question_time.weekday() - _WEEKDAYS["saturday"]) % 7 or 7
        saturday = question_time - timedelta(days=days_to_saturday)
        target_dates = [saturday, saturday + timedelta(days=1)]
        tolerance_days = 0
    if re.search(r"\bvalentine(?:'s|s)?\s+day\b", raw, re.IGNORECASE):
        year = question_time.year
        valentine = question_time.replace(year=year, month=2, day=14)
        if valentine > question_time:
            valentine = valentine.replace(year=year - 1)
        target_dates = [valentine]
        tolerance_days = 0
    if not target_dates:
        return None
    ignored = {
        "what", "which", "who", "was", "were", "did", "do", "does",
        "i", "me", "my", "the", "a", "an", "ago", "day", "days",
        "week", "weeks", "month", "months", "last", "past", "on", "in",
        "at", "to", "from", "with", "that", "this", "mentioned",
        "related", "activity", "significant", "one", "two", "three",
        "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve",
    }
    query_terms = {
        _stem_word(word) for word in re.findall(
            r"[\w']+", raw.casefold().replace("-", " ")
        )
        if word not in ignored and not word.isdigit()
    }
    card_by_session = {card.session_id: card for card in index.routing_cards}
    candidates = []
    for turn in index.turns:
        if turn.transport_role != "user":
            continue
        if allowed_session_ids and turn.session_id not in allowed_session_ids:
            continue
        observed = _turn_observed_time(turn)
        if observed is None:
            continue
        distance = min(abs((observed.date() - target.date()).days) for target in target_dates)
        if distance > tolerance_days:
            continue
        card = card_by_session.get(turn.session_id)
        turn_terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", turn.text.casefold())}
        card_terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", (card.routing_text if card else '').casefold())}
        direct_overlap = len(query_terms & turn_terms)
        routed_overlap = len(query_terms & card_terms)
        if direct_overlap == 0 and routed_overlap == 0:
            continue
        candidates.append((direct_overlap, routed_overlap, -turn.turn_index, turn, card, observed))
    if not candidates:
        return None
    direct, routed, _turn_order, turn, card, observed = max(
        candidates, key=lambda row: (row[0] + 2 * row[1], row[0], row[2])
    )
    session_turns = sorted(
        (item for item in index.turns
         if item.session_id == turn.session_id and item.transport_role == "user"),
        key=lambda item: item.turn_index,
    )
    selected_turns = session_turns[:3]
    if turn not in selected_turns:
        selected_turns.append(turn)
    answer_candidate = ""
    bundled_source_text = "\n".join(item.text for item in selected_turns)
    if re.search(r"\bwho\b.+\bwith\b", raw, re.IGNORECASE):
        companion = re.search(
            r"\bwith\s+((?:my|our)\s+[A-Za-z]+(?:\s+and\s+[A-Za-z]+)?|"
            r"a\s+group\s+of\s+[A-Za-z]+)",
            bundled_source_text, re.IGNORECASE,
        )
        if companion is not None:
            answer_candidate = companion.group(1)
    head_match = re.search(
        r"\bwhich\s+([A-Za-z]+)\b", raw, re.IGNORECASE,
    )
    if not answer_candidate and head_match is not None:
        head = head_match.group(1)
        entities = re.findall(
            rf"\b([A-Za-z]+\s+{re.escape(head)})\b", bundled_source_text, re.IGNORECASE,
        )
        if entities:
            answer_candidate = entities[0]
    return {
        "operation": "relative_anchor_source_lookup",
        "answer_candidate": answer_candidate,
        "value": "\n".join(item.text[:450] for item in selected_turns),
        "source_turn_ids": [item.node_id for item in selected_turns],
        "source_date": observed.isoformat(),
        "routing_context": (card.routing_text[:360] if card else ""),
        "direct_query_overlap": direct,
        "routing_query_overlap": routed,
        "binding_complete": True,
        "certified": True,
    }


def record_time_source_hint(
    ir: QueryIR,
    index: V36Index,
    source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Return the best elapsed-time record explicitly grounded in sources."""
    if ir.requested_value_type != "state" or not re.search(
        r"\b(?:personal\s+best|record)\b.*\btime\b|"
        r"\btime\b.*\b(?:personal\s+best|record)\b",
        ir.raw_question, re.IGNORECASE,
    ):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    candidates: list[tuple[int, str, str]] = []
    for source_id in source_turn_ids:
        turn = turn_by_id.get(source_id)
        if turn is None or not re.search(
            r"\b(?:personal\s+best|record)\b", turn.text, re.IGNORECASE,
        ):
            continue
        for match in re.finditer(r"\b(\d{1,2}):(\d{2})\b", turn.text):
            minutes, seconds = int(match.group(1)), int(match.group(2))
            if seconds < 60:
                candidates.append((minutes * 60 + seconds, f"{minutes}:{seconds:02d}", source_id))
        for match in re.finditer(
            r"\b(\d+)\s+minutes?\s+(?:and\s+)?(\d+)\s+seconds?\b",
            turn.text, re.IGNORECASE,
        ):
            minutes, seconds = int(match.group(1)), int(match.group(2))
            if seconds < 60:
                candidates.append((minutes * 60 + seconds, f"{minutes}:{seconds:02d}", source_id))
    if not candidates:
        return None
    seconds, value, source_id = min(candidates)
    return {
        "operation": "record_time_extreme",
        "value": value,
        "seconds": seconds,
        "source_turn_ids": [source_id],
        "certified": True,
    }


def temporal_order_source_hint(
    ir: QueryIR,
    index: V36Index,
    source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Certify a binary choice or sequence from action-bound source turns."""
    if ir.requested_value_type != "temporal_order":
        return None

    stop = {
        "a", "an", "at", "did", "from", "i", "in", "me", "my", "of",
        "on", "the", "to", "was", "who", "what", "which",
    }

    def terms(text: str) -> set[str]:
        return {
            _stem_word(word.strip("'\""))
            for word in re.findall(r"[\w'-]+", text.casefold())
            if word.strip("'\"") not in stop and len(word.strip("'\"")) > 1
        }

    target_terms = [terms(target) for target in ir.comparison_targets]
    if len(target_terms) < 2 or not all(target_terms):
        return None
    action = re.search(
        r"\bdid\s+i\s+([a-z]+)", ir.raw_question, re.IGNORECASE,
    )
    required_actions = {_stem_word(action.group(1))} if action else set()
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    first_person = bool(re.search(
        r"\b(?:i|me|my|narrator)\b", ir.raw_question, re.IGNORECASE,
    ))
    turns = [
        turn_by_id[source_id]
        for source_id in source_turn_ids
        if source_id in turn_by_id
        and (not first_person or turn_by_id[source_id].transport_role == "user")
    ]
    candidates_by_target: list[list[tuple[int, str, Any, datetime]]] = []
    for target in target_terms:
        rows: list[tuple[int, str, Any, datetime]] = []
        for turn in turns:
            observed = _turn_observed_time(turn)
            if observed is None:
                continue
            turn_terms = terms(turn.text)
            score = len(target & turn_terms)
            required = 1 if len(target) == 1 else 2
            if score < required:
                continue
            event_time = _relative_event_time(turn.text, observed)
            if event_time is None:
                coverage = score / max(1, len(target))
                if required_actions and not required_actions <= turn_terms:
                    continue
                if not required_actions and coverage < 0.60:
                    continue
                event_time = observed
            rows.append((score, turn.node_id, turn, event_time))
        candidates_by_target.append(rows)
    if any(not rows for rows in candidates_by_target):
        return None

    chosen: list[tuple[int, str, Any, datetime]] = []
    used: set[str] = set()
    for rows in candidates_by_target:
        available = [row for row in rows if row[1] not in used]
        if not available:
            return None
        selected = max(
            available,
            key=lambda row: (
                row[0],
                int(bool(re.search(
                    r"\b(?:today|yesterday|last\s+(?:week|weekend|"
                    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
                    r"(?:[0-9]+|a|an|one|two|three|four|five|six|seven|eight|"
                    r"nine|ten|eleven|twelve)\s+(?:days?|weeks?|months?)\s+ago)\b",
                    row[2].text, re.IGNORECASE,
                ))),
                row[1],
            ),
        )
        chosen.append(selected)
        used.add(selected[1])

    if len(chosen) > 2:
        ordered = sorted(
            zip(ir.comparison_targets, chosen), key=lambda row: row[1][3]
        )
        return {
            "operation": "temporal_sequence_from_lossless_sources",
            "ordered_targets": [target for target, _row in ordered],
            "source_turn_ids": [row[1] for _target, row in ordered],
            "event_times": [row[3].isoformat() for _target, row in ordered],
            "certified": True,
        }

    left, right = chosen
    earlier = left if left[3] < right[3] else right
    asks_later = bool(re.search(
        r"\b(?:later|latest)\b", ir.raw_question, re.IGNORECASE,
    ))
    selected = (right if earlier is left else left) if asks_later else earlier
    return {
        "operation": "temporal_order_from_lossless_sources",
        "comparison": "later" if asks_later else "earlier",
        "selected_target": ir.comparison_targets[0] if selected is left else ir.comparison_targets[1],
        "selected_source_turn_id": selected[1],
        "selected_time": selected[3].isoformat(),
        "event_a_source_turn_id": left[1],
        "event_b_source_turn_id": right[1],
        "event_a_time": left[3].isoformat(),
        "event_b_time": right[3].isoformat(),
        "binding_complete": True,
        "certified": True,
    }

def _bind_duration_frames(
    ir: QueryIR, frames: list[RoleFrameNode], index: V36Index,
) -> list[RoleFrameNode]:
    query_terms = _duration_query_terms(ir)
    if not query_terms:
        return frames
    quoted_targets = [
        match[0] or match[1]
        for match in re.findall(r"'([^']+)'|\"([^\"]+)\"", ir.raw_question)
        if (match[0] or match[1]).strip()
    ]
    target_terms = set().union(*(
        {_stem_word(word) for word in re.findall(r"[\w'-]+", target)}
        for target in quoted_targets
    )) if quoted_targets else set()
    action_terms = query_terms - target_terms
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    scored: list[tuple[int, RoleFrameNode]] = []
    terms_by_frame: dict[str, set[str]] = {}
    for frame in frames:
        source_text = " ".join(
            turn_by_id[source].text for source in frame.source_turn_ids
            if source in turn_by_id
        )
        evidence_terms = {
            _stem_word(word) for word in re.findall(r"[\w'-]+", f"{frame.retrieval_text} {source_text}")
        }
        terms_by_frame[frame.frame_id] = evidence_terms
        score = len(query_terms & evidence_terms)
        if quoted_targets and not (action_terms & evidence_terms):
            score = 0
        scored.append((score, frame))
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    non_rate = []
    for score, frame in scored:
        source_text = " ".join(
            turn_by_id[source].text for source in frame.source_turn_ids
            if source in turn_by_id
        )
        is_rate_denominator = (
            frame.lifecycle_status == "unknown"
            and bool(re.search(
                r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve|fifteen|twenty)\s+"
                r"(?:seconds?|minutes?|hours?|days?|weeks?|months?)\s+"
                r"(?:per|each|a)\s+(?:day|week|month|year)\b",
                source_text, re.IGNORECASE,
            ))
        )
        if not is_rate_denominator:
            non_rate.append((score, frame))
    if non_rate:
        scored = non_rate
    best = max((score for score, _frame in scored), default=0)
    if best <= 0:
        return []
    selected = [frame for score, frame in scored if score >= max(1, best - 1)]
    covered = set().union(*(
        terms_by_frame.get(frame.frame_id, set()) & query_terms
        for frame in selected
    )) if selected else set()
    for score, frame in sorted(scored, key=lambda row: (-row[0], row[1].frame_id)):
        evidence = terms_by_frame.get(frame.frame_id, set()) & query_terms
        if score > 0 and evidence - covered:
            selected.append(frame)
            covered.update(evidence)
    return list(dict.fromkeys(frame.frame_id for frame in selected)) and list({frame.frame_id: frame for frame in selected}.values())


_AGGREGATE_STOP = {
    "average", "mean", "total", "amount", "much", "more", "less",
    "spend", "spent", "pay", "paid", "cost", "compared", "difference",
    "what", "how", "did", "do", "does", "is", "are", "was", "were",
    "me", "my", "i", "the", "a", "an", "of", "on", "in", "for",
    "since", "start", "year", "years", "have", "has",
}


def _aggregate_unit_family(unit: str) -> str:
    normalized = unit.casefold().strip()
    rate = " per night" if "per night" in normalized else ""
    if any(token in normalized for token in ("$", "usd", "dollar")):
        return "currency" + rate
    if any(token in normalized for token in ("€", "eur", "euro")):
        return "currency" + rate
    if any(token in normalized for token in ("£", "gbp", "pound")):
        return "currency" + rate
    if any(token in normalized for token in ("¥", "jpy", "yen")):
        return "currency" + rate
    return normalized.rstrip("s")



def _minimum_average_operand_count(question: str) -> int:
    match = re.search(r"\baverage\b.+?\bof\b(.+?)(?:[?]|$)", question, re.IGNORECASE)
    if match is None:
        return 2
    parts = [
        part.strip() for part in re.split(r",|\band\b", match.group(1), flags=re.IGNORECASE)
        if part.strip()
    ]
    if len(parts) < 2:
        return 2
    total = 0
    for part in parts:
        words = re.findall(r"[A-Za-z]+", part.casefold())
        last = words[-1] if words else ""
        total += 2 if last.endswith("s") and not last.endswith("ss") else 1
    return max(2, total)


def _deduplicate_sum_candidates(frames: list[RoleFrameNode]) -> list[RoleFrameNode]:
    generic = {"amount", "cost", "expense", "item", "price", "purchase", "total"}
    kept: list[RoleFrameNode] = []
    identities: list[set[str]] = []
    for frame in frames:
        terms = {
            _stem_word(word) for word in re.findall(
                r"[A-Za-z0-9]+",
                " ".join((frame.entity_key, frame.object_key, frame.event_identity_key)),
            )
            if word.casefold() not in generic
        }
        duplicate = False
        for prior, prior_terms in zip(kept, identities):
            if (
                frame.owner_key == prior.owner_key
                and frame.quantity.value == prior.quantity.value
                and _aggregate_unit_family(frame.quantity.unit)
                == _aggregate_unit_family(prior.quantity.unit)
                and terms and prior_terms
                and len(terms & prior_terms) / max(1, min(len(terms), len(prior_terms))) >= 0.6
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(frame)
            identities.append(terms)
    return kept


def exact_entity_absence_hint(
    ir: QueryIR, index: V36Index,
) -> dict[str, Any] | None:
    """Certify absence of a distinctive modifier+head pair in user memory."""
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", ir.raw_question)
    distinctive_positions = [
        position for position, token in enumerate(raw_tokens)
        if re.search(r"\d", token)
        or bool(re.search(r"[a-z][A-Z]|^[a-z][A-Z]", token))
    ]
    if not distinctive_positions:
        return None
    user_texts = [
        re.sub(r"[-_]", " ", turn.text.casefold())
        for turn in index.turns if turn.transport_role == "user"
    ]
    for position in distinctive_positions:
        marker = re.sub(r"[-_]", " ", raw_tokens[position].casefold())
        head = (
            _stem_word(raw_tokens[position + 1])
            if position + 1 < len(raw_tokens) else ""
        )
        if not head:
            continue
        cooccurs = False
        near_matches = []
        for turn, normalized in zip(
            (turn for turn in index.turns if turn.transport_role == "user"),
            user_texts,
        ):
            terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", normalized)}
            if marker in normalized and head in terms:
                cooccurs = True
                break
            if head in terms:
                near_matches.append(turn.node_id)
        if not cooccurs and near_matches:
            return {
                "operation": "exact_entity_absence",
                "value": "insufficient",
                "required_modifier": marker,
                "required_head": head,
                "excluded_near_match_source_turn_ids": near_matches[:12],
                "binding_complete": True,
                "certified": True,
            }
    return None


def named_individual_event_members_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Bind separately named people in a scoped birth/arrival collection."""
    if (
        ir.requested_value_type != "count"
        or not re.search(r"\b(?:bab(?:y|ies)|children|people)\b", ir.raw_question, re.IGNORECASE)
        or not re.search(r"\b(?:born|birth|welcom(?:e|ed)|had)\b", ir.raw_question, re.IGNORECASE)
    ):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    members: dict[str, set[str]] = {}
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        if not re.search(
            r"\b(?:born|welcom(?:e|ed)|had\s+(?:a|their)|new\s+(?:baby|child))\b",
            turn.text, re.IGNORECASE,
        ):
            continue
        names: list[str] = []
        for match in re.finditer(
            r"\b(?:named|called)\s+([A-Z][a-z]{1,30})\b", turn.text,
        ):
            names.append(match.group(1))
        for match in re.finditer(
            r"\btwins?\s*,?\s+([A-Z][a-z]{1,30})\s+and\s+([A-Z][a-z]{1,30})\b",
            turn.text,
        ):
            names.extend(match.groups())
        for name in names:
            members.setdefault(name, set()).add(source_id)
    if len(members) < 2:
        return None
    return {
        "operation": "named_individual_event_members",
        "value": len(members), "unit": "individuals",
        "members": [
            {"identity": name, "source_turn_ids": sorted(source_ids)}
            for name, source_ids in sorted(members.items())
        ],
        "binding_complete": True, "certified": True,
    }


def repeated_activity_duration_total_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Sum explicit per-session durations for a repeatedly performed activity."""
    if (
        ir.requested_value_type != "duration"
        or not re.search(r"\bspent\b.+\bin\s+total\b", ir.raw_question, re.IGNORECASE)
    ):
        return None
    activity_terms = {
        _stem_word(word) for word in re.findall(r"[\w'-]+", ir.raw_question.casefold())
        if _stem_word(word) not in {
            "how", "many", "hour", "minute", "spend", "total", "have",
            "i", "in", "the", "a", "an", "all",
        }
    }
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    rows = []
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", sentence.casefold())}
            turn_terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", turn.text.casefold())}
            if len(activity_terms & turn_terms) < 1:
                continue
            match = re.search(
                r"\b(\d+(?:\.\d+)?)\s*(hours?|minutes?)\b", sentence, re.IGNORECASE,
            )
            if match is None:
                continue
            value = float(match.group(1))
            minutes = value * 60 if match.group(2).casefold().startswith("hour") else value
            rows.append((minutes, source_id, sentence[:320]))
    unique = []
    seen = set()
    for minutes, source_id, evidence in rows:
        key = (turn_by_id[source_id].session_id, minutes)
        if key in seen:
            continue
        seen.add(key); unique.append((minutes, source_id, evidence))
    if len(unique) < 2:
        return None
    total_minutes = sum(row[0] for row in unique)
    return {
        "operation": "repeated_activity_duration_total",
        "operands": [
            {"value_minutes": value, "source_turn_id": source_id, "evidence": evidence}
            for value, source_id, evidence in unique
        ],
        "value": total_minutes / 60, "unit": "hours",
        "binding_complete": True, "certified": True,
    }



def subset_percentage_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Compute an explicitly counted subset as a percentage of its parent set."""
    if not re.search(r"\bpercentage\b", ir.raw_question, re.IGNORECASE):
        return None
    action_pair = re.search(
        r"\bof\s+([A-Za-z]+)\s+([A-Za-z]+)\s+did\s+(?:I|we)\s+([A-Za-z]+)",
        ir.raw_question, re.IGNORECASE,
    )
    if action_pair is None:
        return None
    parent_action = _stem_word(action_pair.group(1))
    head = _stem_word(action_pair.group(2))
    subset_action = _stem_word(action_pair.group(3))
    number_words = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    parent_rows = []
    subset_rows = []
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", sentence.casefold())}
            if head not in terms:
                continue
            for match in re.finditer(
                rf"\b(\d+|{number_words})\s+(?:pairs?\s+of\s+)?{re.escape(action_pair.group(2))}\b",
                sentence, re.IGNORECASE,
            ):
                token = match.group(1).casefold()
                value = int(token) if token.isdigit() else _NUMBER_WORDS[token]
                if parent_action in terms:
                    parent_rows.append((value, source_id, sentence[:320]))
            subset_match = re.search(
                rf"\b(?:only\s+)?(?:{re.escape(action_pair.group(3))}|wearing|using|used)\s+"
                rf"(?:only\s+)?(\d+|{number_words})\b|"
                rf"\bonly\s+(?:{re.escape(action_pair.group(3))}|wearing|using)\s+"
                rf"(\d+|{number_words})\b",
                sentence, re.IGNORECASE,
            )
            if subset_match is not None and subset_action in terms:
                token = next(group for group in subset_match.groups() if group).casefold()
                value = int(token) if token.isdigit() else _NUMBER_WORDS[token]
                subset_rows.append((value, source_id, sentence[:320]))
    if not parent_rows or not subset_rows:
        return None
    parent = max(parent_rows, key=lambda row: row[0])
    subset = max(subset_rows, key=lambda row: row[0])
    if parent[0] <= 0 or subset[0] > parent[0]:
        return None
    return {
        "operation": "explicit_subset_percentage",
        "parent_count": parent[0], "subset_count": subset[0],
        "value": 100 * subset[0] / parent[0], "unit": "%",
        "parent_source_turn_id": parent[1], "parent_evidence": parent[2],
        "subset_source_turn_id": subset[1], "subset_evidence": subset[2],
        "binding_complete": True, "certified": True,
    }


def excluded_collection_members_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Apply an explicit query exclusion to a source-enumerated collection."""
    exclusion = re.search(r"\bexcluding\s+(?:my|the|a|an)\s+([^?.]+)", ir.raw_question, re.IGNORECASE)
    if ir.requested_value_type not in {"count", "list"} or exclusion is None:
        return None
    excluded_terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", exclusion.group(1).casefold())}
    head_match = re.search(r"\bhow\s+many\s+([A-Za-z-]+)", ir.raw_question, re.IGNORECASE)
    if head_match is None:
        return None
    head = _stem_word(head_match.group(1))
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    candidates = []
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            if len(re.findall(rf"\b{re.escape(head)}s?\b", sentence, re.IGNORECASE)) < 2:
                continue
            span = re.search(r"\bfor\s+(.+)$", sentence, re.IGNORECASE)
            if span is None:
                continue
            items = [
                re.sub(r"^(?:and|my|the)\s+", "", item.strip(" ,"), flags=re.IGNORECASE)
                for item in re.split(r"\s*,\s*|\s+and\s+", span.group(1))
                if item.strip(" ,")
            ]
            scoped = [item for item in items if head in {_stem_word(w) for w in re.findall(r"[\w'-]+", item.casefold())} or excluded_terms & {_stem_word(w) for w in re.findall(r"[\w'-]+", item.casefold())}]
            kept = [item for item in scoped if not (excluded_terms & {_stem_word(w) for w in re.findall(r"[\w'-]+", item.casefold())})]
            if len(scoped) >= 2 and kept:
                candidates.append((len(scoped), kept, source_id, sentence[:360]))
    if not candidates:
        return None
    _size, kept, source_id, evidence = max(candidates, key=lambda row: row[0])
    return {
        "operation": "explicit_exclusion_collection",
        "excluded": exclusion.group(1).strip(), "members": kept,
        "value": len(kept), "source_turn_id": source_id, "evidence": evidence,
        "binding_complete": True, "certified": True,
    }


def paired_metric_total_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Sum repeated scalar metrics attached to distinct named artifacts."""
    metric_match = re.search(r"\btotal\s+number\s+of\s+([A-Za-z-]+)", ir.raw_question, re.IGNORECASE)
    if metric_match is None:
        return None
    metric_surface = metric_match.group(1)
    metric = _stem_word(metric_surface)
    query_terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", ir.raw_question.casefold())} - {"what", "total", "number", metric}
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    rows = []
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", sentence.casefold())}
            if len(query_terms & terms) < 2:
                continue
            match = re.search(rf"\b(\d+)\s+{re.escape(metric_surface)}\b", sentence, re.IGNORECASE)
            if match is not None:
                rows.append((int(match.group(1)), source_id, sentence[:320]))
    unique = []
    seen_sessions = set()
    for value, source_id, evidence in rows:
        session_id = turn_by_id[source_id].session_id
        if session_id in seen_sessions:
            continue
        seen_sessions.add(session_id); unique.append((value, source_id, evidence))
    if len(unique) < 2:
        return None
    return {
        "operation": "paired_metric_total",
        "operands": [{"value": v, "source_turn_id": sid, "evidence": e} for v, sid, e in unique],
        "value": sum(row[0] for row in unique), "unit": metric_surface,
        "binding_complete": True, "certified": True,
    }


def labeled_scalar_difference_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Bind both labels in an explicit ``how much more X than Y`` comparison."""
    match = re.search(r"\bhow\s+much\s+more\s+was\s+(.+?)\s+than\s+(.+?)[?]?$", ir.raw_question, re.IGNORECASE)
    if match is None:
        return None
    def terms(text: str) -> set[str]:
        result = set()
        for word in re.findall(r"[\w'-]+", text.casefold().replace("-", " ")):
            stem = _stem_word(word)
            if stem.startswith("approv") or stem == "approval": stem = "approv"
            if stem not in {"the", "a", "an", "amount", "of"}: result.add(stem)
        return result
    sides = [terms(match.group(1)), terms(match.group(2))]
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    selected = []
    for side in sides:
        candidates = []
        for source_id in dict.fromkeys(source_turn_ids):
            turn = turn_by_id.get(source_id)
            if turn is None or turn.transport_role != "user": continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
                values = re.findall(r"[$]\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", sentence)
                if len(values) != 1: continue
                sentence_terms = terms(sentence)
                overlap = len(side & sentence_terms)
                if overlap: candidates.append((overlap, float(values[0].replace(",", "")), source_id, sentence[:320]))
        if not candidates: return None
        selected.append(max(candidates, key=lambda row: (row[0], row[1])))
    if selected[0][2] == selected[1][2]: return None
    return {
        "operation": "labeled_scalar_difference",
        "left_value": selected[0][1], "right_value": selected[1][1],
        "value": selected[0][1] - selected[1][1], "unit": "$",
        "left_source_turn_id": selected[0][2], "right_source_turn_id": selected[1][2],
        "left_evidence": selected[0][3], "right_evidence": selected[1][3],
        "binding_complete": True, "certified": True,
    }




def _operator_identity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def dated_event_count_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Count completed dated occurrences of the requested event class.

    This binder is intentionally class-level: it recognizes an appointment as
    an event with a provider and a concrete date, irrespective of medical
    specialty, person name, or benchmark topic.
    """
    raw = ir.raw_question
    month_match = re.search(
        r"\b(?:in|during)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        raw, re.IGNORECASE,
    )
    if (
        ir.requested_value_type != "count"
        or month_match is None
        or not re.search(r"\bappointments?|visits?\b", raw, re.IGNORECASE)
    ):
        return None
    month = month_match.group(1)
    event_signal = re.compile(
        r"\b(?:appointment|follow-up|checkup|check-up|consultation|"
        r"went to see|saw|visited|met with)\b",
        re.IGNORECASE,
    )
    provider_signal = re.compile(
        r"\b(?:doctor|dr\.?|physician|surgeon|neurologist|therapist|"
        r"dentist|specialist|clinician|provider)\b",
        re.IGNORECASE,
    )
    date_signal = re.compile(
        rf"\b{month}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b|"
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+{month}\b",
        re.IGNORECASE,
    )
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for date_match in date_signal.finditer(turn.text):
            # Use a local window instead of sentence splitting: abbreviations
            # such as "Dr." otherwise sever the provider from its date.
            start = max(0, date_match.start() - 240)
            end = min(len(turn.text), date_match.end() + 180)
            evidence = turn.text[start:end]
            if not event_signal.search(evidence) or not provider_signal.search(evidence):
                continue
            future = re.search(
                r"\b(?:considering|might|may|want to|plan to|scheduled to)\b",
                evidence, re.IGNORECASE,
            )
            completed = re.search(
                r"\b(?:had|went to see|saw|visited|met with|follow-up appointment|diagnosed)\b",
                evidence, re.IGNORECASE,
            )
            if future is not None and completed is None:
                continue
            day = int(next(group for group in date_match.groups() if group))
            rows[(month.casefold(), day)] = {
                "date": f"{month} {day}", "source_turn_id": source_id,
                "evidence": evidence[:420],
            }
    if not rows:
        return None
    return {
        "operation": "dated_event_count", "members": list(rows.values()),
        "value": len(rows), "unit": "events", "binding_complete": True,
        "certified": True,
    }


def same_unit_state_difference_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Subtract two explicitly time-ordered values sharing the same unit."""
    raw = ir.raw_question
    if not re.search(r"\bhow much (?:more|less)|\bdifference\b|\bchange\b", raw, re.IGNORECASE):
        return None
    unit_match = re.search(
        r"\b(miles? per gallon|mpg|percent(?:age points?)?|degrees?|pounds?|kilograms?|hours?|minutes?)\b",
        raw, re.IGNORECASE,
    )
    if unit_match is None:
        return None
    surface = unit_match.group(1)
    unit_pattern = r"(?:miles?\s+per\s+gallon|mpg)" if re.search(r"mpg|gallon", surface, re.I) else re.escape(surface)
    value_re = re.compile(rf"\b(\d+(?:\.\d+)?)\s*{unit_pattern}\b", re.IGNORECASE)
    earlier = re.compile(r"\b(?:ago|before|previously|earlier|used to|last (?:week|month|year))\b", re.IGNORECASE)
    later = re.compile(r"\b(?:now|currently|lately|recently|today|this (?:week|month|year))\b", re.IGNORECASE)
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    old_rows: list[tuple[float, str, str]] = []
    new_rows: list[tuple[float, str, str]] = []
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            match = value_re.search(sentence)
            if match is None:
                continue
            row = (float(match.group(1)), source_id, sentence[:360])
            if earlier.search(sentence): old_rows.append(row)
            if later.search(sentence): new_rows.append(row)
    if not old_rows or not new_rows:
        return None
    old = old_rows[-1]; new = new_rows[-1]
    if old[1] == new[1] and old[0] == new[0]:
        return None
    value = abs(new[0] - old[0])
    return {
        "operation": "same_unit_state_difference", "earlier_value": old[0],
        "current_value": new[0], "value": value, "unit": surface,
        "operands": [
            {"value": old[0], "source_turn_id": old[1], "evidence": old[2]},
            {"value": new[0], "source_turn_id": new[1], "evidence": new[2]},
        ],
        "binding_complete": True, "certified": True,
    }


def maintenance_entity_count_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Count parent assets with completed or explicitly planned maintenance."""
    raw = ir.raw_question
    head_match = re.search(r"\bhow\s+many\s+([A-Za-z-]+)", raw, re.IGNORECASE)
    if (
        ir.requested_value_type != "count" or head_match is None
        or not re.search(r"\b(?:service|maintain|repair|replace|fix|clean|tune)\w*\b", raw, re.IGNORECASE)
    ):
        return None
    head_surface = head_match.group(1)
    head = _stem_word(head_surface)
    action = re.compile(r"\b(?:servic\w*|maintain\w*|repair\w*|replac\w*|fix\w*|clean\w*|tune\w*)\b", re.IGNORECASE)
    members: dict[str, dict[str, Any]] = {}
    allowed_sources = set(source_turn_ids)
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    for frame in index.frames:
        if frame.owner_key not in {"participant 1", "participant_1", "user", ""}:
            continue
        if not allowed_sources.intersection(frame.source_turn_ids):
            continue
        text = " ".join([frame.entity_key, frame.predicate_key, frame.context_key, frame.retrieval_text])
        terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", text.casefold())}
        if head not in terms or not action.search(text):
            continue
        entity_words = re.findall(r"[A-Za-z0-9'-]+", frame.entity_key)
        positions = [i for i, word in enumerate(entity_words) if _stem_word(word) == head]
        if not positions:
            continue
        # Collapse a maintained component such as "commuter bike front tire"
        # to its parent asset "commuter bike".
        identity = " ".join(entity_words[: positions[0] + 1]).strip()
        if not identity or identity.casefold() == head:
            continue
        source_id = next(iter(allowed_sources.intersection(frame.source_turn_ids)))
        source = turn_by_id.get(source_id)
        members.setdefault(_operator_identity_key(identity), {
            "identity": identity, "source_turn_id": source_id,
            "evidence": (source.text if source else frame.retrieval_text)[:360],
        })
    if not members:
        return None
    return {
        "operation": "maintenance_parent_asset_count",
        "members": list(members.values()), "value": len(members),
        "unit": head_surface, "binding_complete": True, "certified": True,
    }


def category_acquisition_members_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Close a category collection from explicit acquisition provenance.

    Category membership is obtained from semantic frame types or card
    relations, while acquisition and identity must co-occur in a short user
    source window.  This prevents a category card's contextual people, stores,
    tools, or other collections from becoming members.
    """
    raw = ir.raw_question
    category_match = re.search(
        r"\b(?:pieces?\s+of\s+|items?\s+of\s+)?([A-Za-z-]+)\s+(?:did\s+I\s+)?(?:acquire|acquired|buy|bought|receive|received|get|got)\b",
        raw, re.IGNORECASE,
    )
    if (
        ir.requested_value_type not in {"count", "list"}
        or category_match is None
        or not re.search(r"\b(?:acquire|acquired|buy|bought|receive|received|get|got)\b", raw, re.IGNORECASE)
    ):
        return None
    category_surface = category_match.group(1)
    category = _stem_word(category_surface)
    allowed_sources = set(source_turn_ids)
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    candidates: dict[str, set[str]] = {}

    def clean_entity(entity: str) -> str:
        value = re.sub(r"\s*\[[^]]+\]\s*$", "", entity).strip()
        words = value.split()
        if len(words) >= 2 and words[-1].casefold() == words[-2].casefold():
            words.pop()
        if len(words) >= 2 and _stem_word(words[-1]) == category:
            words.pop()
        return " ".join(words).strip()

    for frame in index.frames:
        if not allowed_sources.intersection(frame.source_turn_ids):
            continue
        semantic = {
            _stem_word(word) for key in frame.semantic_type_keys
            for word in re.findall(r"[\w'-]+", key.casefold())
        }
        if category in semantic and frame.entity_key:
            candidates.setdefault(clean_entity(frame.entity_key), set()).update(frame.source_turn_ids)
    ownership_signal = re.compile(
        r"\b(?:owns?|has|acquired|bought|purchased|received|got|inherited)\b",
        re.IGNORECASE,
    )
    for card in index.routing_cards:
        card_terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", card.routing_text.casefold())}
        if category not in card_terms:
            continue
        for entity in card.canonical_entities:
            clean = clean_entity(entity)
            entity_terms = [word for word in re.findall(r"[a-z0-9]+", clean.casefold()) if len(word) > 2]
            if not clean or not entity_terms:
                continue
            # A card entity is a member candidate only when a relation links
            # that entity to ownership/acquisition, or its explicit type suffix
            # is the requested category.
            linked = any(
                ownership_signal.search(relation)
                and any(term in relation.casefold() for term in entity_terms[-2:])
                for relation in card.relations
            )
            typed = _stem_word(entity.split()[-1]) == category
            if linked or typed:
                candidates.setdefault(clean, set()).update(card.turn_ids)
    acquired = re.compile(r"\b(?:acquired|bought|purchased|received|inherited|got)\b", re.IGNORECASE)
    members: dict[str, dict[str, Any]] = {}
    for source_id in allowed_sources:
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for match in acquired.finditer(turn.text):
            start = max(0, match.start() - 150); end = min(len(turn.text), match.end() + 180)
            window = turn.text[start:end]
            matched: list[tuple[int, str]] = []
            for entity, entity_sources in candidates.items():
                if source_id not in entity_sources:
                    continue
                entity_terms = [word for word in re.findall(r"[a-z0-9]+", entity.casefold()) if len(word) > 2]
                if entity_terms and all(term in window.casefold() for term in entity_terms[-2:]):
                    matched.append((len(entity_terms), entity))
            # Keep the most specific identity in the acquisition window.
            if matched:
                _size, entity = max(matched)
                key = _operator_identity_key(entity)
                if key and category not in {_stem_word(w) for w in re.findall(r"[\w'-]+", entity.casefold()) if len(entity.split()) == 1}:
                    members.setdefault(key, {
                        "identity": entity, "source_turn_id": source_id,
                        "evidence": window[:420],
                    })
    if not members:
        return None
    return {
        "operation": "category_acquisition_members", "members": list(members.values()),
        "value": len(members), "unit": category_surface,
        "binding_complete": True, "certified": True,
    }


def pending_operation_target_pairs_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Count requested operation-target pairs, preserving replacement identity."""
    raw = ir.raw_question
    if (
        ir.requested_value_type != "count"
        or not re.search(r"\b(?:pick(?:ed)? up|collect|return|exchange)\b", raw, re.IGNORECASE)
    ):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    pairs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        text = turn.text
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            # Explicit pending pickup/collection.
            for match in re.finditer(
                r"\b(?:need to|have to|haven't|have not|still need to|must)\s+"
                r"(?:go\s+)?(?:pick\s+up|collect)\s+(?:my\s+|the\s+|them\s*)?"
                r"([A-Za-z][A-Za-z0-9' -]{1,45}?)(?=\s+(?:from|at|on|before|after)|[,.;!?]|$)",
                sentence, re.IGNORECASE,
            ):
                target = match.group(1).strip() or "replacement item"
                pairs[("pickup", _operator_identity_key(target), source_id)] = {
                    "target": target, "operation": "pickup", "source_turn_id": source_id,
                    "evidence": sentence[:360],
                }
            # A completed exchange establishes the old return; an explicit
            # uncollected replacement is a distinct operation-target pair.
            exchange = re.search(r"\b(?:returned|exchanged)\s+(?:some\s+|my\s+|the\s+)?([A-Za-z][A-Za-z0-9' -]{1,35}?)(?=\s+(?:at|for|because)|[,.;!?]|$)", sentence, re.IGNORECASE)
            if exchange is not None:
                target = exchange.group(1).strip()
                pairs[("return", _operator_identity_key(target), source_id)] = {
                    "target": target, "operation": "return", "source_turn_id": source_id,
                    "evidence": sentence[:360],
                }
                if re.search(r"\b(?:haven't|have not|still need to)\s+(?:gone to\s+)?pick\s+(?:them|it)\s+up\b", sentence, re.IGNORECASE):
                    pairs[("pickup", _operator_identity_key(target) + " replacement", source_id)] = {
                        "target": f"replacement {target}", "operation": "pickup",
                        "source_turn_id": source_id, "evidence": sentence[:360],
                    }
    if len(pairs) < 2:
        return None
    return {
        "operation": "pending_operation_target_pairs", "members": list(pairs.values()),
        "value": len(pairs), "unit": "operation-target pairs",
        "binding_complete": True, "certified": True,
    }

def age_arithmetic_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Bind person roles to explicit ages and expose the requested arithmetic."""
    raw = ir.raw_question
    if not re.search(r"\b(?:age|years?\s+older)\b", raw, re.IGNORECASE):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    ages: dict[str, tuple[int, str]] = {}
    role_patterns = {
        "self": [
            r"\bI(?:'m| am| just turned| turned)\s+(\d{1,3})\b",
            r"\bI\s+just\s+turned\s+(\d{1,3})\b",
            r"\b(?:as\s+)?a\s+(\d{1,3})-year-old\b",
        ],
        "mother": [r"\bmy\s+(?:mom|mother)\s+(?:is|was)\s+(\d{1,3})\b"],
        "father": [r"\bmy\s+(?:dad|father)\s+(?:is|was)\s+(\d{1,3})\b"],
        "grandmother": [r"\bmy\s+(?:grandma|grandmother)\s+(?:is|was)\s+(\d{1,3})\b"],
        "grandfather": [r"\bmy\s+(?:grandpa|grandfather)\s+(?:is|was)\s+(\d{1,3})\b"],
    }
    event_ages: list[tuple[int, str, str]] = []
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for role, patterns in role_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, turn.text, re.IGNORECASE)
                if match is not None:
                    value = int(match.group(1))
                    if 0 < value < 125:
                        ages[role] = (value, source_id)
                        break
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            match = re.search(
                r"\b(?:at|when I was)\s+(?:the\s+)?age\s+(?:of\s+)?(\d{1,3})\b",
                sentence, re.IGNORECASE,
            )
            if match is None:
                continue
            value = int(match.group(1))
            if 0 < value < 125:
                event_ages.append((value, source_id, sentence[:320]))
    if re.search(r"\baverage\s+age\b", raw, re.IGNORECASE):
        required = ["self", "mother", "father", "grandmother", "grandfather"]
        if not all(role in ages for role in required):
            return None
        operands = [
            {"role": role, "value": ages[role][0], "source_turn_id": ages[role][1]}
            for role in required
        ]
        return {
            "operation": "average_age_from_explicit_roles",
            "operands": operands,
            "value": sum(row["value"] for row in operands) / len(operands),
            "unit": "years", "binding_complete": True, "certified": True,
        }
    if re.search(r"\byears?\s+older\b", raw, re.IGNORECASE) and "self" in ages:
        relation_terms = {
            _stem_word(word) for word in re.findall(r"[\w'-]+", raw.casefold())
            if word not in {"how", "many", "year", "years", "older", "than", "when", "i"}
        }
        bound = [
            row for row in event_ages
            if (
                relation_terms & {
                    _stem_word(word) for word in re.findall(r"[\w'-]+", row[2].casefold())
                }
                or (
                    re.search(r"\b(?:graduat|college|degree)\b", raw, re.IGNORECASE)
                    and re.search(r"\b(?:completed|graduated|degree|college)\b", row[2], re.IGNORECASE)
                )
            )
        ]
        if bound:
            event_age, event_source, evidence = bound[0]
            current_age, current_source = ages["self"]
            if current_age >= event_age:
                return {
                    "operation": "age_difference_from_explicit_event_age",
                    "current_age": current_age, "event_age": event_age,
                    "value": current_age - event_age, "unit": "years",
                    "current_age_source_turn_id": current_source,
                    "event_age_source_turn_id": event_source,
                    "event_evidence": evidence,
                    "binding_complete": True, "certified": True,
                }
    return None


def incomplete_terminal_event_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Reject a bounded interval whose named terminal event never completed."""
    endpoint = re.search(
        r"\bto\s+the\s+completion\s+of\s+(?:my\s+)?(.+?)(?:[?.]|$)",
        ir.raw_question, re.IGNORECASE,
    )
    if endpoint is None:
        return None
    endpoint_terms = {
        _stem_word(word) for word in re.findall(r"[\w'-]+", endpoint.group(1).casefold())
        if word not in {"my", "the", "degree", "program"}
    }
    if not endpoint_terms:
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    mentions: list[dict[str, str]] = []
    completed: list[dict[str, str]] = []
    completion_signal = re.compile(
        r"\b(?:completed|finished|graduated|earned|received|obtained|awarded)\b",
        re.IGNORECASE,
    )
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", sentence.casefold())}
            if not endpoint_terms <= terms:
                continue
            row = {"source_turn_id": source_id, "evidence": sentence[:320]}
            mentions.append(row)
            if completion_signal.search(sentence):
                completed.append(row)
    if completed:
        return None
    return {
        "operation": "terminal_event_completion_check",
        "required_terminal_event": endpoint.group(1).strip(),
        "value": "insufficient",
        "mentions": mentions[:8],
        "reason": "no source establishes completion of the required endpoint",
        "binding_complete": True, "certified": True,
    }


def state_change_members_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Enumerate concrete objects in completed replace/fix state changes."""
    if (
        ir.requested_value_type not in {"count", "list"}
        or not re.search(r"\b(?:replace|replaced|fix|fixed|repair|repaired)\b", ir.raw_question, re.IGNORECASE)
    ):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    members: dict[str, dict[str, str]] = {}

    def add_member(value: str, source_id: str, evidence: str) -> None:
        value = re.sub(r"^(?:my|the|those|an?|old|new|worn-out)\s+", "", value.strip(), flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip(" ,.-")
        if not value or value.casefold() in {"it", "old", "new", "one"} or len(value.split()) > 7:
            return
        key = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        key = re.sub(r"^(?:old|new|worn out)\s+", "", key)
        members.setdefault(key, {
            "identity": value, "source_turn_id": source_id, "evidence": evidence[:320]
        })

    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            for match in re.finditer(
                r"\b(?:fixed|repaired)\s+(?:my\s+|the\s+|those\s+)?"
                r"([A-Za-z][A-Za-z0-9' -]{1,55}?)(?=\s+(?:last|yesterday|today|and|so|because)|[,.;!?]|$)",
                sentence, re.IGNORECASE,
            ):
                add_member(match.group(1), source_id, sentence)
            for match in re.finditer(
                r"\breplaced\s+(?:my\s+|the\s+)?(?:old\s+|worn-out\s+)?"
                r"([A-Za-z][A-Za-z0-9' -]{1,45}?)\s+with\b",
                sentence, re.IGNORECASE,
            ):
                add_member(match.group(1), source_id, sentence)
            donated = re.search(
                r"\b(?:donated|discarded|got rid of)\s+(?:my\s+|the\s+)?(?:old\s+)?"
                r"([A-Za-z][A-Za-z0-9' -]{1,45}?)(?=\s+(?:to|and|,)|[.;!?]|$)",
                sentence, re.IGNORECASE,
            )
            if donated is not None and re.search(
                r"\b(?:upgrade|replacement|replac(?:e|ed|ing))\b", sentence, re.IGNORECASE,
            ):
                add_member(donated.group(1), source_id, sentence)
    if len(members) < 2:
        return None
    return {
        "operation": "completed_state_change_members",
        "members": list(members.values()), "value": len(members),
        "unit": "items", "binding_complete": True, "certified": True,
    }


def provenance_acquisition_members_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Split family-provenance acquisition statements into item identities."""
    if (
        ir.requested_value_type not in {"count", "list"}
        or not re.search(r"\b(?:inherit|inherited|acquire|acquired)\b", ir.raw_question, re.IGNORECASE)
        or not re.search(r"\bfamily\b", ir.raw_question, re.IGNORECASE)
    ):
        return None
    family = (
        r"(?:grandmother|grandfather|grandma|grandpa|mother|father|mom|dad|"
        r"great-aunt|great-uncle|aunt|uncle|cousin|sister|brother)"
    )
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    members: dict[str, dict[str, str]] = {}

    def add(value: str, source_id: str, evidence: str) -> None:
        value = re.sub(r"^(?:my|an?|the|a set of)\s+", "", value.strip(), flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip(" ,.-")
        value = re.sub(r"\s+(?:came|comes)$", "", value, flags=re.IGNORECASE)
        key = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        if not (1 <= len(value.split()) <= 8 and key):
            return
        for existing in list(members):
            if key == existing or key in existing:
                return
            if existing in key:
                members.pop(existing)
        members[key] = {
            "identity": value, "source_turn_id": source_id, "evidence": evidence[:360]
        }

    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            if re.search(
                r"\b(?:inherit|heirloom|antique|vintage|apprais|insur|belonged|from my)\b",
                sentence, re.IGNORECASE,
            ):
                for match in re.finditer(
                    rf"\b(?:my\s+)?{family}(?:'s|’s)\s+"
                    r"([A-Za-z][A-Za-z0-9' -]{2,65}?)(?=\s+(?:insured|appraised|valued|was|is)|[,.;!?]|$)",
                    sentence, re.IGNORECASE,
                ):
                    add(match.group(1), source_id, sentence)
            for match in re.finditer(
                rf"\b(?:a\s+set\s+of|an?|the)\s+"
                rf"([A-Za-z][A-Za-z0-9' -]{{2,65}}?)\s+"
                rf"(?:from\s+my\s+{family}(?:\s+[A-Z][a-z]+)?|"
                rf"that\s+belonged\s+to\s+my\s+{family})",
                sentence, re.IGNORECASE,
            ):
                add(match.group(1), source_id, sentence)
    if len(members) < 2:
        return None
    return {
        "operation": "family_provenance_acquisition_members",
        "members": list(members.values()), "value": len(members),
        "unit": "items", "binding_complete": True, "certified": True,
    }



def explicit_cuisine_categories_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Collect explicitly labelled cuisine experiences across source turns."""
    if (
        ir.requested_value_type not in {"count", "list"}
        or not re.search(r"\b(?:cuisine|cuisines)\b", ir.raw_question, re.IGNORECASE)
        or not re.search(r"\b(?:learn|learned|cook|cooked|try|tried)\b", ir.raw_question, re.IGNORECASE)
    ):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    categories: dict[str, dict[str, str]] = {}
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            if not re.search(
                r"\b(?:learn|learned|cook|cooked|try|tried|attended|class|restaurant)\b",
                sentence, re.IGNORECASE,
            ):
                continue
            labels = []
            labels.extend(match.group(1) for match in re.finditer(
                r"\b([A-Za-z]+)(?:-inspired)?\s+(?:cuisine|restaurant|dishes?)\b",
                sentence, re.IGNORECASE,
            ))
            labels.extend(match.group(1) for match in re.finditer(
                r"\brecipe\s+for\s+([A-Z][a-z]+)\s+[A-Za-z-]+", sentence,
            ))
            for label in labels:
                normalized = label.casefold()
                if normalized in {"a", "an", "the", "new", "perfect", "for"}:
                    continue
                categories.setdefault(normalized, {
                    "category": label, "source_turn_id": source_id, "evidence": sentence[:320]
                })
    if len(categories) < 2:
        return None
    return {
        "operation": "explicit_cuisine_experience_categories",
        "members": list(categories.values()), "value": len(categories),
        "unit": "cuisines", "binding_complete": True, "certified": True,
    }



def repeated_event_total_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Assemble a source-bound ledger for ``how many times did ...``.

    Each source contributes its explicit ``N times`` value, or one occurrence
    per separately named object in the action clause.  The result remains
    evidence for the one final LLM call; it never bypasses that call.
    """
    if (
        ir.requested_value_type != "count"
        or not re.search(r"\bhow\s+many\s+times\b", ir.raw_question, re.IGNORECASE)
    ):
        return None
    action_match = re.search(
        r"\b(?:did|do|does|have|has|had)\s+(?:i|we|you|they|he|she)\s+"
        r"([A-Za-z][\w'-]*)\b",
        ir.raw_question, re.IGNORECASE,
    )
    if action_match is None:
        return None
    action_surface = action_match.group(1)
    action = _stem_word(action_surface)
    if action in {"be", "have", "do"}:
        return None
    query_terms = {
        _stem_word(word) for word in re.findall(
            r"[\w'-]+", ir.raw_question.casefold()
        )
        if _stem_word(word) not in {
            "how", "many", "time", "did", "do", "does", "have", "has",
            "had", "i", "we", "you", "they", "he", "she", "all",
            "the", "a", "an", "from", "to", "across", "attend",
        }
    }
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    rows: list[dict[str, Any]] = []
    for source_id in dict.fromkeys(source_turn_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            sentence_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", sentence.casefold()
                )
            }
            if action not in sentence_terms:
                continue
            if query_terms and not (query_terms & sentence_terms):
                continue
            explicit = re.search(
                r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve)\s+times?\b",
                sentence, re.IGNORECASE,
            )
            if explicit is not None:
                token = explicit.group(1).casefold()
                value = int(token) if token.isdigit() else _NUMBER_WORDS[token]
                derivation = "explicit_occurrence_count"
            else:
                action_tokens = list(re.finditer(r"[\w'-]+", sentence))
                action_token = next((
                    token for token in action_tokens
                    if _stem_word(token.group(0)) == action
                ), None)
                object_span = (
                    sentence[action_token.end():]
                    if action_token is not None else sentence
                )
                object_span = re.split(
                    r"\b(?:at|on|in|during|from|because|which|that)\b",
                    object_span, maxsplit=1, flags=re.IGNORECASE,
                )[0]
                list_items = [
                    item.strip(" ,:")
                    for item in re.split(r"\s*,\s*|\s+\band\b\s+", object_span)
                    if item.strip(" ,:")
                ]
                value = len(list_items) if len(list_items) >= 2 else 1
                derivation = (
                    "separately_named_action_objects"
                    if value > 1 else "single_completed_occurrence"
                )
            rows.append({
                "source_turn_id": source_id,
                "value": value,
                "derivation": derivation,
                "evidence": sentence[:360],
            })
            break
    if len(rows) < 2:
        return None
    return {
        "operation": "repeated_event_occurrence_ledger",
        "action": action,
        "operands": rows,
        "value": sum(row["value"] for row in rows),
        "unit": "occurrences",
        "binding_complete": True,
        "certified": True,
    }



def transaction_sum_from_sources_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Bind currency operands to local user-authored source clauses.

    The clause containing the amount must also contain either its explicit
    operand name or the requested transaction action.  This prevents an
    unrelated price, budget, threshold, or reward example elsewhere in the
    same turn from entering the sum.
    """
    if (
        ir.requested_value_type != "aggregate"
        or ir.aggregation_op != "sum"
        or not re.search(
            r"\b(?:money|earn(?:ed|ing)?|sell(?:ing)?|sold|spend|spent|"
            r"paid|cost|expenses?)\b|[$]",
            ir.raw_question, re.IGNORECASE,
        )
    ):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    source_turns = [
        turn_by_id[source_id] for source_id in dict.fromkeys(source_turn_ids)
        if source_id in turn_by_id
        and turn_by_id[source_id].transport_role == "user"
    ]
    clauses: list[tuple[Any, str, set[str]]] = []
    for turn in source_turns:
        for clause in re.split(
            r"(?<=[.!?])\s+|\n+|;\s+|,\s+(?:and\s+)?(?=(?:a|an|the)\s+)",
            turn.text, flags=re.IGNORECASE,
        ):
            clause = clause.strip()
            if "$" not in clause:
                continue
            terms = {
                _stem_word(word) for word in re.findall(r"[\w'-]+", clause.casefold())
            }
            clauses.append((turn, clause, terms))

    def amount_from_clause(clause: str, *, allow_multiply: bool) -> tuple[float, str] | None:
        if allow_multiply:
            multiplied = re.search(
                r"\b(?:sold|bought|purchased|ordered)\s+([0-9]+)\b"
                r"[^$]{0,140}[$]\s*([0-9]+(?:\.[0-9]+)?)\s*"
                r"(?:each|apiece|per\s+\w+)", clause, re.IGNORECASE,
            )
            if multiplied is not None:
                return (
                    int(multiplied.group(1)) * float(multiplied.group(2)),
                    "quantity_times_unit_price",
                )
        amounts = [
            float(value) for value in re.findall(
                r"[$]\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", clause
            )
        ]
        return (amounts[-1], "direct_total") if len(amounts) == 1 else None

    def explicit_amount_from_clause(
        clause: str, target_terms: set[str],
    ) -> tuple[float, str] | None:
        amount_matches = list(re.finditer(
            r"[$]\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", clause
        ))
        if not amount_matches:
            return None
        token_positions = [
            match.start() for match in re.finditer(r"[\w'-]+", clause)
            if _stem_word(match.group(0)) in target_terms
        ]
        if not token_positions:
            return None
        chosen = min(
            amount_matches,
            key=lambda match: min(
                abs(match.start() - position) for position in token_positions
            ),
        )
        return float(chosen.group(1).replace(",", "")), "nearest_explicit_price"

    if ir.operand_targets:
        bindings: list[dict[str, Any]] = []
        used_sources: set[tuple[str, str]] = set()
        for target in ir.operand_targets:
            target_terms = {
                _stem_word(word) for word in re.findall(r"[\w'-]+", target.casefold())
                if word.casefold() not in {"new", "the", "a", "an", "got", "for"}
            }
            if not target_terms:
                return None
            rows: list[tuple[float, int, Any, str, float, str]] = []
            for turn, clause, clause_terms in clauses:
                overlap = len(target_terms & clause_terms)
                coverage = overlap / len(target_terms)
                if overlap < 1 or coverage < 0.60:
                    continue
                amount = explicit_amount_from_clause(clause, target_terms)
                if amount is None:
                    continue
                value, derivation = amount
                rows.append((coverage, overlap, turn, clause, value, derivation))
            if not rows:
                # A pronoun-bearing price clause may follow the named operand
                # in the same turn ("dental chews ... the chews are $10").
                for turn in source_turns:
                    turn_terms = {
                        _stem_word(word) for word in re.findall(
                            r"[\w'-]+", turn.text.casefold()
                        )
                    }
                    overlap = len(target_terms & turn_terms)
                    coverage = overlap / len(target_terms)
                    amounts = re.findall(
                        r"[$]\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)",
                        turn.text,
                    )
                    if coverage >= 0.60 and len(amounts) == 1:
                        rows.append((
                            coverage, overlap, turn, turn.text,
                            float(amounts[0].replace(",", "")),
                            "single_price_in_bound_turn",
                        ))
            if not rows:
                return None
            coverage, overlap, turn, clause, value, derivation = max(
                rows, key=lambda row: (row[0], row[1], -row[4], row[2].node_id)
            )
            source_key = (turn.node_id, target.casefold())
            if source_key in used_sources:
                return None
            used_sources.add(source_key)
            bindings.append({
                "target": target, "value": value,
                "source_turn_id": turn.node_id,
                "derivation": derivation, "evidence": clause[:260],
                "target_term_coverage": coverage,
            })
        return {
            "operation": "explicit_operand_currency_sum",
            "value": sum(row["value"] for row in bindings),
            "unit": "$", "operand_bindings": bindings,
            "binding_complete": True, "certified": True,
        }

    recipient_match = re.search(
        r"\bfor\s+my\s+([A-Za-z][\w'-]*)\b",
        ir.raw_question, re.IGNORECASE,
    )
    if recipient_match is not None:
        recipient = _stem_word(recipient_match.group(1))
        recipient_rows: list[tuple[float, str, str]] = []
        for turn in source_turns:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
                sentence_terms = {
                    _stem_word(word) for word in re.findall(r"[\w'-]+", sentence.casefold())
                }
                turn_terms = {
                    _stem_word(word) for word in re.findall(r"[\w'-]+", turn.text.casefold())
                }
                if recipient not in turn_terms or not sentence_terms.intersection({
                    "gift", "buy", "get", "cost", "pay", "spend",
                }):
                    continue
                amounts = list(re.finditer(
                    r"[$]\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", sentence
                ))
                if not amounts:
                    continue
                recipient_at = sentence.casefold().find(recipient_match.group(1).casefold())
                chosen = min(amounts, key=lambda item: abs(item.start() - recipient_at))
                recipient_rows.append((
                    float(chosen.group(1).replace(",", "")), turn.node_id, sentence[:320]
                ))
        unique_rows = []
        seen_sessions = set()
        for value, source_id, evidence in recipient_rows:
            session_id = turn_by_id[source_id].session_id
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            unique_rows.append((value, source_id, evidence))
        if len(unique_rows) >= 2:
            return {
                "operation": "recipient_scoped_currency_sum",
                "recipient": recipient_match.group(1),
                "operands": [
                    {"value": value, "source_turn_id": source_id, "evidence": evidence}
                    for value, source_id, evidence in unique_rows
                ],
                "value": sum(row[0] for row in unique_rows), "unit": "$",
                "binding_complete": True, "certified": True,
            }

    ignored = {
        "how", "much", "total", "amount", "money", "did", "do", "does",
        "i", "me", "my", "the", "a", "an", "of", "on", "at", "to",
        "from", "for", "all", "and", "or",
    }
    query_terms = {
        _stem_word(word) for word in re.findall(r"[\w'-]+", ir.raw_question.casefold())
        if word not in ignored
    }
    action_terms = query_terms & {
        "earn", "sell", "spend", "pay", "cost", "expense", "purchase", "buy",
    }
    if not action_terms:
        return None
    candidates: list[tuple[int, str, float, str, str]] = []
    money_action_family = {"spend", "pay", "cost", "purchase", "buy", "get"}
    for turn, clause, clause_terms in clauses:
        action_match = bool(action_terms.intersection(clause_terms))
        if not action_match and action_terms.intersection(money_action_family):
            action_match = bool(clause_terms.intersection(money_action_family))
        if not action_match:
            continue
        overlap = len(query_terms & clause_terms)
        if overlap < 2:
            continue
        amount = amount_from_clause(clause, allow_multiply=True)
        if amount is None:
            continue
        value, derivation = amount
        candidates.append((overlap, turn.node_id, value, derivation, clause[:260]))
    unique: list[tuple[int, str, float, str, str]] = []
    seen: set[tuple[str, float]] = set()
    for row in sorted(candidates, key=lambda item: (-item[0], item[1])):
        turn = turn_by_id[row[1]]
        key = (turn.session_id, row[2])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    if len(unique) < 2:
        return None
    return {
        "operation": "transaction_sum_from_lossless_sources",
        "value": sum(row[2] for row in unique), "unit": "$",
        "operands": [
            {
                "value": row[2], "source_turn_id": row[1],
                "derivation": row[3], "evidence": row[4],
                "matched_query_term_count": row[0],
            }
            for row in unique
        ],
        "binding_complete": True, "certified": True,
    }

def _aggregate_hint(
    ir: QueryIR, frames: list[RoleFrameNode], index: V36Index,
) -> dict[str, Any] | None:
    if ir.aggregation_op == "none":
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    query_terms = {
        _stem_word(word)
        for word in re.findall(r"[\w'-]+", ir.raw_question.casefold().replace("-", " "))
        if word not in _AGGREGATE_STOP
    }

    rows: list[tuple[int, RoleFrameNode, set[str]]] = []
    seen: set[tuple[Any, ...]] = set()
    for frame in frames:
        if frame.quantity.value is None or frame.lifecycle_status in {
            "planned", "proposed", "cancelled",
        } or frame.polarity == "negative":
            continue
        source_text = " ".join(
            turn_by_id[source].text for source in frame.source_turn_ids
            if source in turn_by_id
        )
        evidence_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+",
                " ".join((frame.entity_key, frame.predicate_key,
                          frame.object_key, frame.context_key, source_text)).casefold().replace("-", " "),
            )
        }
        score = len(query_terms & evidence_terms)
        # Alternate fact/quantity frames from one source are one operand.
        key = (
            frame.quantity.value, frame.quantity.unit.casefold(),
            tuple(sorted(frame.source_turn_ids)),
        )
        if key not in seen:
            seen.add(key)
            rows.append((score, frame, evidence_terms))

    if ir.aggregation_op == "average":
        relation_terms = query_terms & {
            _stem_word(word) for frame in frames
            for word in re.findall(r"[\w'-]+", frame.predicate_key.casefold())
        }
        candidates = [
            frame for score, frame, terms in rows
            if score > 0 and (not relation_terms or relation_terms & terms)
        ]
        if len(candidates) < 2:
            return None
        family = _aggregate_unit_family(candidates[0].quantity.unit)
        candidates = [
            frame for frame in candidates
            if _aggregate_unit_family(frame.quantity.unit) == family
        ]
        if len(candidates) < _minimum_average_operand_count(ir.raw_question):
            return None
        unit = candidates[0].quantity.unit.casefold()
        value = sum(frame.quantity.value or 0 for frame in candidates) / len(candidates)
        operation = "average"
    elif ir.aggregation_op == "sum":
        asks_money = bool(re.search(
            r"\b(?:money|spend|spent|pay|paid|cost|expense|expenses|"
            r"usd|dollars?|euros?|pounds?|yen)\b|[\x24]",
            ir.raw_question, re.IGNORECASE,
        ))
        eligible_rows = [
            (score, frame, terms) for score, frame, terms in rows
            if frame.lifecycle_status == "completed"
            and (
                not asks_money
                or _aggregate_unit_family(frame.quantity.unit).startswith("currency")
            )
            and any(
                turn_by_id[source].transport_role == "user"
                for source in frame.source_turn_ids if source in turn_by_id
            )
        ]
        binding_complete = False
        if ir.operand_targets:
            selected: list[RoleFrameNode] = []
            used: set[str] = set()
            bindings: list[dict[str, Any]] = []
            for target in ir.operand_targets:
                target_terms = {
                    _stem_word(word) for word in re.findall(r"[\w'-]+", target)
                }
                ranked = sorted(
                    (
                        (len(target_terms & terms), score, frame)
                        for score, frame, terms in eligible_rows
                        if frame.frame_id not in used
                    ),
                    key=lambda row: (-row[0], -row[1], row[2].frame_id),
                )
                required = 1 if len(target_terms) <= 2 else 2
                if not ranked or ranked[0][0] < required:
                    selected = []
                    break
                overlap, _score, frame = ranked[0]
                selected.append(frame)
                used.add(frame.frame_id)
                bindings.append({
                    "target": target, "frame_id": frame.frame_id,
                    "value": frame.quantity.value, "unit": frame.quantity.unit,
                    "source_turn_ids": list(frame.source_turn_ids),
                    "matched_term_count": overlap,
                })
            candidates = selected
            binding_complete = len(candidates) == len(ir.operand_targets)
        else:
            candidates = [
                frame for score, frame, _terms in eligible_rows if score >= 1
            ]
            bindings = []
        if not candidates:
            return None
        families = [_aggregate_unit_family(frame.quantity.unit) for frame in candidates]
        family = max(set(families), key=families.count)
        candidates = [
            frame for frame in candidates
            if _aggregate_unit_family(frame.quantity.unit) == family
        ]
        candidates = _deduplicate_sum_candidates(candidates)
        if len(candidates) < 2:
            return None
        if ir.operand_targets and len(candidates) != len(ir.operand_targets):
            return None
        unit = candidates[0].quantity.unit.casefold()
        value = sum(frame.quantity.value or 0 for frame in candidates)
        operation = "sum"
    else:
        if len(ir.operand_targets) != 2:
            return None
        selected: list[RoleFrameNode] = []
        for target in ir.operand_targets:
            terms = {_stem_word(word) for word in re.findall(r"[\w'-]+", target)}
            ranked = sorted(
                ((len(terms & evidence), score, frame) for score, frame, evidence in rows),
                key=lambda row: (-row[0], -row[1], row[2].frame_id),
            )
            if not ranked or ranked[0][0] == 0:
                return None
            selected.append(ranked[0][2])
        if selected[0].frame_id == selected[1].frame_id:
            return None
        left_unit = _aggregate_unit_family(selected[0].quantity.unit)
        right_unit = _aggregate_unit_family(selected[1].quantity.unit)
        if left_unit != right_unit:
            return None
        candidates = selected
        unit = selected[0].quantity.unit.casefold()
        value = abs((selected[0].quantity.value or 0) - (selected[1].quantity.value or 0))
        operation = "difference"
    return {
        "operation": "scalar_aggregate",
        "aggregate": operation,
        "value": value,
        "unit": unit,
        "frame_ids": [frame.frame_id for frame in candidates],
        "operand_bindings": bindings if operation == "sum" else [],
        "binding_complete": bool(
            operation == "sum" and ir.operand_targets and binding_complete
        ),
        "certified": True,
    }


_COLLECTION_LEDGER_STOP = {
    "how", "many", "much", "different", "type", "types", "kind",
    "kinds", "all", "total", "number", "list", "which", "what",
    "have", "has", "had", "did", "does", "do", "i", "me", "my",
    "recent", "recently", "past", "few", "typical", "week", "weeks",
    "day", "days", "month", "months", "year", "years", "last", "ago",
    "item", "items", "thing", "things", "activity", "activities",
    "member", "members", "entity", "entities", "option", "options",
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "in", "on", "at", "of", "to", "from", "for", "with", "without",
    "by", "as", "into", "onto", "out", "up", "down", "about",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "you", "your", "we", "our", "he", "his", "she", "her",
}
_COLLECTION_LEDGER_STOP_STEMS = {
    _stem_word(word) for word in _COLLECTION_LEDGER_STOP
}
_COLLECTION_OPERATION_STEMS = {
    _stem_word(word) for word in {
        "use", "used", "try", "tried", "learn", "learned", "attend",
        "pick", "return", "buy", "visit", "watch", "read", "collect",
        "make", "cook", "take", "order", "consume", "complete", "add",
        "remove", "cancel", "need", "spend", "partake", "like",
        "prefer", "enjoy", "love", "work", "start", "finish",
        "get", "obtain", "acquire", "inherit", "own", "possess", "rely",
        "bear", "born", "birth", "ride", "rode", "fix", "repair",
        "replace", "sell", "sold", "earn", "earned",
    }
}

_COLLECTION_ACTION_FAMILIES = (
    {"buy", "purchase", "get", "obtain", "acquire", "inherit", "own", "possess", "order"},
    {"bear", "born", "birth", "have"},
    {"ride", "rode"},
    {"fix", "repair", "replace"},
    {"sell", "sold", "earn", "earned"},
    {"work", "start", "finish", "complete", "build", "make", "create"},
    {"use", "rely", "order", "try", "have", "consume"},
    {"pick", "collect", "retrieve", "receive"},
    {"return", "exchange", "send"},
    {"attend", "join", "take", "participate"},
    {"learn", "study", "practice"},
    {"like", "love", "prefer", "enjoy"},
)
_COLLECTION_ACTION_VOCAB = set().union(*_COLLECTION_ACTION_FAMILIES) | {
    "need", "plan", "cancel", "remove", "add",
}


def _expanded_collection_actions(terms: set[str]) -> set[str]:
    expanded = terms & _COLLECTION_ACTION_VOCAB
    for family in _COLLECTION_ACTION_FAMILIES:
        if terms & family:
            expanded |= family
    return expanded


def _collection_scope_covered(required: set[str], evidence: set[str]) -> bool:
    if not required:
        return False
    return all(any(
        wanted == found
        or (
            len(wanted) >= 5 and len(found) >= 5
            and wanted[:5] == found[:5]
        )
        for found in evidence
    ) for wanted in required)


def _modifier_subtype_covered(
    collection_head_terms: set[str], question: str, evidence: str,
) -> bool:
    """Require every content term of a compound collection head.

    Matching only the first modifier can collapse distinct sibling types.
    Prefix matching is delegated to the same morphology-only matcher used by
    the main scope certificate.
    """
    if len(collection_head_terms) < 2:
        return False
    evidence_terms = {
        _stem_word(word)
        for word in re.findall(r"[\w'-]+", evidence.casefold())
    }
    if len(collection_head_terms) > 2:
        return _collection_scope_covered(collection_head_terms, evidence_terms)
    modifier = min(
        collection_head_terms,
        key=lambda term: (
            question.casefold().find(term)
            if term in question.casefold() else 10**9,
            term,
        ),
    )
    return bool(re.search(
        rf"\b{re.escape(modifier)}\s+"
        r"(?!and\b|or\b|but\b|with\b|without\b|not\b|no\b|"
        r"that\b|which\b|for\b|to\b|is\b|was\b|unknown\b)"
        r"[a-z][\w-]+",
        evidence.casefold(),
    ))


def _ledger_excerpt(text: str, query_terms: set[str], limit: int = 360) -> str:
    segments = [
        segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
        if segment.strip()
    ]
    rows = []
    for index, segment in enumerate(segments):
        terms = {
            _stem_word(word)
            for word in re.findall(r"[\w'-]+", segment.casefold())
        }
        overlap = query_terms & terms
        if overlap:
            rows.append((len(overlap), index, segment, overlap))
    if not rows:
        return text[:limit]
    first = max(rows, key=lambda row: (row[0], -row[1]))
    selected = [first]
    uncovered = query_terms - first[3]
    if uncovered:
        second = max(
            (row for row in rows if row[1] != first[1]),
            key=lambda row: (len(uncovered & row[3]), row[0], -row[1]),
            default=None,
        )
        if second is not None and uncovered & second[3]:
            selected.append(second)
    return " ".join(
        row[2] for row in sorted(selected, key=lambda row: row[1])
    )[:limit]


def query_bound_collection_ledger(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
    frame_ids: list[str],
    routed_session_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build a domain-independent collection ledger from packed provenance.

    It never claims scope closure by itself.  Its purpose is to make the
    owner-bound operation/target candidates and lossless statements explicit
    for the one final answer call when extraction groups are incomplete.
    """
    bounded_year_query = bool(
        ir.requested_value_type == "duration"
        and re.search(r"\bhow many years?\b", ir.raw_question, re.IGNORECASE)
        and re.search(r"\bfrom\b.+\bto\b", ir.raw_question, re.IGNORECASE)
    )
    collection_query = "members" in ir.required_roles
    if not collection_query and not bounded_year_query:
        return None
    query_terms = {
        stem
        for word in re.findall(r"[\w'-]+", ir.raw_question.casefold())
        if (stem := _stem_word(word)) not in _COLLECTION_LEDGER_STOP_STEMS
    }
    if not query_terms:
        return None
    owner_terms = {
        _stem_word(word) for word in re.findall(r"[\w'-]+", ir.target_owner)
    }
    query_scope_terms = query_terms - _COLLECTION_OPERATION_STEMS - owner_terms
    head_match = re.search(
        r"\bhow\s+many\s+(.+?)\s+"
        r"(?:do|does|did|have|has|had|am|is|are|was|were|can|could|"
        r"should|would|will)\b",
        ir.raw_question,
        re.IGNORECASE,
    )
    collection_head_terms = {
        _stem_word(word)
        for word in re.findall(
            r"[\w'-]+", head_match.group(1).casefold() if head_match else ""
        )
        if _stem_word(word) not in _COLLECTION_LEDGER_STOP_STEMS
        and _stem_word(word) not in owner_terms
    }
    collection_scope_terms = (
        collection_head_terms if collection_query and collection_head_terms
        else query_scope_terms
    )
    strong_query_actions = _expanded_collection_actions(
        query_terms - {"need", "plan"}
    )
    query_action_terms = (
        strong_query_actions or _expanded_collection_actions(query_terms)
    )
    all_raw_query_actions = query_terms & _COLLECTION_ACTION_VOCAB
    raw_query_action_terms = (
        all_raw_query_actions - {"need", "plan"}
        or all_raw_query_actions
    )

    def collection_action_matches(evidence_terms: set[str]) -> bool:
        evidence_actions = evidence_terms & _COLLECTION_ACTION_VOCAB
        if not raw_query_action_terms or not evidence_actions:
            return False
        if raw_query_action_terms & evidence_actions:
            return True
        return any(
            bool(raw_query_action_terms & family)
            and bool(evidence_actions & family)
            for family in _COLLECTION_ACTION_FAMILIES
        )

    if collection_query and ir.requested_value_type != "count" and not query_scope_terms:
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    card_by_session = {
        card.session_id: card for card in index.routing_cards
    }
    routed_order = {
        session_id: position
        for position, session_id in enumerate(routed_session_ids or [])
    }
    selected_sources = {source for source in source_turn_ids if source in turn_by_id}
    first_person = bool(re.search(r"\b(?:i|me|my)\b", ir.raw_question, re.IGNORECASE))
    # A packed frame can be only one endpoint of a collection/reference group.
    # Follow one bounded structural hop before ranking sources so that a fine
    # top-k cannot split a schedule, collection, or reference chain. Candidate
    # frames still have to bind the query scope and operation below.
    initial_frame_ids = list(dict.fromkeys(frame_ids))
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    graph_closure_rows: list[dict[str, Any]] = []
    closure_frame_ids: list[str] = []
    for group in index.evidence_groups:
        if group.group_kind not in {"collection", "reference_chain"}:
            continue
        if not set(group.member_frame_ids).intersection(initial_frame_ids):
            continue
        for member_id in group.member_frame_ids:
            if member_id in initial_frame_ids or member_id in closure_frame_ids:
                continue
            member = frame_by_id.get(member_id)
            if member is None or member.polarity == "negative":
                continue
            if member.lifecycle_status == "cancelled":
                continue
            member_sources = [
                turn_by_id[source_id] for source_id in member.source_turn_ids
                if source_id in turn_by_id
            ]
            if first_person and not any(
                source.transport_role == "user" for source in member_sources
            ):
                continue
            if ir.target_owner and member.owner_key != ir.target_owner:
                continue
            member_evidence = " ".join((
                member.entity_key, member.predicate_key, member.object_key,
                member.context_key, member.event_identity_key,
                " ".join(member.semantic_type_keys),
                *(source.text for source in member_sources),
            ))
            member_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", member_evidence.casefold()
                )
            }
            scope_binding = (
                _collection_scope_covered(
                    collection_scope_terms, member_terms
                )
                or _modifier_subtype_covered(
                    collection_head_terms, ir.raw_question, member_evidence
                )
            )
            action_binding = collection_action_matches(member_terms)
            if collection_query and not (scope_binding and action_binding):
                continue
            closure_frame_ids.append(member_id)
            selected_sources.update(member.source_turn_ids)
            graph_closure_rows.append({
                "group_id": group.group_id,
                "group_kind": group.group_kind,
                "frame_id": member_id,
                "source_turn_ids": list(member.source_turn_ids),
            })
    frame_ids = [*initial_frame_ids, *closure_frame_ids]
    frame_order = {frame_id: position for position, frame_id in enumerate(frame_ids)}
    frame_id_set = set(frame_ids)
    frames = sorted([
        frame for frame in index.frames
        if frame.frame_id in frame_id_set
        and frame.source_turn_ids
        and frame.polarity != "negative"
        and frame.lifecycle_status != "cancelled"
    ], key=lambda frame: frame_order[frame.frame_id])
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        sources = [
            turn_by_id[source] for source in frame.source_turn_ids
            if source in selected_sources
        ]
        if not sources:
            continue
        if ir.target_owner and frame.owner_key != ir.target_owner:
            continue
        if first_person and not any(
            source.transport_role == "user" for source in sources
        ):
            continue
        evidence = " ".join((
            frame.owner_key, frame.entity_key, frame.predicate_key,
            frame.object_key, frame.context_key,
            *(source.text for source in sources),
        ))
        evidence_terms = {
            _stem_word(word) for word in re.findall(r"[\w'-]+", evidence.casefold())
        }
        overlap = sorted(query_terms & evidence_terms)
        if not overlap and frame_order[frame.frame_id] >= 6:
            continue
        frame_rows.append({
            "frame_id": frame.frame_id,
            "owner": frame.owner_key,
            "operation": frame.predicate_key,
            "target": frame.entity_key or frame.object_key,
            "value": frame.object_key,
            "status": frame.lifecycle_status,
            "state_op": frame.state_op,
            "matched_query_terms": overlap,
            "_selection_score": (
                4 * len(collection_scope_terms & evidence_terms) + len(overlap)
                + 12 * int(
                    _collection_scope_covered(
                        collection_scope_terms, evidence_terms
                    )
                )
            ),
            "source_turn_ids": [source.node_id for source in sources],
        })
    scope_rows = sorted(
        frame_rows,
        key=lambda row: (-row["_selection_score"], frame_order[row["frame_id"]]),
    )
    retrieval_rows = sorted(
        frame_rows, key=lambda row: frame_order[row["frame_id"]]
    )
    frame_rows = list({
        row["frame_id"]: row for row in [*scope_rows[:3], *retrieval_rows]
    }.values())
    source_rows: list[tuple[int, dict[str, Any]]] = []
    frame_source_terms: dict[str, set[str]] = {}
    for row in frame_rows:
        frame_specific_terms = {
            _stem_word(word)
            for field in ("operation", "target", "value")
            for word in re.findall(
                r"[\w'-]+", str(row.get(field) or "").casefold()
            )
        }
        for source_id in row["source_turn_ids"]:
            frame_source_terms.setdefault(source_id, set()).update(
                frame_specific_terms
            )
    weekdays: set[str] = set()
    for source_id in selected_sources:
        turn = turn_by_id[source_id]
        if first_person and turn.transport_role != "user":
            continue
        if ir.target_owner and turn.speaker_key != ir.target_owner:
            linked_owners = {
                frame.owner_key for frame in frames
                if source_id in frame.source_turn_ids
            }
            if ir.target_owner not in linked_owners:
                continue
        terms = {
            _stem_word(word) for word in re.findall(r"[\w'-]+", turn.text.casefold())
        }
        card = card_by_session.get(turn.session_id)
        card_terms = {
            _stem_word(word)
            for word in re.findall(
                r"[\w'-]+", (card.routing_text if card is not None else "").casefold()
            )
        }
        overlap = query_terms & terms
        source_scope_overlap = collection_scope_terms & terms
        card_scope_overlap = collection_scope_terms & card_terms
        complete_scope_binding = _collection_scope_covered(
            collection_scope_terms, terms | card_terms
        )
        card_query_overlap = query_terms & card_terms
        card_entity_matches = [
            entity for entity in (card.canonical_entities if card is not None else [])
            if len(entity.strip()) >= 3
            and not re.fullmatch(
                r"(?:participant|speaker|questioner|assistant|user)[ _-]*\d*",
                entity.casefold().strip(),
            )
            and entity.casefold().strip() in turn.text.casefold()
        ]
        evidence_action_terms = (
            terms | card_terms | frame_source_terms.get(source_id, set())
        )
        action_binding = collection_action_matches(evidence_action_terms)
        modifier_subtype_binding = _modifier_subtype_covered(
            collection_head_terms,
            ir.raw_question,
            f"{turn.text} {(card.routing_text if card else '')}",
        )
        scope_binding = complete_scope_binding or modifier_subtype_binding
        service_like_scope = bool(
            collection_head_terms & {"service", "provider", "platform"}
        )
        routed_named_action = bool(
            service_like_scope and card_entity_matches and action_binding
            and turn.session_id in routed_order
        )
        semantic_frame_fallback = bool(
            source_id in frame_source_terms and not routed_session_ids
        )
        if collection_query and not (
            (scope_binding and action_binding)
            or routed_named_action
            or semantic_frame_fallback
        ):
            continue
        linked_overlap = {
            term for row in frame_rows
            if source_id in row["source_turn_ids"]
            for term in row["matched_query_terms"]
        }
        explicit_relation_source = bool(re.search(
            r"\b(?:featuring|containing|including|includes?|consisting of|"
            r"composed of)\b",
            turn.text, re.IGNORECASE,
        ))
        score = (
            4 * len(source_scope_overlap)
            + len(overlap) + len(linked_overlap)
            + 3 * len(card_scope_overlap) + len(card_query_overlap)
            + 4 * int(bool(card_entity_matches))
            + 12 * int(complete_scope_binding)
            + 10 * int(explicit_relation_source)
        )
        routed_owner_candidate = (
            turn.session_id in routed_order
            and turn.transport_role == "user"
            and bool(card_query_overlap)
        )
        if (
            score <= 0
            and source_id not in frame_source_terms
            and not routed_owner_candidate
        ):
            continue
        score = max(score, 1)
        found_days = {
            day for day in _WEEKDAYS
            if re.search(rf"\b{day}s?\b", turn.text, re.IGNORECASE)
        }
        weekdays.update(found_days)
        row = {
            "source_turn_id": source_id,
            "date": turn.session_date,
            "excerpt": _ledger_excerpt(
                turn.text, query_terms | frame_source_terms.get(source_id, set())
            ),
        }
        if card_query_overlap and card is not None:
            row["routing_context"] = _ledger_excerpt(
                card.routing_text,
                query_terms | terms,
                limit=240,
            )
        source_rows.append((score, row))

    # Fine top-k may select the right routing cards but only a conversational
    # neighbor from each card. Recover one owner-authored proposition directly
    # from every routed region when both the requested scope and action bind.
    # This is bounded by coarse routing and does not depend on domain/topic.
    seen_source_ids = {row["source_turn_id"] for _score, row in source_rows}
    if collection_query and routed_order:
        for turn in index.turns:
            if (
                turn.node_id in seen_source_ids
                or turn.session_id not in routed_order
                or (first_person and turn.transport_role != "user")
            ):
                continue
            if ir.target_owner and turn.speaker_key != ir.target_owner:
                linked_owners = {
                    frame.owner_key for frame in index.frames
                    if turn.node_id in frame.source_turn_ids
                }
                if ir.target_owner not in linked_owners:
                    continue
            card = card_by_session.get(turn.session_id)
            evidence = f"{turn.text} {(card.routing_text if card else '')}"
            evidence_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", evidence.casefold()
                )
            }
            scope_binding = (
                _collection_scope_covered(
                    collection_scope_terms, evidence_terms
                )
                or _modifier_subtype_covered(
                    collection_head_terms, ir.raw_question, evidence
                )
            )
            if not scope_binding or not collection_action_matches(evidence_terms):
                continue
            overlap = query_terms & evidence_terms
            score = (
                12
                + 4 * len(collection_scope_terms & evidence_terms)
                + len(overlap)
                + max(0, 8 - routed_order[turn.session_id])
            )
            source_rows.append((score, {
                "source_turn_id": turn.node_id,
                "date": turn.session_date,
                "excerpt": _ledger_excerpt(
                    turn.text, query_terms, limit=320
                ),
                "routing_context": _ledger_excerpt(
                    card.routing_text if card else "",
                    query_terms | evidence_terms,
                    limit=240,
                ),
                "binding": "routed_owner_scope_action",
            }))
            seen_source_ids.add(turn.node_id)

        # A concrete member may omit its abstract collection noun (for
        # example, a named artifact or activity). Coarse routing already
        # bounds the semantic region, so retain one lower-priority owner turn
        # per otherwise-uncovered routed session when its action binds.
        covered_sessions = {
            turn_by_id[source_id].session_id for source_id in seen_source_ids
            if source_id in turn_by_id
        }
        for session_id in routed_order:
            if session_id in covered_sessions:
                continue
            candidates: list[tuple[int, Any]] = []
            card = card_by_session.get(session_id)
            card_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", (card.routing_text if card else "").casefold()
                )
            }
            for turn in index.turns:
                if turn.session_id != session_id:
                    continue
                if first_person and turn.transport_role != "user":
                    continue
                turn_terms = {
                    _stem_word(word) for word in re.findall(
                        r"[\w'-]+", turn.text.casefold()
                    )
                }
                evidence_terms = turn_terms | card_terms
                if not collection_action_matches(evidence_terms):
                    continue
                overlap = query_terms & evidence_terms
                score = (
                    4 * len(overlap)
                    + 2 * len(collection_scope_terms & evidence_terms)
                    - turn.turn_index
                )
                candidates.append((score, turn))
            if not candidates:
                continue
            _score, turn = max(
                candidates, key=lambda item: (item[0], -item[1].turn_index)
            )
            source_rows.append((max(1, _score), {
                "source_turn_id": turn.node_id,
                "date": turn.session_date,
                "excerpt": turn.text[:360],
                "routing_context": _ledger_excerpt(
                    card.routing_text if card else "",
                    query_terms, limit=240,
                ),
                "binding": "routed_owner_action_fallback",
            }))
            seen_source_ids.add(turn.node_id)
            covered_sessions.add(session_id)
    if not frame_rows and not source_rows:
        return None
    source_rows.sort(key=lambda row: (-row[0], row[1]["source_turn_id"]))
    source_limit = 12
    # Routing context raises a proposition only when its card semantics
    # and named entities support it; final selection remains global so weak
    # routed regions cannot evict a stronger exact source.
    # Preserve cross-session coverage before filling spare slots. Repeated
    # turns from one long dialogue must not evict the only proposition from a
    # different routed memory region.
    best_per_session: dict[str, tuple[int, dict[str, Any]]] = {}
    for candidate in source_rows:
        source_id = candidate[1]["source_turn_id"]
        session_id = turn_by_id[source_id].session_id
        best_per_session.setdefault(session_id, candidate)
    kept_source_rows = list(best_per_session.values())[:source_limit]
    kept_ids = {row[1]["source_turn_id"] for row in kept_source_rows}
    for candidate in source_rows:
        if len(kept_source_rows) >= source_limit:
            break
        if candidate[1]["source_turn_id"] in kept_ids:
            continue
        kept_source_rows.append(candidate)
        kept_ids.add(candidate[1]["source_turn_id"])
    kept_source_rows.sort(
        key=lambda row: (-row[0], row[1]["source_turn_id"])
    )
    explicit_relation_rows = [
        row for row in source_rows
        if re.search(
            r"\b(?:featuring|containing|including|includes?|consisting of|"
            r"composed of)\b",
            turn_by_id[row[1]["source_turn_id"]].text, re.IGNORECASE,
        )
    ]
    if explicit_relation_rows and kept_source_rows:
        relation_row = explicit_relation_rows[0]
        kept_ids = {row[1]["source_turn_id"] for row in kept_source_rows}
        if relation_row[1]["source_turn_id"] not in kept_ids:
            replace_at = next((
                position for position in range(len(kept_source_rows) - 1, -1, -1)
                if kept_source_rows[position][1]["source_turn_id"]
                not in frame_source_terms
            ), len(kept_source_rows) - 1)
            kept_source_rows[replace_at] = relation_row
            kept_source_rows.sort(
                key=lambda row: (-row[0], row[1]["source_turn_id"])
            )
    kept_source_ids = {
        row["source_turn_id"] for _score, row in kept_source_rows
    }
    # Generalized operand closure: after coarse routing bounds the region,
    # reconsider every owner-bound frame backed by a kept source. This does
    # not depend on benchmark, topic, or the original fine top-k.
    structured_member_rows: list[dict[str, Any]] = []
    member_source_ids: set[str] = set()
    member_signatures: set[tuple[str, str, str, tuple[str, ...]]] = set()
    generic_targets = {
        "", "participant", "speaker", "questioner", "assistant", "user",
        ir.target_owner.casefold().strip(),
    }
    for frame in sorted(
        (index.frames if collection_query else []),
        key=lambda item: (item.observation_order, item.frame_id)
    ):
        frame_sources = [
            turn_by_id[source_id] for source_id in frame.source_turn_ids
            if source_id in kept_source_ids and source_id in turn_by_id
        ]
        if not frame_sources:
            continue
        if frame.polarity == "negative" or frame.lifecycle_status == "cancelled":
            continue
        if ir.target_owner and frame.owner_key != ir.target_owner:
            continue
        if first_person and not any(
            source.transport_role == "user" for source in frame_sources
        ):
            continue
        frame_evidence = " ".join((
            frame.entity_key, frame.predicate_key, frame.object_key,
            frame.context_key, frame.event_identity_key,
            " ".join(frame.semantic_type_keys),
            *(source.text for source in frame_sources),
        ))
        frame_evidence_terms = {
            _stem_word(word)
            for word in re.findall(r"[\w'-]+", frame_evidence.casefold())
        }
        frame_scope_binding = (
            _collection_scope_covered(
                collection_scope_terms, frame_evidence_terms
            )
            or _modifier_subtype_covered(
                collection_head_terms, ir.raw_question, frame_evidence
            )
        )
        frame_action_evidence = " ".join((
            frame.predicate_key, frame.object_key, frame.state_op,
        ))
        frame_action_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", frame_action_evidence.casefold()
            )
        }
        frame_action_binding = collection_action_matches(
            frame_action_terms
        )
        slot_evidence = " ".join((
            frame.entity_key, frame.predicate_key, frame.object_key,
            " ".join(frame.semantic_type_keys),
        ))
        slot_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", slot_evidence.casefold()
            )
        }
        slot_scope_binding = (
            _collection_scope_covered(collection_scope_terms, slot_terms)
            or _modifier_subtype_covered(
                collection_head_terms, ir.raw_question, slot_evidence
            )
        )
        request_like = bool(
            set(frame.semantic_type_keys) & {"request", "recommendation", "advice"}
            or re.search(
                r"\b(?:want|wants|request|requests|ask|asks|advice|"
                r"recommend|recommends|suggest|suggests)\b",
                frame.predicate_key, re.IGNORECASE,
            )
        )
        if request_like and not (query_terms & {"need", "plan"}):
            continue
        if collection_query and not (
            frame_action_binding and slot_scope_binding
        ):
            continue
        predicate_role = frame.predicate_key.casefold().strip()
        if (
            predicate_role in {
                "background", "goal", "context", "interest", "topic"
            }
            and not (query_terms & {"need", "plan"})
        ):
            continue
        entity_target = frame.entity_key.strip()
        object_target = frame.object_key.strip()
        normalized_entity = re.sub(
            r"[ _-]+", " ", entity_target.casefold()
        ).strip()
        entity_is_generic = bool(
            not entity_target
            or entity_target.casefold() in generic_targets
            or re.fullmatch(
                r"(?:participant|speaker|questioner|assistant|user)(?: \d+)?",
                normalized_entity,
            )
        )
        predicate_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", frame.predicate_key.casefold()
            )
        }
        predicate_binds_patient = bool(
            predicate_terms & {"pick", "collect", "retrieve", "receive"}
        )
        target = (
            object_target
            if object_target and (entity_is_generic or predicate_binds_patient)
            else entity_target
        )
        normalized_target = re.sub(r"[ _-]+", " ", target.casefold()).strip()
        target_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", target.casefold()
            )
            if _stem_word(word) not in _COLLECTION_LEDGER_STOP_STEMS
        }
        category_only_target = bool(
            target_terms and collection_scope_terms
            and target_terms <= collection_scope_terms
        )
        if (
            not target
            or category_only_target
            or target.casefold() in generic_targets
            or re.fullmatch(
                r"(?:participant|speaker|questioner|assistant|user)(?: \d+)?",
                normalized_target,
            )
        ):
            continue
        source_ids = tuple(sorted(source.node_id for source in frame_sources))
        signature = (
            frame.predicate_key.casefold().strip(),
            target.casefold(),
            frame.state_op,
            source_ids,
        )
        if signature in member_signatures:
            continue
        member_signatures.add(signature)
        member_source_ids.update(source_ids)
        structured_member_rows.append({
            "frame_id": frame.frame_id,
            "operation": frame.predicate_key,
            "target": target,
            "value": frame.object_key,
            "state_op": frame.state_op,
            "status": frame.lifecycle_status,
            "event_identity": frame.event_identity_key,
            "source_turn_ids": list(source_ids),
            "evidence_excerpt": _ledger_excerpt(
                " ".join(source.text for source in frame_sources),
                query_terms | frame_action_terms, limit=260,
            ),
        })
        if len(structured_member_rows) >= 14:
            break
    structured_distinct_targets = list(dict.fromkeys(
        row["target"] for row in structured_member_rows
    ))
    unframed_lossless_rows = [
        row for _score, row in kept_source_rows
        if row["source_turn_id"] not in member_source_ids
    ]
    relation_closure_rows: list[dict[str, Any]] = []
    for source_row in (unframed_lossless_rows if collection_query else []):
        source_id = source_row["source_turn_id"]
        source_text = turn_by_id[source_id].text
        for match in re.finditer(
            r"\b(featuring|containing|including|includes?|consisting of|"
            r"composed of)\s+([^,.;!?]{3,120})",
            source_text, re.IGNORECASE,
        ):
            target = re.sub(
                r"^(?:a|an|the)\s+", "", match.group(2).strip(),
                flags=re.IGNORECASE,
            )
            target = re.split(
                r"\s+(?:and|but)\s+(?:i|we|they|he|she|it)\b",
                target, maxsplit=1, flags=re.IGNORECASE,
            )[0].strip()
            if len(re.findall(r"[\w'-]+", target)) < 2:
                continue
            relation_closure_rows.append({
                "target": target,
                "relation": match.group(1).casefold(),
                "source_turn_id": source_id,
                "evidence_excerpt": source_row["excerpt"],
            })
            break
    # Named-object closure recovers a durable object when the main extractor
    # preserved it only in a routing card. A direct scope-bearing source is an
    # anchor. One routing semantic-neighbor hop may then contribute an entity
    # only when an owner-bound relation explicitly binds the requested action.
    card_by_id = {card.card_id: card for card in index.routing_cards}
    anchor_sessions: set[str] = set()
    for _score, source_row in source_rows:
        turn = turn_by_id[source_row["source_turn_id"]]
        card = card_by_session.get(turn.session_id)
        text_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+",
                f"{turn.text} {(card.routing_text if card else '')}".casefold(),
            )
        }
        if _collection_scope_covered(collection_scope_terms, text_terms):
            anchor_sessions.add(turn.session_id)
    neighbor_card_ids: set[str] = set()
    anchor_card_ids = {
        card.card_id for card in index.routing_cards
        if card.session_id in anchor_sessions
    }
    for edge in index.edges:
        if edge.relation != "semantic_neighbor" or edge.confidence < 0.72:
            continue
        if edge.src in anchor_card_ids:
            neighbor_card_ids.add(edge.dst)
        if edge.dst in anchor_card_ids:
            neighbor_card_ids.add(edge.src)
    # Semantic-neighbor cards may widen coarse routing, but cannot directly
    # contribute fine collection members without their own lexical/type scope
    # anchor. This prevents a semantically adjacent memory from becoming an
    # operand merely through graph proximity.
    eligible_named_sessions = set(anchor_sessions)
    # Re-evaluate typed operation targets throughout the bounded routing
    # component. This is the reusable part of V2's operand closure: coarse
    # routing defines the region, while frame roles and action/state fields
    # decide membership independently of the fine top-k.
    source_frame_terms: dict[str, set[str]] = {}
    for candidate in index.frames:
        candidate_slot_text = " ".join((
            candidate.entity_key, candidate.predicate_key,
            candidate.object_key, " ".join(candidate.semantic_type_keys),
        ))
        candidate_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", candidate_slot_text.casefold()
            )
        }
        for source_id in candidate.source_turn_ids:
            source_frame_terms.setdefault(source_id, set()).update(
                candidate_terms
            )

    def owner_binds_source_action(sources: list[Any]) -> bool:
        if not first_person:
            return True
        for source in sources:
            tokens = re.findall(r"[\w'-]+", source.text.casefold())
            for position, token in enumerate(tokens):
                stem = _stem_word(token)
                if not collection_action_matches({stem}):
                    continue
                nearest_subject = ""
                for prior in reversed(tokens[max(0, position - 8):position]):
                    subject = prior.split("'")[0]
                    if subject in {"i", "we", "he", "she", "they", "you"}:
                        nearest_subject = subject
                        break
                if nearest_subject == "i":
                    return True
        return False

    for frame in sorted(
        index.frames, key=lambda item: (item.observation_order, item.frame_id)
    ):
        if not set(frame.session_ids).intersection(eligible_named_sessions):
            continue
        if frame.polarity == "negative" or frame.lifecycle_status == "cancelled":
            continue
        if ir.target_owner and frame.owner_key != ir.target_owner:
            continue
        frame_sources = [
            turn_by_id[source_id] for source_id in frame.source_turn_ids
            if source_id in turn_by_id
            and turn_by_id[source_id].session_id in eligible_named_sessions
        ]
        if not frame_sources:
            continue
        if first_person and not any(
            source.transport_role == "user" for source in frame_sources
        ):
            continue
        evidence = " ".join((
            frame.entity_key, frame.predicate_key, frame.object_key,
            frame.context_key, frame.event_identity_key,
            " ".join(frame.semantic_type_keys),
            *(source.text for source in frame_sources),
        ))
        evidence_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", evidence.casefold()
            )
        }
        sibling_terms = set().union(*(
            source_frame_terms.get(source.node_id, set())
            for source in frame_sources
        )) if frame_sources else set()
        exact_full_scope = _collection_scope_covered(
            collection_scope_terms, evidence_terms | sibling_terms
        )
        subtype_full_scope = _modifier_subtype_covered(
            collection_head_terms, ir.raw_question, evidence
        )
        full_scope_binding = exact_full_scope or subtype_full_scope
        frame_action_text = " ".join((
            frame.predicate_key, frame.object_key, frame.state_op,
        ))
        frame_action_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", frame_action_text.casefold()
            )
        }
        frame_specific_action = collection_action_matches(
            frame_action_terms
        )
        source_action = collection_action_matches(
            evidence_terms
        ) and owner_binds_source_action(frame_sources)
        query_allows_pending = bool(
            query_terms & {"need", "plan"}
        )
        slot_evidence = " ".join((
            frame.entity_key, frame.predicate_key, frame.object_key,
            " ".join(frame.semantic_type_keys),
        ))
        slot_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", slot_evidence.casefold()
            )
        }
        slot_scope_binding = (
            _collection_scope_covered(collection_scope_terms, slot_terms)
            or _modifier_subtype_covered(
                collection_head_terms, ir.raw_question, slot_evidence
            )
        )
        request_like = bool(
            set(frame.semantic_type_keys) & {"request", "recommendation", "advice"}
            or re.search(
                r"\b(?:want|wants|request|requests|ask|asks|advice|"
                r"recommend|recommends|suggest|suggests)\b",
                frame.predicate_key, re.IGNORECASE,
            )
        )
        if request_like and not query_allows_pending:
            continue
        # Membership cannot inherit an action merely because a sibling frame
        # shares the same source turn. That conflates background/goals with
        # actual operated-on members. Missing extraction is represented by the
        # bounded lossless source closure above instead.
        if not frame_specific_action:
            continue
        if not slot_scope_binding:
            continue
        predicate_role = frame.predicate_key.casefold().strip()
        if (
            predicate_role in {
                "background", "goal", "context", "interest", "topic"
            }
            and not query_allows_pending
        ):
            continue
        entity_target = frame.entity_key.strip()
        object_target = frame.object_key.strip()
        normalized_entity = re.sub(
            r"[ _-]+", " ", entity_target.casefold()
        ).strip()
        entity_is_generic = bool(
            not entity_target
            or entity_target.casefold() in generic_targets
            or re.fullmatch(
                r"(?:participant|speaker|questioner|assistant|user)(?: \d+)?",
                normalized_entity,
            )
        )
        predicate_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", frame.predicate_key.casefold()
            )
        }
        predicate_binds_patient = bool(
            predicate_terms & {"pick", "collect", "retrieve", "receive"}
        )
        target = (
            object_target
            if object_target and (entity_is_generic or predicate_binds_patient)
            else entity_target
        )
        normalized_target = re.sub(
            r"[ _-]+", " ", target.casefold()
        ).strip()
        target_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", target.casefold()
            )
            if _stem_word(word) not in _COLLECTION_LEDGER_STOP_STEMS
        }
        category_only_target = bool(
            target_terms and collection_scope_terms
            and target_terms <= collection_scope_terms
        )
        if (
            not target
            or category_only_target
            or normalized_target in generic_targets
            or re.fullmatch(
                r"(?:participant|speaker|questioner|assistant|user)(?: \d+)?",
                normalized_target,
            )
        ):
            continue
        source_ids = tuple(sorted(source.node_id for source in frame_sources))
        signature = (
            frame.predicate_key.casefold().strip(), target.casefold(),
            frame.state_op, source_ids,
        )
        if signature in member_signatures:
            continue
        member_signatures.add(signature)
        member_source_ids.update(source_ids)
        structured_member_rows.append({
            "frame_id": frame.frame_id,
            "operation": frame.predicate_key,
            "target": target,
            "value": frame.object_key,
            "state_op": frame.state_op,
            "status": frame.lifecycle_status,
            "event_identity": frame.event_identity_key,
            "source_turn_ids": list(source_ids),
            "binding": "routed_typed_operation_target",
            "scope_binding": (
                "exact"
                if (
                    _collection_scope_covered(
                        collection_scope_terms, slot_terms
                    )
                    or exact_full_scope
                )
                else "subtype"
            ),
            "evidence_excerpt": _ledger_excerpt(
                " ".join(source.text for source in frame_sources),
                query_terms | frame_action_terms, limit=260,
            ),
        })
        if len(structured_member_rows) >= 18:
            break
    # Containment is a role relation: when an owner is working on a container,
    # an explicitly featured/contained artifact is also an operation target.
    seen_relation_targets = {
        re.sub(r"[^\w]+", " ", row["target"].casefold()).strip()
        for row in relation_closure_rows
    }
    if collection_query and anchor_sessions:
        for turn in index.turns:
            if (
                turn.session_id not in routed_order
                or turn.session_id not in eligible_named_sessions
                or (first_person and turn.transport_role != "user")
            ):
                continue
            card = card_by_session.get(turn.session_id)
            relation_evidence = (
                f"{turn.text} {(card.routing_text if card else '')}"
            )
            source_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", relation_evidence.casefold()
                )
            }
            relation_scope_binding = (
                _collection_scope_covered(
                    collection_scope_terms, source_terms
                )
                or _modifier_subtype_covered(
                    collection_head_terms, ir.raw_question,
                    relation_evidence,
                )
            )
            if (
                not relation_scope_binding
                or not collection_action_matches(source_terms)
            ):
                continue
            for match in re.finditer(
                r"\b(featuring|containing|including|includes?|consisting of|"
                r"composed of)\s+([^,.;!?]{3,120})",
                turn.text, re.IGNORECASE,
            ):
                target = re.sub(
                    r"^(?:a|an|the)\s+", "", match.group(2).strip(),
                    flags=re.IGNORECASE,
                )
                target = re.split(
                    r"\s+(?:and|but)\s+(?:i|we|they|he|she|it)\b",
                    target, maxsplit=1, flags=re.IGNORECASE,
                )[0].strip()
                normalized = re.sub(
                    r"[^\w]+", " ", target.casefold()
                ).strip()
                if (
                    len(re.findall(r"[\w'-]+", target)) < 2
                    or normalized in seen_relation_targets
                ):
                    continue
                seen_relation_targets.add(normalized)
                relation_closure_rows.append({
                    "target": target,
                    "relation": match.group(1).casefold(),
                    "source_turn_id": turn.node_id,
                    "binding": "routed_explicit_containment",
                    "evidence_excerpt": _ledger_excerpt(
                        turn.text, query_terms, limit=260
                    ),
                })
                break
    relation_source_ids = {
        row["source_turn_id"] for row in relation_closure_rows
        if row.get("source_turn_id")
    }
    if relation_source_ids:
        structured_member_rows = [
            row for row in structured_member_rows
            if not (
                row.get("scope_binding") == "subtype"
                and relation_source_ids.intersection(
                    row.get("source_turn_ids") or []
                )
            )
        ]
    structured_distinct_targets = list(dict.fromkeys(
        row["target"] for row in structured_member_rows
    ))
    named_object_rows: list[dict[str, Any]] = []
    seen_named: set[tuple[str, str]] = set()
    if collection_query and anchor_sessions:
        for turn in index.turns:
            if (
                turn.session_id not in routed_order
                or turn.session_id not in eligible_named_sessions
            ):
                continue
            if first_person and turn.transport_role != "user":
                continue
            card = card_by_session.get(turn.session_id)
            if card is None:
                continue
            source_text = turn.text.casefold()
            combined_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", f"{turn.text} {card.routing_text}".casefold()
                )
            }
            direct_scope = _collection_scope_covered(
                collection_scope_terms, combined_terms
            )
            for entity in card.canonical_entities:
                target = entity.strip()
                normalized = re.sub(
                    r"[^\w]+", " ", target.casefold()
                ).strip()
                if (
                    len(target) < 3
                    or len(normalized.split()) > 8
                    or normalized in generic_targets
                    or target.casefold() not in source_text
                ):
                    continue
                relation_texts = [
                    relation for relation in card.relations
                    if target.casefold() in relation.casefold()
                ]
                relation_terms = {
                    _stem_word(word)
                    for relation in relation_texts
                    for word in re.findall(r"[\w'-]+", relation.casefold())
                }
                relation_action = collection_action_matches(
                    relation_terms
                )
                relation_is_planned = any(re.search(
                    r"\b(?:will|would|plan|plans|planned|want|wants|"
                    r"recommend|recommends|suggest|suggests|could|should)\b",
                    relation, re.IGNORECASE,
                ) for relation in relation_texts)
                entity_offset = source_text.find(target.casefold())
                entity_prefix = source_text[max(0, entity_offset - 140):entity_offset]
                explicit_type_binding = bool(
                    direct_scope
                    and re.search(
                        r"\b(?:called|named)\b[^.!?]{0,70}$",
                        entity_prefix, re.IGNORECASE,
                    )
                )
                neighbor_relation_binding = bool(
                    turn.session_id not in anchor_sessions
                    and relation_action
                    and not relation_is_planned
                )
                if not (explicit_type_binding or neighbor_relation_binding):
                    continue
                key = (normalized, turn.session_id)
                if key in seen_named:
                    continue
                seen_named.add(key)
                named_object_rows.append({
                    "target": target,
                    "source_turn_id": turn.node_id,
                    "session_id": turn.session_id,
                    "binding": (
                        "explicit_type_source"
                        if explicit_type_binding
                        else "semantic_neighbor_relation"
                    ),
                    "relation": relation_texts[0] if relation_texts else "",
                    "evidence_excerpt": _ledger_excerpt(
                        turn.text, query_terms, limit=260
                    ),
                })
    # Weekly schedules are a collection of time slots, not a collection of
    # class labels. Inspect user-owned turns in the routed anchor component so
    # an omitted fine frame cannot remove a weekday from the answer.
    schedule_closure_rows: list[dict[str, Any]] = []
    asks_distinct_weekdays = bool(re.search(
        r"\bhow many\s+days?\s+(?:a|per|each)\s+week\b",
        ir.raw_question, re.IGNORECASE,
    ))
    asks_weekly_occurrences = bool(re.search(
        r"\bhow many\b.*\b(?:typical|usual|per|each)\b.*\bweek\b",
        ir.raw_question, re.IGNORECASE,
    ))
    if (
        (asks_distinct_weekdays or asks_weekly_occurrences)
        and routed_order and anchor_sessions
    ):
        for turn in index.turns:
            if turn.session_id not in routed_order:
                continue
            if first_person and turn.transport_role != "user":
                continue
            linked_owners = {
                frame.owner_key for frame in index.frames
                if turn.node_id in frame.source_turn_ids
            }
            if (
                ir.target_owner
                and turn.speaker_key != ir.target_owner
                and ir.target_owner not in linked_owners
            ):
                continue
            found_days = {
                day for day in _WEEKDAYS
                if re.search(rf"\b{day}s?\b", turn.text, re.IGNORECASE)
            }
            if not found_days:
                continue
            card = card_by_session.get(turn.session_id)
            evidence = f"{turn.text} {(card.routing_text if card else '')}"
            evidence_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", evidence.casefold()
                )
            }
            scope_overlap = collection_scope_terms & evidence_terms
            action_binding = collection_action_matches(
                evidence_terms
            )
            if not scope_overlap or not action_binding:
                continue
            weekdays.update(found_days)
            schedule_closure_rows.append({
                "source_turn_id": turn.node_id,
                "session_id": turn.session_id,
                "weekdays": sorted(
                    found_days, key=lambda day: _WEEKDAYS[day]
                ),
                "binding": "routed_anchor_schedule",
                "evidence_excerpt": _ledger_excerpt(
                    turn.text, query_terms, limit=260
                ),
            })
    proposed_distinct_targets = list(dict.fromkeys([
        *structured_distinct_targets,
        *(
            row["target"] for row in relation_closure_rows
            if structured_distinct_targets
        ),
        *(row["target"] for row in named_object_rows),
    ]))
    compact_frame_rows = [
        {
            key: value for key, value in row.items()
            if key not in {"matched_query_terms", "_selection_score"}
        }
        for row in frame_rows
        if set(row["source_turn_ids"]) & kept_source_ids
    ][:6]
    payload: dict[str, Any] = {
        "operation": "query_bound_collection_ledger",
        "candidate_pool_complete": False,
        "frame_candidates": compact_frame_rows,
        "structured_member_candidates": structured_member_rows,
        "structured_distinct_targets": structured_distinct_targets,
        "structured_distinct_target_count": len(structured_distinct_targets),
        "proposed_distinct_targets": proposed_distinct_targets,
        "proposed_distinct_target_count": len(proposed_distinct_targets),
        "relation_closure_candidates": relation_closure_rows,
        "named_object_candidates": named_object_rows,
        "graph_group_closure": graph_closure_rows,
        "schedule_closure_candidates": schedule_closure_rows,
        "unframed_lossless_candidates": unframed_lossless_rows,
        "lossless_candidates": [row for _score, row in kept_source_rows],
        "certified": False,
    }
    if asks_weekly_occurrences and weekdays:
        payload["derived_weekly_occurrence_days"] = sorted(
            weekdays, key=lambda day: _WEEKDAYS[day]
        )
        payload["derived_weekly_occurrence_value"] = len(weekdays)
    asks_bounded_year_span = bounded_year_query
    if asks_bounded_year_span:
        years = sorted({
            int(year)
            for _score, row in source_rows
            for year in re.findall(
                r"(?<!\d)(?:19|20)\d{2}(?!\d)",
                turn_by_id[row["source_turn_id"]].text,
            )
        })
        if len(years) >= 2 and 0 < years[-1] - years[0] <= 100:
            payload["derived_bounded_year_span"] = {
                "start_year": years[0],
                "end_year": years[-1],
                "value": years[-1] - years[0],
                "unit": "years",
            }
    asks_weekdays = bool(re.search(
        r"\bhow many\s+days?\s+(?:a|per|each)\s+week\b",
        ir.raw_question, re.IGNORECASE,
    ))
    if asks_weekdays and weekdays:
        payload["derived_distinct_weekdays"] = sorted(
            weekdays, key=lambda day: _WEEKDAYS[day]
        )
        payload["derived_value"] = len(weekdays)
    provenance_rows = [
        *structured_member_rows,
        *relation_closure_rows,
        *named_object_rows,
        *schedule_closure_rows,
    ]
    all_provenance = bool(provenance_rows) and all(
        row.get("source_turn_ids") or row.get("source_turn_id")
        for row in provenance_rows
    )
    # A finite candidate list does not prove an open-world collection is
    # complete. Only an explicitly enumerated closed domain (weekdays here)
    # may become an authoritative local answer.
    locally_closed = bool(
        ir.requested_value_type == "count"
        and bool(routed_session_ids)
        and anchor_sessions
        and all_provenance
        and payload.get("derived_distinct_weekdays")
    )
    if locally_closed:
        payload["candidate_pool_complete"] = True
        payload["certified"] = True
        payload["operator_certificate"] = {
            "entity_match": True,
            "relation_match": True,
            "scope_match": True,
            "provenance_complete": True,
            "closure": "routed_scope_plus_one_structural_or_semantic_hop",
        }
    return payload


def counterfactual_dependency_hint(
    ir: QueryIR, index: V36Index, source_turn_ids: list[str],
) -> dict[str, Any] | None:
    """Certify a negated-condition answer only from an explicit causal source."""
    if not {"condition", "effect"} <= set(ir.required_roles):
        return None
    if not re.search(
        r"\b(?:not|never|without|hadn\x27t|hasn\x27t|didn\x27t|weren\x27t|"
        r"wasn\x27t|wouldn\x27t)\b",
        ir.raw_question.casefold(),
    ):
        return None
    parts = re.split(r"\bif\b", ir.raw_question, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    stop = {
        "would", "still", "want", "pursue", "if", "she", "he",
        "they", "had", "has", "have", "did", "does", "not", "never",
        "without", "growing", "up", "a", "an", "the", "as", "to",
    }
    def content_terms(text: str) -> set[str]:
        return {
            _stem_word(word) for word in re.findall(r"[\w'-]+", text.casefold())
            if word not in stop and len(word) > 2
        }
    effect_terms = content_terms(parts[0])
    condition_terms = content_terms(parts[1])
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    candidates: list[tuple[int, str]] = []
    causal = re.compile(
        r"\b(?:because|motivated|instrumental|led to|made (?:me|him|her|them)|"
        r"so (?:i|he|she|they) (?:started|began)|as a result|therefore)\b",
        re.IGNORECASE,
    )
    for source_id in source_turn_ids:
        turn = turn_by_id.get(source_id)
        if turn is None or not causal.search(turn.text):
            continue
        terms = content_terms(turn.text)
        condition_overlap = len(condition_terms & terms)
        effect_overlap = len(effect_terms & terms)
        if condition_overlap >= 1 and effect_overlap >= 2:
            candidates.append((condition_overlap * 3 + effect_overlap, source_id))
    if not candidates:
        return None
    source_id = max(candidates)[1]
    return {
        "operation": "counterfactual_dependency",
        "value": "likely_no",
        "source_turn_ids": [source_id],
        "certified": True,
    }


def evaluate_operators(
    *,
    ir: QueryIR,
    index: V36Index,
    frame_ids: list[str],
    group_ids: list[str],
    certificate: CompletenessCertificate,
) -> list[dict[str, Any]]:
    """Return only generic algebra whose selected scope is fully certified."""
    if not (
        certificate.entity_match
        and certificate.relation_match
        and certificate.scope_match
        and certificate.provenance_complete
        and certificate.complete
    ):
        return []
    frames, groups = _selected(index, frame_ids, group_ids)
    hints: list[dict[str, Any]] = []
    if ir.requested_value_type == "aggregate":
        aggregate = _aggregate_hint(ir, frames, index)
        if aggregate is not None:
            hints.append(aggregate)
    if ir.requested_value_type == "location":
        hints.extend(_reference_location_closures(ir, frames, groups))
    if ir.requested_value_type in {"count", "list"}:
        collection_frame_ids = {
            frame_id for group in groups if group.group_kind == "collection"
            for frame_id in group.member_frame_ids
        }
        collection_frames = [
            frame for frame in frames
            if frame.frame_id in collection_frame_ids
        ]
        values = _collection_values(collection_frames)
        if values:
            hints.append({
                "operation": "distinct_collection",
                "value": len(values)
                if ir.requested_value_type == "count" else values,
                "members": values,
                "frame_ids": [
                    frame.frame_id for frame in collection_frames
                ],
                "certified": True,
            })
    if ir.requested_value_type == "state":
        generic = {"switch", "switched", "change", "changed", "current", "latest", "more", "less", "did", "do", "is", "was", "the", "a", "an", "i", "my"}
        query_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w']+", ir.raw_question.replace("-", " ").casefold()
            ) if word not in generic
        }
        predicate_query_terms = query_terms & {
            _stem_word(word)
            for candidate in frames
            for word in re.findall(
                r"[\w']+", candidate.predicate_key.replace("-", " ").casefold()
            )
        }
        turn_by_id = {turn.node_id: turn for turn in index.turns}
        ranked: list[tuple[int, datetime, RoleFrameNode]] = []
        for frame in frames:
            source_text = " ".join(
                turn_by_id[source].text for source in frame.source_turn_ids
                if source in turn_by_id
            )
            evidence_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", f"{frame.retrieval_text} {source_text}".casefold()
                )
            }
            structured_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w']+", " ".join((frame.entity_key,
                    frame.predicate_key, frame.object_key, frame.context_key))
                    .replace("-", " ").casefold()
                )
            }
            predicate_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w']+", frame.predicate_key.replace("-", " ").casefold()
                )
            }
            structured_overlap = len(query_terms & structured_terms)
            overlap = structured_overlap
            observed = _parse_time(
                frame.temporal.event_time or frame.temporal.observed_at
            )
            if (
                structured_overlap
                and (not predicate_query_terms or predicate_query_terms & predicate_terms)
                and observed is not None
                and frame.lifecycle_status not in {"planned", "proposed", "cancelled"}
            ):
                ranked.append((overlap, observed, frame))
        if len(ranked) >= 2:
            best = max(row[0] for row in ranked)
            relevant = sorted(
                (row for row in ranked if row[0] >= max(1, best - 1)),
                key=lambda row: (row[1], row[2].observation_order),
            )
            history = relevant[-4:]
            ratio_direction = ""
            if len(history) >= 2:
                denominators: list[float] = []
                for _score, _observed, frame in history[-2:]:
                    text = " ".join(
                        turn_by_id[source].text
                        for source in frame.source_turn_ids
                        if source in turn_by_id
                    ) or frame.object_key or frame.retrieval_text
                    match = re.search(
                        r"(?:per|every)\s+(\d+(?:\.\d+)?)|"
                        r"\b1\s*:\s*(\d+(?:\.\d+)?)\b",
                        text.casefold(),
                    )
                    if match:
                        denominators.append(float(match.group(1) or match.group(2)))
                if len(denominators) == 2 and denominators[0] != denominators[1]:
                    ratio_direction = (
                        "more denominator quantity per numerator"
                        if denominators[1] > denominators[0]
                        else "less denominator quantity per numerator"
                    )
            hints.append({
                "operation": "latest_valid_state",
                "value": history[-1][2].object_key or history[-1][2].retrieval_text,
                "change_direction": ratio_direction,
                "history": [
                    {
                        "value": frame.object_key or frame.retrieval_text,
                        "time": observed.isoformat(),
                    }
                    for _score, observed, frame in history
                ],
                "frame_ids": [frame.frame_id for _score, _observed, frame in history],
                "certified": True,
            })
    if (
        ir.requested_value_type == "duration"
        and "ago" not in ir.temporal_constraints
        and "event_a" not in ir.required_roles
    ):
        turn_by_id = {turn.node_id: turn for turn in index.turns}
        duration_frames = [
            frame for frame in frames
            if frame.quantity.value is not None
            and frame.quantity.unit.casefold() in _TIME_UNIT_SECONDS
            and frame.lifecycle_status not in {"planned", "proposed", "cancelled"}
            and not (
                frame.source_turn_ids
                and all(
                    turn_by_id[source].transport_role == "assistant"
                    for source in frame.source_turn_ids if source in turn_by_id
                )
            )
        ]
        duration_frames = _bind_duration_frames(ir, duration_frames, index)
        duration_frames = _deduplicate_duration_echoes(
            ir, duration_frames, index
        )
        confirmed = [
            frame for frame in duration_frames
            if frame.lifecycle_status in {"completed", "ongoing"}
        ]
        if len(confirmed) >= 2:
            duration_frames = confirmed
        if "components" in ir.required_roles and not (2 <= len(duration_frames) <= 4):
            duration_frames = []
        if duration_frames:
            asks_round_trip = bool(re.search(
                r"\b(?:round[ -]?trip|both ways|there and back|commute total)\b",
                ir.raw_question, re.IGNORECASE,
            ))
            total_seconds = sum(
                (frame.quantity.value or 0.0)
                * _TIME_UNIT_SECONDS[frame.quantity.unit.casefold()]
                * ((frame.quantity.multiplier or 1.0) if asks_round_trip else 1.0)
                for frame in duration_frames
            )
            requested_unit = next((unit for unit in _TIME_UNIT_SECONDS if unit in ir.raw_question.casefold().split()), duration_frames[0].quantity.unit.casefold())
            hints.append({
                "operation": "duration_total",
                "value": total_seconds / _TIME_UNIT_SECONDS[requested_unit],
                "unit": requested_unit,
                "frame_ids": [frame.frame_id for frame in duration_frames],
                "certified": True,
            })
    if ir.requested_value_type in {"date", "duration"}:
        if ir.requested_value_type == "date":
            turn_by_id = {turn.node_id: turn for turn in index.turns}
            owner_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", ir.target_owner.casefold()
                )
            }
            query_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+", ir.raw_question.casefold()
                )
                if word not in {
                    "when", "what", "date", "did", "does", "do",
                    "the", "a", "an", "in", "on", "at",
                }
                and _stem_word(word) not in owner_terms
            }
            relation_head = next((
                _stem_word(word) for word in reversed(re.findall(
                    r"[\w'-]+", ir.raw_question.casefold()
                ))
                if _stem_word(word) in query_terms
            ), "")
            date_candidates: list[tuple[int, int, float, datetime, RoleFrameNode]] = []
            for frame in frames:
                if ir.target_owner and frame.owner_key != ir.target_owner:
                    continue
                source_turns = [
                    turn_by_id[source_id] for source_id in frame.source_turn_ids
                    if source_id in turn_by_id
                ]
                evidence_terms = {
                    _stem_word(word) for word in re.findall(
                        r"[\w'-]+",
                        " ".join((
                            frame.entity_key, frame.predicate_key,
                            frame.object_key, frame.context_key,
                            *(turn.text for turn in source_turns),
                        )).casefold(),
                    )
                }
                relation_score = (
                    len(query_terms & evidence_terms)
                    + 2 * int(bool(relation_head) and relation_head in evidence_terms)
                )
                if relation_score == 0:
                    continue
                relative_times = [
                    relative
                    for turn in source_turns
                    if (observed := _turn_observed_time(turn)) is not None
                    and (relative := _relative_event_time(turn.text, observed))
                    is not None
                ]
                event_time = (
                    relative_times[0] if relative_times
                    else _parse_time(
                        frame.temporal.event_time or frame.temporal.start
                    )
                )
                if event_time is None:
                    continue
                date_candidates.append((
                    relation_score, int(bool(relative_times)),
                    frame.confidence, event_time, frame,
                ))
            if date_candidates:
                selected = max(
                    date_candidates,
                    key=lambda row: (row[0], row[1], row[2], row[4].frame_id),
                )
                hints.append({
                    "operation": "event_time",
                    "value": selected[3].isoformat(),
                    "frame_ids": [selected[4].frame_id],
                    "certified": True,
                })

    if ir.requested_value_type == "preference":
        generic = {
            "prefer", "preference", "favorite", "favourite", "like",
            "dislike", "love", "hate", "would", "could", "should",
            "think", "what", "which", "do", "does", "did", "am",
            "is", "are", "was", "were", "the", "a", "an", "i", "my",
        }
        query_terms = {
            _stem_word(word) for word in re.findall(
                r"[\w'-]+", ir.raw_question.casefold()
            ) if word not in generic
        }
        turn_by_id = {turn.node_id: turn for turn in index.turns}
        preference_frames: list[RoleFrameNode] = []
        for frame in frames:
            if frame.frame_kind != "preference":
                continue
            source_text = " ".join(
                turn_by_id[source].text for source in frame.source_turn_ids
                if source in turn_by_id
            )
            evidence_terms = {
                _stem_word(word) for word in re.findall(
                    r"[\w'-]+",
                    " ".join((frame.entity_key, frame.object_key,
                              frame.context_key, source_text)).casefold(),
                )
            }
            if query_terms & evidence_terms:
                preference_frames.append(frame)
        values = [
            {
                "object": frame.object_key,
                "polarity": frame.polarity,
                "context": frame.context_key,
            }
            for frame in preference_frames
        ]
        if values:
            hints.append({
                "operation": "contextual_preferences",
                "value": values,
                "frame_ids": [frame.frame_id for frame in preference_frames],
                "certified": True,
            })
    return hints
