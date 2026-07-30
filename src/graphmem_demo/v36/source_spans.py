from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from ..v3.action_semantics import action_families
from ..v3.build import canonical_key
from .schema import (
    QueryIR, RoleFrameNode, SourceSpanCandidate, SourceSpanClosure, TurnNodeV36,
)
from .dialogue_topology import infer_dialogue_topology

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_STOP = {"a","an","and","are","as","at","be","been","by","can","did","do","does","for","from","had","has","have","how","i","in","is","it","me","my","of","on","or","the","to","was","were","what","when","where","which","who","with","would","you","your"}
_GENERIC = {"all","answer","count","current","currently","different","each","fact","latest","many","memory","number","previous","recent","recently","remind","some","tell","total"}
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b|\b(?:today|yesterday|tomorrow|last|past|next|ago|before|after|january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
_NUMBER_RE = re.compile(r"(?:[$€£¥]\s*)?\b\d[\d,]*(?:\.\d+)?(?:\s*%)?\b")
_NEGATIVE_RE = re.compile(r"\b(?:not|never|no longer|without|dislike|hate|avoid|cancelled|canceled)\b", re.I)
_PLANNED_RE = re.compile(r"\b(?:plan|planning|planned|intend|intending|consider|considering|might|may|should|could|want|hope|thinking of|going to|will)\b", re.I)
_COMPLETED_RE = re.compile(r"\b(?:already|accepted|achieved|attended|bought|built|completed|earned|finished|fixed|flew|got|graduated|had|hired|launched|made|met|opened|participated|purchased|received|repaired|returned|serviced|signed|started|took|tried|used|visited|went|won|worked)\b", re.I)
_CANCELLED_RE = re.compile(r"\b(?:cancelled|canceled|abandoned|returned|removed|sold|no longer)\b", re.I)
_PREFERENCE_RE = re.compile(r"\b(?:like|liked|love|loved|prefer|preferred|enjoy|enjoyed|dislike|disliked|hate|hated|avoid)\b", re.I)
_TYPE_FAMILIES = {
    "device": {"aid", "device", "equipment", "instrument", "machine", "meter", "monitor", "system", "tracker"},
}


def _stem(token: str) -> str:
    value = token.casefold().strip("'")
    irregular = {"attendance":"attend","attended":"attend","baking":"bake","cooking":"cook","making":"make","bought":"buy","built":"build","did":"do","fixed":"fix","finished":"finish","got":"get","made":"make","met":"meet","paid":"pay","received":"receive","serviced":"service","took":"take","went":"go"}
    if value in irregular:
        return irregular[value]
    if value in {"anything", "everything", "nothing", "something"}:
        return value
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def _tokens(value: str) -> set[str]:
    return {_stem(token) for token in _WORD_RE.findall(value) if len(token) > 1 and token.casefold() not in _STOP}


def binding_tokens(value: str) -> set[str]:
    """Public normalizer shared by routing and lossless source binding."""
    return _tokens(value)


def fuzzy_term_overlap(query: set[str], evidence: set[str]) -> set[str]:
    """Conservative typo-tolerant overlap for question-time binding."""
    matched = query & evidence
    for term in query - matched:
        if len(term) < 5:
            continue
        if any(len(value) >= 5 and SequenceMatcher(None, term, value).ratio() >= 0.86 for value in evidence):
            matched.add(term)
    return matched


def query_binding_terms(ir: QueryIR) -> tuple[set[str], set[str]]:
    """Return normalized answer-target and relation terms for question-time binding."""
    question = ir.raw_question
    match = re.search(
        r"\bhow\s+many\s+(.+?)\s+"
        r"(?:am|are|did|do|does|had|has|have|is|was|were|will|would)\b"
        r"(?P<relation>.+?)(?:\?|$)", question, re.I,
    )
    if match:
        target = _tokens(match.group(1)) - {
            "amount", "different", "item", "kind", "number", "piece", "type",
        }
        return target, _tokens(match.group("relation")) - _GENERIC

    # Direct slot questions expose a target phrase before the auxiliary and
    # the requested relation after it. This prevents temporal words such as
    # “four weeks ago” from being mistaken for the answer entity.
    for pattern in (
        r"\b(?:what|which)\s+(.+?)\s+did\s+i\s+(.+?)(?:\?|$)",
        r"\bwhat\s+(?:was|is)\s+(?:the\s+)?(.+?)\s+i\s+(.+?)(?:\?|$)",
    ):
        direct = re.search(pattern, question, re.I)
        if direct:
            target = _tokens(direct.group(1)) - {
                "name", "time", "date", "kind", "type",
            }
            return target, _tokens(direct.group(2)) - _GENERIC

    mentioned = re.search(
        r"\bi\s+mentioned\s+(.+?)\s+"
        r"((?:a\s+couple\s+of|\w+)\s+(?:days?|weeks?|months?|years?)\s+ago|"
        r"last\s+\w+)", question, re.I,
    )
    if mentioned:
        target = _tokens(mentioned.group(1)) - {"something", "thing"}
        relation = {"mention", *_tokens(mentioned.group(2))}
        relation.update(action_families(question))
        return target, relation - _GENERIC

    received = re.search(
        r"\bi\s+(received|bought|got|acquired|visited|attended|used)\s+"
        r"(.+?)\s+((?:last|yesterday|today|\w+\s+ago)\b.+?)(?:\?|$)",
        question, re.I,
    )
    if received:
        return _tokens(received.group(2)), {
            _stem(received.group(1)), *_tokens(received.group(3)),
        } - _GENERIC

    relation = _tokens(ir.target_relation) - _GENERIC
    named = set()
    for value in [*ir.comparison_targets, *ir.operand_targets]:
        named.update(_tokens(value))
    if named:
        return named, relation - named
    return set(ir.target_entities) - _GENERIC, relation


def relative_target_date(text: str, observed_at: str | None) -> datetime | None:
    """Resolve common relative dates against the supplied conversation anchor."""
    if not observed_at:
        return None
    match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", observed_at)
    if match is None:
        return None
    observed = datetime(*(int(part) for part in match.groups()))
    lowered = text.casefold()
    if re.search(r"\btoday\b", lowered):
        return observed
    if re.search(r"\byesterday\b", lowered):
        return observed - timedelta(days=1)
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "couple": 2, "few": 3,
    }
    relative = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|a\s+couple(?:\s+of)?|(?:a\s+)?few)\s+"
        r"(days?|weeks?|months?|years?)\s+ago\b",
        lowered,
    )
    if relative:
        raw = relative.group(1)
        amount = int(raw) if raw.isdigit() else 2 if raw.startswith("a couple") else 3 if raw.endswith("few") else words[raw]
        unit = relative.group(2).rstrip("s")
        days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        return observed - timedelta(days=days)
    duration = re.search(
        r"\b(?:for\s+(?:the\s+)?(?:past|last)?|during\s+the\s+(?:past|last))\s*"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(hours?|days?|weeks?|months?|years?)\b",
        lowered,
    )
    if duration:
        raw = duration.group(1)
        amount = int(raw) if raw.isdigit() else words[raw]
        unit = duration.group(2).rstrip("s")
        days = amount * {"hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}[unit]
        return observed - timedelta(days=days)
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    weekday = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered)
    if weekday:
        target = weekdays[weekday.group(1)]
        delta = (observed.weekday() - target) % 7 or 7
        return observed - timedelta(days=delta)
    return None


def _span_windows(text: str) -> list[tuple[int, str]]:
    segments = [value.strip(" \t-*•") for value in re.split(r"(?<=[.!?])\s+|\n+", text) if value.strip(" \t-*•")]
    rows = []
    for index, segment in enumerate(segments):
        rows.append((index, segment))
        if index + 1 < len(segments):
            rows.append((index, f"{segment} {segments[index + 1]}"))
    return rows


def _identity_keys(text: str, target_terms: set[str]) -> list[str]:
    words = _WORD_RE.findall(text)
    keys = []
    for index, word in enumerate(words):
        if _stem(word) not in target_terms:
            continue
        key = canonical_key(" ".join(words[max(0, index - 3):index + 1]))
        if key and key not in keys:
            keys.append(key)
    return keys[:6]


def _roles(ir: QueryIR, *, target: bool, relation: bool, timed: bool, numbered: bool, preference: bool, role: str, lifecycle: str) -> set[str]:
    result = {"source"}
    if target:
        result.update({"entity","identity","fact"})
    if relation:
        result.update({"relation","event"})
    if target and relation:
        result.update({"member","members"})
    if timed:
        result.update({"time","times","event_time"})
    if numbered:
        result.update({"quantity","duration"})
    if preference:
        result.update({"preference","polarity","context"})
    if lifecycle in {"completed","planned","cancelled"}:
        result.update({"operations","status"})
    if lifecycle in {"completed","ongoing"}:
        result.add("current_state")
    if role == "user":
        result.update({"owner","prompt_turn"})
    elif role == "assistant":
        result.update({"reply_turn","reply_content"})
    if not ir.temporal_constraints or timed:
        result.add("scope")
    return result


def build_source_span_closure(
    ir: QueryIR, turns: list[TurnNodeV36], routed_session_ids: set[str],
    *, frames: list[RoleFrameNode] | None = None, question_date: str | None = None,
    target_session_hints: dict[str, str] | None = None,
    preferred_source_turn_ids: list[str] | None = None, max_candidates: int = 24,
) -> SourceSpanClosure:
    """Create a question-time relation-bound table over lossless source spans."""
    topology = infer_dialogue_topology(turns)
    target_terms, relation_terms = query_binding_terms(ir)
    expanded_target_terms = set(target_terms)
    for family, aliases in _TYPE_FAMILIES.items():
        if family in target_terms or target_terms & aliases:
            expanded_target_terms.update(aliases)
    requested_families = action_families(ir.raw_question)
    relative_target = relative_target_date(ir.raw_question, question_date)
    owner = canonical_key(ir.target_owner)
    candidates = []
    frames_by_source = defaultdict(list)
    for frame in frames or []:
        for source_id in frame.source_turn_ids:
            frames_by_source[source_id].append(frame)
    by_position = {(turn.session_id, turn.turn_index): turn for turn in turns}
    dialogue = bool({"prompt_turn","reply_turn","reply_content"} & set(ir.required_roles))
    for turn in turns:
        if turn.session_id not in routed_session_ids:
            continue
        if (
            not topology.peer_dialogue
            and turn.transport_role == "assistant"
            and not dialogue
        ):
            continue
        for span_index, text in _span_windows(turn.text):
            terms = _tokens(text)
            source_frames = frames_by_source.get(turn.node_id, [])
            frame_kinds = {frame.frame_kind for frame in source_frames}
            frame_text = " ".join(
                " ".join([
                    frame.entity_key, frame.predicate_key, frame.object_key,
                    frame.context_key, " ".join(frame.semantic_type_keys),
                ])
                for frame in source_frames
            )
            frame_terms = _tokens(frame_text)
            frame_type_terms = _tokens(" ".join(
                value for frame in source_frames
                for value in frame.semantic_type_keys
            ))
            span_target_overlap = fuzzy_term_overlap(expanded_target_terms, terms)
            type_target_overlap = fuzzy_term_overlap(expanded_target_terms, frame_type_terms)
            target_overlap = sorted(span_target_overlap | type_target_overlap)
            relation_overlap = sorted(
                relation_terms & (
                    terms | (frame_terms if target_overlap else set())
                )
            )
            frame_families = action_families(frame_text) if target_overlap else set()
            source_families = action_families(text) | frame_families
            if (
                "use" in requested_families
                and re.search(r"\b(?:i|we)\b.{0,180}\bwith\s+(?:my|our|the|a|an)\b", text, re.I)
            ):
                source_families.add("use")
            family_overlap = requested_families & source_families
            target_match = bool(target_overlap)
            relation_match = bool(family_overlap or relation_overlap or (not requested_families and len(relation_overlap) >= 2))
            if not target_match and not relation_match and relative_target is None:
                continue
            owner_match = not owner or turn.speaker_key == owner or canonical_key(turn.speaker) == owner or owner in terms
            if owner and not owner_match:
                continue
            frame_event = next((
                relative_target_date("today", frame.temporal.event_time)
                for frame in source_frames if frame.temporal.event_time
            ), None)
            source_event = (
                relative_target_date(text, turn.session_date)
                or frame_event
                or relative_target_date("today", turn.session_date)
            )
            temporal_distance = (
                abs((source_event.date() - relative_target.date()).days)
                if source_event is not None and relative_target is not None else None
            )
            if temporal_distance is not None and temporal_distance > 2:
                continue
            timed = bool(_DATE_RE.search(text)) or source_event is not None
            numbered = bool(_NUMBER_RE.search(text))
            explicit_completed = bool(_COMPLETED_RE.search(text))
            if _CANCELLED_RE.search(text):
                lifecycle = "cancelled"
            elif explicit_completed:
                lifecycle = "completed"
            elif _PLANNED_RE.search(text):
                lifecycle = "planned"
            else:
                lifecycle = "ongoing" if relation_match else "unknown"
            if lifecycle == "unknown" and any(
                frame.lifecycle_status != "unknown" for frame in source_frames
            ):
                lifecycle = max(
                    (frame.lifecycle_status for frame in source_frames
                     if frame.lifecycle_status != "unknown"),
                    key=lambda value: {
                        "cancelled": 4, "completed": 3, "ongoing": 2,
                        "planned": 1, "proposed": 0,
                    }.get(value, 0),
                )
            polarity = (
                "negative"
                if _NEGATIVE_RE.search(text)
                or any(frame.polarity == "negative" for frame in source_frames)
                else "positive"
            )
            roles = _roles(
                ir, target=target_match, relation=relation_match,
                timed=timed or any(frame.temporal.event_time for frame in source_frames),
                numbered=numbered or any(frame.quantity.value is not None for frame in source_frames),
                preference=bool(_PREFERENCE_RE.search(text)) or "preference" in frame_kinds,
                role=("user" if topology.peer_dialogue else turn.transport_role),
                lifecycle=lifecycle,
            )
            if "state" in frame_kinds:
                roles.update({"current_state", "status", "operations"})
            score = 2.4*len(target_overlap)+2.8*len(relation_overlap)+4.0*len(family_overlap)+1.5*float(target_match and relation_match)+0.8*float(timed and bool(ir.temporal_constraints))+2.0*float(temporal_distance is not None and temporal_distance <= 2)+0.7*float(numbered and ir.requested_value_type in {"count","aggregate","duration"})+0.8*float(owner_match)+0.4*float(turn.transport_role=="user")+1.5*float(explicit_completed)+0.4*float(lifecycle=="completed")-2.0*float(lifecycle=="planned" and "plan" not in _tokens(ir.raw_question))
            date_match = _DATE_RE.search(text)
            event_time_text = (
                source_event.date().isoformat() if source_event is not None
                else date_match.group(0) if date_match else ""
            )
            identities = _identity_keys(text, expanded_target_terms)
            for frame in source_frames:
                typed_terms = _tokens(
                    " ".join([frame.entity_key, *frame.semantic_type_keys])
                )
                if expanded_target_terms & typed_terms:
                    key = canonical_key(frame.entity_key or frame.object_key)
                    if key and key not in identities:
                        identities.append(key)
            candidates.append(SourceSpanCandidate(source_turn_id=turn.node_id, session_id=turn.session_id, span_index=span_index, text=text[:900], speaker_key=turn.speaker_key, transport_role=turn.transport_role, target_terms=target_overlap, relation_terms=relation_overlap, action_families=sorted(family_overlap), roles=sorted(roles), lifecycle_status=lifecycle, polarity=polarity, event_time_text=event_time_text, identity_keys=identities[:8], score=round(score,6)))

    best = {}
    for candidate in candidates:
        signature = (candidate.source_turn_id, tuple(candidate.identity_keys), " ".join(candidate.action_families or candidate.relation_terms))
        if signature not in best or candidate.score > best[signature].score:
            best[signature] = candidate
    candidates = sorted(best.values(), key=lambda row:(-row.score,row.session_id,row.span_index))
    if relative_target is not None:
        best_by_source = {}
        for candidate in candidates:
            best_by_source.setdefault(candidate.source_turn_id, candidate)
        candidates = list(best_by_source.values())
    if target_session_hints and ir.comparison_targets:
        hinted_sessions = set(target_session_hints.values())
        candidates = [row for row in candidates if row.session_id in hinted_sessions]
    selected = []
    candidate_by_source = {row.source_turn_id: row for row in candidates}
    for source_id in preferred_source_turn_ids or []:
        candidate = candidate_by_source.get(source_id)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    if ir.requested_value_type in {"count","list"} or "members" in ir.required_roles:
        per_session = defaultdict(int)
        for candidate in candidates:
            if not {"member","members"} & set(candidate.roles) or per_session[candidate.session_id] >= 4:
                continue
            if candidate not in selected:
                selected.append(candidate)
            per_session[candidate.session_id] += 1
            if len(selected) >= 16:
                break
    if ir.temporal_constraints:
        per_session = defaultdict(int)
        for candidate in candidates:
            if not candidate.event_time_text or per_session[candidate.session_id] >= 2:
                continue
            if candidate not in selected:
                selected.append(candidate)
            per_session[candidate.session_id] += 1
            if len(selected) >= 16:
                break
    if ir.requested_value_type == "preference":
        for candidate in candidates:
            if "preference" not in candidate.roles:
                continue
            selected.append(candidate)
            if len(selected) >= 8:
                break
    if ir.requested_value_type == "state" or "current_state" in ir.required_roles:
        for candidate in candidates:
            if not {"current_state", "status", "operations"} & set(candidate.roles):
                continue
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= 8:
                break
    support = {}
    support_targets = [*ir.comparison_targets, *ir.operand_targets]
    support_token_sets = [_tokens(target) for target in support_targets]
    shared_support_terms = (
        set.intersection(*support_token_sets)
        if len(support_token_sets) >= 2 else set()
    )
    for target, target_tokens in zip(support_targets, support_token_sets):
        distinctive = target_tokens - shared_support_terms or target_tokens
        hinted_session = (target_session_hints or {}).get(target)
        matches = [
            candidate for candidate in candidates
            if (not hinted_session or candidate.session_id == hinted_session)
            and fuzzy_term_overlap(distinctive, _tokens(candidate.text))
        ][:3]
        support[target] = [candidate.source_turn_id for candidate in matches]
        if matches and matches[0] not in selected:
            selected.append(matches[0])
    present = {role for candidate in selected for role in candidate.roles}
    missing = set(ir.required_roles) - present
    while missing and len(selected) < max_candidates:
        options = [candidate for candidate in candidates if candidate not in selected]
        if not options:
            break
        candidate = max(options,key=lambda row:(len(missing & set(row.roles)),row.score))
        if not missing & set(candidate.roles):
            break
        selected.append(candidate); present.update(candidate.roles); missing=set(ir.required_roles)-present
    selected_ids = {candidate.source_turn_id for candidate in selected}
    if dialogue:
        for candidate in list(selected):
            turn = next((item for item in turns if item.node_id==candidate.source_turn_id),None)
            if turn is None:
                continue
            neighbor = by_position.get((turn.session_id,turn.turn_index+(1 if turn.transport_role=="user" else -1)))
            if neighbor is not None:
                selected_ids.add(neighbor.node_id)
    present = {role for candidate in selected for role in candidate.roles}
    missing_roles = sorted(set(ir.required_roles)-present)
    ordered_ids = [candidate.source_turn_id for candidate in selected]
    ordered_ids.extend(sorted(selected_ids-set(ordered_ids)))
    return SourceSpanClosure(candidates=selected[:max_candidates],selected_source_turn_ids=list(dict.fromkeys(ordered_ids)),present_roles=sorted(present),missing_roles=missing_roles,target_support=support,complete=not missing_roles and all(support.values()))
