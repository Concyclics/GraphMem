from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable

from ..clients import cosine_similarity
from ..models import QuestionCase
from ..v3.build import canonical_key
from .schema import (
    GRAPHMEM_V36_SCHEMA,
    CoverageEntry,
    EvidenceGroup,
    GraphEdgeV36,
    QuantityValue,
    RoleFrameNode,
    RoutingCard,
    StateChainV36,
    StateVersionV36,
    TemporalValue,
    TurnNodeV36,
    V36Index,
)


V36_PROMPT_VERSION = "graphmem_v36_role_frame_extract_20260729b"
V36_BUILD_VERSION = "graphmem_v36_evidence_group_build_20260729e"

_FRAME_KINDS = {
    "fact", "event", "state", "preference", "quantity", "dialogue_answer",
}
_POLARITIES = {"positive", "negative", "unknown"}
_MODALITIES = {"asserted", "planned", "possible", "conditional", "unknown"}
_LIFECYCLES = {
    "proposed", "planned", "ongoing", "completed", "cancelled", "unknown",
}
_STATE_OPS = {
    "set", "add", "remove", "increment", "decrement", "cancel", "complete",
    "none",
}
_COVERAGE_CLASSES = {
    "memory_frame", "dialogue_context", "non_durable", "boilerplate",
    "lossless_only",
}
_ALLOWED_RELATIONS = {
    "source", "next_turn", "dialogue_pair", "reference", "same_event",
    "state_transition", "collection_member", "temporal_endpoint", "contrast",
    "routing_contains", "semantic_neighbor",
}
_FORBIDDEN_RELATIONS = {
    "participant", "temporal_scope", "episode_member", "theme_member",
    "operand_projection", "event_frame_member",
}
_NUMBER_WORDS = {"a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0, "eleven": 11.0, "twelve": 12.0}
_QUANTITY_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?|percent|percentage|dollars?|euros?|pounds?)"
    r"(?P<half>\s+and\s+a\s+half)?\b", re.IGNORECASE,
)
_CURRENCY_QUANTITY_RE = re.compile(
    r"(?P<unit>[$€£¥])\s*(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)
_COMPLETED_VALUE_RE = re.compile(
    r"\b(?:bought|purchased|paid|spent|cost|got|received|booked|stayed|finished|completed)\b",
    re.IGNORECASE,
)
_PLANNED_VALUE_RE = re.compile(
    r"\b(?:plan|planning|budget|looking|recommend|suggest|consider|might|could|would|next)\b",
    re.IGNORECASE,
)

_QUESTION_CUE = re.compile(
    r"\?|^(?:who|what|when|where|which|why|how|can|could|would|will|do|does|"
    r"did|is|are|was|were|have|has|should)\b",
    re.IGNORECASE,
)
_REQUEST_CUE = re.compile(
    r"\b(?:please|tell me|show me|give me|recommend|suggest|help me|"
    r"could you|would you|can you)\b",
    re.IGNORECASE,
)


def _is_turn_alias(value: Any) -> bool:
    return bool(re.fullmatch(r"T\d+", str(value or "").strip(), re.IGNORECASE))


def _clean_temporal(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if not text or text.casefold() in {"none", "unknown"} or _is_turn_alias(text) else text


def _enum(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in allowed else default


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        str(item).strip() for item in value if str(item).strip()
    ))


def _routing_items(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            text = " ".join(str(part).strip() for part in item if str(part).strip())
        elif isinstance(item, dict):
            text = " ".join(str(part).strip() for part in item.values() if str(part).strip())
        else:
            text = str(item).strip()
        if text and text not in rows:
            rows.append(text)
    return rows


def _augment_explicit_quantities(
    frames: list[RoleFrameNode], *, question_id: str, session_id: str,
    turns: list[TurnNodeV36],
) -> None:
    normalize_unit = lambda value: value.casefold().removesuffix("s") if value.casefold() not in {"percent", "percentage"} else "percent"
    existing = {
        (source, frame.quantity.value, normalize_unit(frame.quantity.unit))
        for frame in frames for source in frame.source_turn_ids
        if frame.quantity.value is not None
    }
    existing_value = {
        (source, frame.quantity.value)
        for frame in frames for source in frame.source_turn_ids
        if frame.quantity.value is not None
    }
    turn_by_id = {turn.node_id: turn for turn in turns}

    # Recover a missing scalar inside an otherwise grounded quantity frame.
    # This is deliberately limited to user-authored currency assertions with a
    # single unambiguous value in the cited source.
    for frame in frames:
        if frame.frame_kind != "quantity" or frame.quantity.value is not None:
            continue
        source_turns = [
            turn_by_id[source] for source in frame.source_turn_ids
            if source in turn_by_id and turn_by_id[source].transport_role == "user"
        ]
        matches = [
            match for turn in source_turns
            for match in _CURRENCY_QUANTITY_RE.finditer(turn.text)
        ]
        if len(matches) != 1:
            continue
        match = matches[0]
        frame.quantity.value = _float(match.group("value").replace(",", ""))
        rate = frame.quantity.unit.strip()
        frame.quantity.unit = (
            f"{match.group('unit')} {rate}" if rate and match.group("unit") not in rate
            else match.group("unit")
        )
        for source in frame.source_turn_ids:
            existing.add((source, frame.quantity.value, normalize_unit(frame.quantity.unit)))
            existing_value.add((source, frame.quantity.value))

    patterns = [(_QUANTITY_RE, False), (_CURRENCY_QUANTITY_RE, True)]
    for turn in turns:
        for pattern, currency in patterns:
            if currency and turn.transport_role != "user":
                continue
            for match in pattern.finditer(turn.text):
                raw_value = match.group("value").casefold().replace(",", "")
                value = _float(raw_value) if raw_value not in _NUMBER_WORDS else _NUMBER_WORDS[raw_value]
                if value is None:
                    continue
                if not currency and match.groupdict().get("half"):
                    value += 0.5
                unit = match.group("unit")
                if currency and (turn.node_id, value) in existing_value:
                    continue
                key = (turn.node_id, value, normalize_unit(unit))
                if key in existing:
                    continue
                start, end = max(0, match.start() - 60), min(len(turn.text), match.end() + 60)
                context_text = turn.text[start:end].strip()
                completed = (
                    turn.transport_role == "user"
                    and bool(_COMPLETED_VALUE_RE.search(context_text))
                    and not bool(_PLANNED_VALUE_RE.search(context_text))
                )
                frames.append(RoleFrameNode(
                    frame_id=f"{question_id}:{session_id}:frame:{len(frames)}",
                    question_id=question_id, session_ids=[session_id], frame_kind="quantity",
                    owner_key=turn.speaker_key, entity_key=turn.speaker_key,
                    predicate_key="mentioned quantity", object_key=canonical_key(match.group(0)),
                    quantity=QuantityValue(value=value, unit=unit),
                    temporal=TemporalValue(observed_at=turn.session_date),
                    lifecycle_status="completed" if completed else "unknown",
                    semantic_type_keys=["explicit quantity"], source_turn_ids=[turn.node_id],
                    confidence=0.99, retrieval_text=f"{turn.speaker} | explicit quantity | {match.group(0)} | {context_text}",
                ))
                existing.add(key)
                existing_value.add((turn.node_id, value))


def _lossless_role_frame(
    *,
    question_id: str,
    session_id: str,
    turn: TurnNodeV36,
    frame_index: int,
) -> RoleFrameNode:
    """Preserve a model-declared durable turn without inventing semantics.

    This deliberately weak, generic frame gives retrieval and dialogue-pair
    construction a provenance-complete object while retaining the exact turn
    text. Unknown state and lifecycle fields keep it out of state operators.
    """
    return RoleFrameNode(
        frame_id=f"{question_id}:{session_id}:frame:{frame_index}",
        question_id=question_id,
        session_ids=[session_id],
        frame_kind="fact",
        owner_key=turn.speaker_key,
        entity_key=turn.speaker_key,
        predicate_key="stated",
        object_key="",
        polarity="unknown",
        modality="asserted",
        lifecycle_status="unknown",
        state_op="none",
        temporal=TemporalValue(observed_at=turn.session_date),
        semantic_type_keys=["lossless fallback"],
        source_turn_ids=[turn.node_id],
        confidence=0.55,
        retrieval_text=f"{turn.speaker} | stated | {turn.text}",
        coverage_mask=[],
        observation_order=frame_index,
    )


def _speaker(message: dict[str, Any], role: str) -> str:
    explicit = str(message.get("speaker") or "").strip()
    if explicit:
        return explicit
    return "participant_1" if role == "user" else "participant_2"


def build_turn_nodes(case: QuestionCase) -> list[TurnNodeV36]:
    turns: list[TurnNodeV36] = []
    seen: dict[str, int] = defaultdict(int)
    allocated: set[str] = set()
    for original_session_id, session_date, messages in zip(
        case.haystack_session_ids, case.haystack_dates, case.haystack_sessions
    ):
        seen[original_session_id] += 1
        occurrence = seen[original_session_id]
        session_id = (
            original_session_id if occurrence == 1
            else f"{original_session_id}__occ{occurrence}"
        )
        while session_id in allocated:
            occurrence += 1
            session_id = f"{original_session_id}__occ{occurrence}"
        allocated.add(session_id)
        for turn_index, message in enumerate(messages):
            role = str(message.get("role") or "unknown").casefold()
            speaker = _speaker(message, role)
            listener = str(message.get("listener") or "").strip()
            text = str(message.get("content") or "").strip()
            turns.append(
                TurnNodeV36(
                    node_id=(
                        f"{case.question_id}:{session_id}:turn:{turn_index}"
                    ),
                    question_id=case.question_id,
                    session_id=session_id,
                    session_date=session_date,
                    turn_index=turn_index,
                    speaker=speaker,
                    speaker_key=canonical_key(speaker) or speaker.casefold(),
                    listener=listener,
                    transport_role=role,
                    text=text,
                    retrieval_text=" | ".join(filter(None, (
                        f"speaker {speaker}",
                        f"listener {listener}" if listener else "",
                        text,
                    ))),
                )
            )
    return turns


def _verbose_session_extraction_messages(
    session_id: str,
    session_date: str | None,
    turns: list[TurnNodeV36],
) -> list[dict[str, str]]:
    frame_schema = [
        "kind", "owner", "entity", "predicate", "object", "context",
        "polarity", "modality", "lifecycle", "state_op",
        "quantity_or_null", "unit", "multiplier_or_null",
        "event_time", "start", "end", "time_precision", "time_anchor_source",
        "event_identity", ["semantic_types"], ["source_turn_ids"],
        "confidence",
    ]
    payload = {
        "session_id": session_id,
        "session_date": session_date,
        "frame_schema": frame_schema,
        "routing_card_schema": {
            "entities": [], "relations": [], "events": [], "current_states": [],
            "time_range": "",
        },
        "coverage_schema": [["source_turn_id", "coverage_class"]],
        "turns": [
            [
                turn.node_id, turn.speaker, turn.listener, turn.transport_role,
                turn.text,
            ]
            for turn in turns
        ],
    }
    system = (
        "Build a question-independent, role-neutral long-term memory index from "
        "the conversation. Return JSON with exactly frames, routing_card, and "
        "coverage. Frames are positional arrays matching frame_schema. Do not "
        "treat transport user/assistant roles as fact ownership: use the named "
        "speaker and the entity the statement is about. Preserve exact names, "
        "answer text, lists, numbers, units, negation, uncertainty, preferences, "
        "plans, completion and cancellation. Separate collection items. Anchor "
        "relative time to the session date and cite source turn IDs verbatim. "
        "Use dialogue_answer when a reply itself supplies a requested name, "
        "number, list, table value, explanation or decision. Use one semantic "
        "frame per durable proposition; do not duplicate a proposition as claim, "
        "event and operand. Generic acknowledgement and boilerplate need no "
        "frame, but every input turn must have one coverage row. Frame sources "
        "are the sole provenance mapping; never repeat frame indexes in coverage. Coverage class "
        "is memory_frame, dialogue_context, non_durable, or boilerplate. There is "
        "no fixed frame count: cover all durable memory before adding optional "
        "detail. The routing card must be a compact 100-180 token route, never a "
        "transcript or evaluation-facing answer. Semantic types describe source "
        "meaning only. Do not mention datasets, evaluation labels, test topics, "
        "questions outside this conversation, or infer unsaid facts. JSON only."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]



def _compact_extraction_text(text: str, max_chars: int = 1800) -> str:
    """Bound the semantic extraction view while the TurnNode stays lossless."""
    if len(text) <= max_chars:
        return text
    segments = [
        item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", text)
        if item.strip()
    ]
    if not segments:
        return text[: max_chars // 2] + " … " + text[-max_chars // 2 :]
    cue = re.compile(
        r"\b(?:not|never|prefer|favorite|dislike|plan(?:ned|ning)?|"
        r"completed?|cancel(?:led|ed)?|bought|paid|spent|started?|finished|"
        r"now|currently|ago|yesterday|today|tomorrow)\b|"
        r"(?:[$€£¥]|\b\d+(?:\.\d+)?\b)",
        re.IGNORECASE,
    )
    scores = []
    for position, segment in enumerate(segments):
        boundary = 3 if position in {0, len(segments) - 1} else 0
        scores.append((boundary + min(3, len(cue.findall(segment))), position))
    selected = {0, len(segments) - 1}
    used = sum(len(segments[position]) + 1 for position in selected)
    for _score, position in sorted(scores, key=lambda row: (-row[0], row[1])):
        if position in selected:
            continue
        cost = len(segments[position]) + 1
        if used + cost > max_chars - 24:
            continue
        selected.add(position)
        used += cost
    compact = " ".join(segments[position] for position in sorted(selected))
    if len(compact) > max_chars:
        compact = compact[: max_chars // 2] + " … " + compact[-max_chars // 2 :]
    return compact


def session_extraction_messages(
    session_id: str, session_date: str | None, turns: list[TurnNodeV36],
) -> list[dict[str, str]]:
    """Compact lossless protocol: short turn aliases avoid repeated global IDs."""
    compact_turns = [
        _compact_extraction_text(turn.text) for turn in turns
    ]
    frame_budget = min(40, max(6, (sum(len(text) for text in compact_turns) + 899) // 900))
    payload = {
        "s": session_id, "d": session_date, "B": frame_budget,
        "F": {
            "kind": "fact|event|state|preference|quantity|dialogue_answer",
            "owner": "", "entity": "", "predicate": "", "object": "",
            "context": "", "polarity": "positive|negative|unknown",
            "modality": "asserted|planned|possible|conditional|unknown",
            "lifecycle": "proposed|planned|ongoing|completed|cancelled|unknown",
            "op": "set|add|remove|increment|decrement|cancel|complete|none",
            "quantity": None, "unit": "", "multiplier": None,
            "event_time": None, "start": None, "end": None,
            "precision": "unknown", "anchor": None,
            "event_identity": "", "semantic_types": [],
            "sources": ["Tn"], "confidence": 0.0,
        },
        "T": [[f"T{index}", turn.speaker, turn.listener, compact_turns[index]] for index, turn in enumerate(turns)],
    }
    system = (
        "Create a question-independent role-frame memory index. Return JSON "
        "{frames:[F objects],routing_card:{entities,relations,events,current_states,time_range},coverage:[[Tn,class]]}. Every frame must be a JSON object using the exact F field names; never emit positional arrays, prepend a turn ID, rename fields, or omit sources. Frame sources are the only turn-to-frame mapping; never put frame indexes in coverage. "
        "Use named speakers, not transport roles. Entity is the real-world subject, never a frame type. Preserve exact answers, names, quantities, units, negation, uncertainty, preferences, plans, completion, cancellation and dates. Emit a separate quantity frame for every distinct measure (for example an item count and an elapsed duration); put its number and unit only in quantity/unit, not context. "
        "Split list items only for a durable set/inventory or later per-item operations; otherwise keep an exact explanatory/recommendation list in one dialogue_answer frame plus its lossless source. "
        "Frame kind: fact/event/state/preference/quantity/dialogue_answer. Lifecycle: proposed/planned/ongoing/completed/cancelled/unknown. "
        "Op: set/add/remove/increment/decrement/cancel/complete/none. Cite only supplied Tn aliases in the sources array. Do not duplicate one proposition. "
        "Every turn needs coverage class memory_frame/dialogue_context/non_durable/boilerplate. Cover all durable content; omit generic boilerplate. "
        "B is an adaptive maximum frame count: never emit more than B frames. "
        "Keep object, context, event_identity and each routing item under 160 characters; cite the lossless source instead of copying long prose, tables, or lists. "
        "Card is a compact route, not evidence. Infer nothing and return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def prompt_hash() -> str:
    messages = session_extraction_messages("s", "2026-01-01", [])
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _partial_array_items(text: str, name: str) -> list[Any]:
    marker = re.search(rf"\"{re.escape(name)}\"\s*:\s*\[", text)
    if marker is None:
        return []
    decoder = json.JSONDecoder()
    position = marker.end()
    items: list[Any] = []
    while position < len(text):
        while position < len(text) and text[position] in " \r\n\t,":
            position += 1
        if position >= len(text) or text[position] == "]":
            break
        try:
            value, consumed = decoder.raw_decode(text[position:])
        except Exception:
            break
        if isinstance(value, (list, dict)):
            items.append(value)
        position += consumed
    return items


def _json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text)
        return (value, None) if isinstance(value, dict) else (None, "invalid_json")
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    return value, "json_wrapper_salvaged"
            except Exception:
                pass
    # Recover complete top-level arrays from a truncated response.
    recovered: dict[str, Any] = {}
    for name in ("frames", "coverage"):
        marker = re.search(rf'"{name}"\s*:\s*', text)
        if marker is None:
            continue
        start = text.find("[", marker.end())
        if start < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except Exception:
            value = _partial_array_items(text, name)
        if isinstance(value, list) and value:
            recovered[name] = value
    return (
        (recovered, "partial_json_salvaged")
        if recovered.get("frames") else (None, "invalid_json")
    )


def _fallback(
    *,
    question_id: str,
    session_id: str,
    turns: list[TurnNodeV36],
) -> tuple[list[RoleFrameNode], RoutingCard, list[CoverageEntry]]:
    frames: list[RoleFrameNode] = []
    # Invalid extraction falls back to lossless turns only; never manufacture
    # duplicate low-confidence fact nodes from the transcript.
    card = _routing_card(
        question_id=question_id,
        session_id=session_id,
        turns=turns,
        frames=frames,
        payload={},
    )
    coverage = [CoverageEntry(turn.node_id, "lossless_only", []) for turn in turns]
    return frames, card, coverage


def _routing_card(
    *,
    question_id: str,
    session_id: str,
    turns: list[TurnNodeV36],
    frames: list[RoleFrameNode],
    payload: dict[str, Any],
) -> RoutingCard:
    entities = _routing_items(payload.get("entities"))[:24]
    relations = _routing_items(payload.get("relations"))[:16]
    events = _routing_items(payload.get("events"))[:12]
    states = _routing_items(payload.get("current_states"))[:12]
    if not entities:
        entities = list(dict.fromkeys(
            value for frame in frames
            for value in (frame.owner_key, frame.entity_key, frame.object_key)
            if value
        ))[:24]
    if not relations:
        relations = list(dict.fromkeys(
            frame.predicate_key for frame in frames if frame.predicate_key
        ))[:16]
    if not events:
        events = [
            frame.retrieval_text for frame in frames
            if frame.frame_kind == "event"
        ][:8]
    if not states:
        states = [
            frame.retrieval_text for frame in frames
            if frame.frame_kind in {"state", "preference"}
        ][:8]
    raw_time_range = payload.get("time_range")
    if isinstance(raw_time_range, dict):
        time_range = " to ".join(str(value).strip() for value in raw_time_range.values() if str(value).strip())
    elif isinstance(raw_time_range, (list, tuple)):
        time_range = " to ".join(str(value).strip() for value in raw_time_range if str(value).strip())
    else:
        time_range = str(raw_time_range or "").strip()
    if not time_range:
        dates = [turn.session_date for turn in turns if turn.session_date]
        time_range = " to ".join([min(dates), max(dates)]) if dates else "unknown"
    sections = [
        "entities: " + ", ".join(entities),
        "relations: " + ", ".join(relations),
        "events: " + "; ".join(events),
        "current states: " + "; ".join(states),
        f"time: {time_range}",
    ]
    # Bound card size deterministically; the source pointers remain lossless.
    routing_text = " | ".join(section for section in sections if section)
    if not frames and not any((entities, relations, events, states)):
        excerpts = " ".join(turn.text for turn in turns if turn.text)
        routing_text = f"speakers: {', '.join(dict.fromkeys(turn.speaker_key for turn in turns))} | lossless route: {excerpts[:650]}"
    routing_text = routing_text[:720]
    return RoutingCard(
        card_id=f"{question_id}:{session_id}:card",
        question_id=question_id,
        session_id=session_id,
        speaker_keys=list(dict.fromkeys(turn.speaker_key for turn in turns)),
        canonical_entities=entities,
        relations=relations,
        key_events=events,
        current_states=states,
        time_range=time_range,
        frame_ids=[frame.frame_id for frame in frames],
        turn_ids=[turn.node_id for turn in turns],
        routing_text=routing_text,
    )


_ROUTING_VIEW_STOP = {
    "participant", "user", "assistant", "person", "thing", "item",
    "state", "fact", "event", "current", "reported", "said",
}


def _attach_routing_relations(
    frames: list[RoleFrameNode], card: RoutingCard,
) -> None:
    """Add a compact same-session routing view without creating duplicate nodes."""
    relation_rows = []
    for position, relation in enumerate(card.relations):
        tokens = {
            canonical_key(word)
            for word in re.findall(r"[\w'-]+", relation)
            if len(canonical_key(word)) >= 3
            and canonical_key(word) not in _ROUTING_VIEW_STOP
        }
        relation_rows.append((position, relation, tokens))
    for frame in frames:
        anchor_text = " ".join((
            frame.entity_key, frame.object_key, frame.predicate_key,
            frame.event_identity_key,
        ))
        anchors = {
            canonical_key(word)
            for word in re.findall(r"[\w'-]+", anchor_text)
            if len(canonical_key(word)) >= 3
            and canonical_key(word) not in _ROUTING_VIEW_STOP
        }
        if not anchors:
            continue
        matches = sorted(
            (
                (len(anchors & tokens), position, relation)
                for position, relation, tokens in relation_rows
                if anchors & tokens
            ),
            key=lambda row: (-row[0], row[1]),
        )[:2]
        if not matches:
            continue
        suffix = "; ".join(row[2] for row in matches)[:280]
        frame.retrieval_text = (
            f"{frame.retrieval_text} | local routing relation: {suffix}"
        )


def parse_session_extraction(
    text: str,
    *,
    question_id: str,
    session_id: str,
    session_date: str | None,
    turns: list[TurnNodeV36],
) -> tuple[
    list[RoleFrameNode], RoutingCard, list[CoverageEntry], str | None
]:
    payload, parse_status = _json_object(text)
    if payload is None:
        frames, card, coverage = _fallback(
            question_id=question_id, session_id=session_id, turns=turns
        )
        return frames, card, coverage, "invalid_json"
    valid_turn_ids = {turn.node_id for turn in turns}
    source_ref_map = {turn.node_id: turn.node_id for turn in turns}
    source_ref_map.update({f"T{index}": turn.node_id for index, turn in enumerate(turns)})
    source_ref_map.update({f"t{index}": turn.node_id for index, turn in enumerate(turns)})
    source_ref_map.update({str(index): turn.node_id for index, turn in enumerate(turns)})
    frames: list[RoleFrameNode] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in payload.get("frames", []):
        if isinstance(raw, dict):
            raw_quantity = raw.get("quantity")
            quantity_payload = raw_quantity if isinstance(raw_quantity, dict) else {}
            values = [
                raw.get("kind") or raw.get("frame_kind") or raw.get("type"),
                raw.get("owner") or raw.get("owner_key") or raw.get("subject"),
                raw.get("entity") or raw.get("entity_key"),
                raw.get("predicate") or raw.get("relation"),
                raw.get("object") or raw.get("object_key") or raw.get("value"),
                raw.get("context") or raw.get("context_key"),
                raw.get("polarity"), raw.get("modality"),
                raw.get("lifecycle") or raw.get("lifecycle_status") or raw.get("status"),
                raw.get("op") or raw.get("state_op"),
                quantity_payload.get("value") if quantity_payload else raw_quantity,
                raw.get("unit") or quantity_payload.get("unit"),
                raw.get("multiplier") or quantity_payload.get("multiplier"),
                raw.get("event_time"), raw.get("start"), raw.get("end"),
                raw.get("precision") or raw.get("time_precision"),
                raw.get("anchor") or raw.get("time_anchor_source"),
                raw.get("event_identity") or raw.get("event_identity_key"),
                raw.get("semantic_types") or raw.get("semantic_type_keys"),
                raw.get("sources") or raw.get("source_turn_ids") or raw.get("source_turns"),
                raw.get("confidence"),
            ]
        elif isinstance(raw, list):
            raw_values = list(raw)
            # Some compatible providers wrap a positional frame with a generic
            # record tag: ["fact", actual_kind, ...]. Remove only this
            # schema-detectable extra slot; never guess from topic content.
            if (
                len(raw_values) >= 23
                and str(raw_values[0] or "").casefold() in {"fact", "frame"}
                and str(raw_values[1] or "").casefold() in _FRAME_KINDS
            ):
                raw_values = raw_values[1:]
            values = raw_values + [None] * max(0, 22 - len(raw_values))
        else:
            continue
        sources = list(dict.fromkeys(
            source_ref_map[item] for item in _strings(values[20])
            if item in source_ref_map
        ))
        if not sources:
            continue
        owner = canonical_key(str(values[1] or ""))
        entity = canonical_key(str(values[2] or "")) or owner
        predicate = canonical_key(str(values[3] or "")) or "said"
        obj = canonical_key(str(values[4] or ""))
        context = canonical_key(str(values[5] or ""))
        if context in {f"t{index}" for index in range(len(turns))}:
            context = ""
        kind = _enum(values[0], _FRAME_KINDS, "fact")
        if kind == "dialogue_answer" and obj in {"", "none"} and predicate in {"none", "said"}:
            continue
        dedupe_key = (
            owner, entity, predicate, obj, context,
            _enum({"affirmed": "positive", "preserved": "positive", "negated": "negative"}.get(str(values[6] or "").casefold(), values[6]), _POLARITIES, "unknown"),
            _float(values[10]), canonical_key(str(values[11] or "")), _float(values[12]),
            _clean_temporal(values[13]), _clean_temporal(values[14]), _clean_temporal(values[15]),
            "" if _is_turn_alias(values[18]) else canonical_key(str(values[18] or "")),
            tuple(sources),
        )
        if dedupe_key in seen:
            continue
        frame_id = f"{question_id}:{session_id}:frame:{len(frames)}"
        seen.add(dedupe_key)
        quantity = QuantityValue(
            value=_float(values[10]),
            unit=str(values[11] or "").strip(),
            multiplier=_float(values[12]),
        )
        temporal = TemporalValue(
            event_time=_clean_temporal(values[13]),
            observed_at=session_date,
            start=_clean_temporal(values[14]),
            end=_clean_temporal(values[15]),
            precision=str(values[16] or "unknown").strip(),
            anchor_source=_clean_temporal(values[17]),
        )
        display = " | ".join(filter(None, (
            owner, entity, predicate, str(values[4] or "").strip(), context,
            temporal.event_time or temporal.start or "",
        )))
        frames.append(
            RoleFrameNode(
                frame_id=frame_id,
                question_id=question_id,
                session_ids=[session_id],
                frame_kind=kind,  # type: ignore[arg-type]
                owner_key=owner,
                entity_key=entity,
                predicate_key=predicate,
                object_key=obj,
                context_key=context,
                polarity=_enum(  # type: ignore[arg-type]
                    {"affirmed": "positive", "preserved": "positive", "negated": "negative"}.get(str(values[6] or "").casefold(), values[6]),
                    _POLARITIES, "unknown",
                ),
                modality=_enum(  # type: ignore[arg-type]
                    values[7], _MODALITIES, "unknown"
                ),
                lifecycle_status=_enum(  # type: ignore[arg-type]
                    values[8], _LIFECYCLES, "unknown"
                ),
                state_op=_enum(values[9], _STATE_OPS, "none"),  # type: ignore[arg-type]
                quantity=quantity,
                temporal=temporal,
                event_identity_key=("" if _is_turn_alias(values[18]) else canonical_key(str(values[18] or ""))),
                semantic_type_keys=[
                    canonical_key(item) for item in _strings(values[19])
                    if canonical_key(item)
                ][:6],
                source_turn_ids=sources,
                confidence=max(0.0, min(1.0, _float(values[21]) or 0.7)),
                retrieval_text=display,
                observation_order=len(frames),
            )
        )
    _augment_explicit_quantities(
        frames, question_id=question_id, session_id=session_id, turns=turns
    )
    declared_coverage: dict[str, CoverageEntry] = {}
    for raw in payload.get("coverage", []):
        if not isinstance(raw, list) or len(raw) < 2:
            continue
        turn_id = source_ref_map.get(str(raw[0] or ""), "")
        coverage_class = _enum(raw[1], _COVERAGE_CLASSES, "non_durable")
        if turn_id in valid_turn_ids:
            declared_coverage[turn_id] = CoverageEntry(
                turn_id=turn_id, coverage_class=coverage_class, frame_ids=[]
            )

    if not frames:
        if (
            set(declared_coverage) == valid_turn_ids
            and all(
                entry.coverage_class
                in {"dialogue_context", "non_durable", "boilerplate"}
                for entry in declared_coverage.values()
            )
        ):
            card = _routing_card(
                question_id=question_id,
                session_id=session_id,
                turns=turns,
                frames=[],
                payload=(
                    payload.get("routing_card")
                    if isinstance(payload.get("routing_card"), dict)
                    else {}
                ),
            )
            return (
                [], card,
                [declared_coverage[turn.node_id] for turn in turns],
                "empty_durable_memory",
            )

        # A provider-declared durable turn is already a useful, grounded
        # memory signal. Preserve it as a weak exact-text frame when the rich
        # extraction was invalid, rather than inventing missing semantics.
        durable_turns = [
            turn for turn in turns
            if declared_coverage.get(turn.node_id)
            and declared_coverage[turn.node_id].coverage_class == "memory_frame"
        ]
        if not durable_turns:
            fallback_frames, card, fallback_coverage = _fallback(
                question_id=question_id,
                session_id=session_id,
                turns=turns,
            )
            return fallback_frames, card, fallback_coverage, "empty_frames"
        for turn in durable_turns:
            frames.append(_lossless_role_frame(
                question_id=question_id,
                session_id=session_id,
                turn=turn,
                frame_index=len(frames),
            ))

    # Frame.sources is the single authoritative turn-to-frame mapping. Older
    # provider responses may still contain coverage frame indexes; accepting
    # the class while ignoring those indexes keeps replay compatibility and
    # eliminates the redundant, error-prone positional ledger.
    coverage_by_turn = declared_coverage
    sourced: defaultdict[str, list[str]] = defaultdict(list)
    for frame in frames:
        for source in frame.source_turn_ids:
            sourced[source].append(frame.frame_id)
    for turn in turns:
        entry = coverage_by_turn.get(turn.node_id)
        if entry is None:
            entry = CoverageEntry(
                turn_id=turn.node_id,
                coverage_class=(
                    "memory_frame" if sourced[turn.node_id] else "lossless_only"
                ),
                frame_ids=list(dict.fromkeys(sourced[turn.node_id])),
            )
            coverage_by_turn[turn.node_id] = entry
        elif sourced[turn.node_id]:
            entry.frame_ids = list(dict.fromkeys(sourced[turn.node_id]))
            # Provenance wins over the compact coverage label.
            entry.coverage_class = "memory_frame"
        elif entry.coverage_class == "memory_frame":
            fallback = _lossless_role_frame(
                question_id=question_id,
                session_id=session_id,
                turn=turn,
                frame_index=len(frames),
            )
            frames.append(fallback)
            sourced[turn.node_id].append(fallback.frame_id)
            entry.frame_ids = [fallback.frame_id]
            entry.coverage_class = "memory_frame"
    coverage = [coverage_by_turn[turn.node_id] for turn in turns]
    frame_by_id = {frame.frame_id: frame for frame in frames}
    for frame in frames:
        frame.coverage_mask = []
    for entry in coverage:
        for frame_id in entry.frame_ids:
            frame_by_id[frame_id].coverage_mask.append(entry.coverage_class)
    card = _routing_card(
        question_id=question_id,
        session_id=session_id,
        turns=turns,
        frames=frames,
        payload=(
            payload.get("routing_card")
            if isinstance(payload.get("routing_card"), dict)
            else {}
        ),
    )
    _attach_routing_relations(frames, card)
    return frames, card, coverage, parse_status



def _edge(
    index: V36Index,
    src: str,
    dst: str,
    relation: str,
    *,
    confidence: float = 1.0,
    provenance: dict[str, Any] | None = None,
    role: str = "",
) -> None:
    if src == dst:
        return
    index.edges.append(
        GraphEdgeV36(
            edge_id=f"{index.turns[0].question_id if index.turns else 'memory'}:edge:{len(index.edges)}",
            question_id=index.turns[0].question_id if index.turns else "",
            src=src,
            dst=dst,
            relation=relation,  # type: ignore[arg-type]
            directed=True,
            confidence=max(0.0, min(1.0, confidence)),
            provenance=provenance or {"local_rule": relation},
            role=role,
        )
    )


def _group(
    index: V36Index,
    kind: str,
    frames: Iterable[RoleFrameNode],
    *,
    required_roles: list[str],
    role_by_frame: dict[str, str] | None = None,
    source_turn_ids: Iterable[str] = (),
    confidence: float | None = None,
) -> EvidenceGroup | None:
    members = list(dict.fromkeys(frame.frame_id for frame in frames))
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    sources = list(dict.fromkeys([
        *source_turn_ids,
        *[
            source for frame_id in members
            for source in frame_by_id[frame_id].source_turn_ids
        ],
    ]))
    if not members and kind != "dialogue_pair":
        return None
    roles = role_by_frame or {}
    present = set(roles.values())
    member_rows = [frame_by_id[frame_id] for frame_id in members]
    if sources:
        present.add("source")
    if member_rows:
        present.update({"fact", "members"})
    if any(frame.context_key or frame.entity_key or frame.predicate_key for frame in member_rows):
        present.add("scope")
    if any(frame.state_op != "none" for frame in member_rows):
        present.add("operations")
    if any(frame.temporal.event_time or frame.temporal.start or frame.temporal.observed_at for frame in member_rows):
        present.add("time")
    for frame_id, role in roles.items():
        frame = frame_by_id[frame_id]
        if role == "event_a" and (frame.temporal.event_time or frame.temporal.start):
            present.add("time_a")
        if role == "event_b" and (frame.temporal.event_time or frame.temporal.start):
            present.add("time_b")
    if not roles and required_roles:
        present.update(role for role in required_roles if role not in {"source", "time", "time_a", "time_b"})
    mask = {role: role in present for role in required_roles}
    text_parts = [
        frame_by_id[frame_id].retrieval_text for frame_id in members
    ]
    group = EvidenceGroup(
        group_id=f"{index.turns[0].question_id}:group:{len(index.evidence_groups)}",
        question_id=index.turns[0].question_id,
        group_kind=kind,  # type: ignore[arg-type]
        member_frame_ids=members,
        source_turn_ids=sources,
        required_roles=required_roles,
        completeness_mask=mask,
        provenance_complete=bool(sources),
        confidence=confidence if confidence is not None else min(
            [frame_by_id[item].confidence for item in members] or [0.8]
        ),
        retrieval_text=" | ".join(text_parts),
        session_ids=list(dict.fromkeys(
            session for frame_id in members
            for session in frame_by_id[frame_id].session_ids
        )),
    )
    index.evidence_groups.append(group)
    return group


_ACTIVITY_STOP = {
    "about", "activities", "activity", "completed", "event", "family",
    "made", "person", "share", "shared", "thing", "trip", "visited", "went",
}


def _activity_terms(frame: RoleFrameNode) -> set[str]:
    text = " ".join([
        frame.entity_key, frame.predicate_key, frame.event_identity_key,
        *frame.semantic_type_keys,
    ])
    terms: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", text.casefold()):
        if len(word) < 4 or word in _ACTIVITY_STOP or word == frame.owner_key:
            continue
        term = word
        if len(word) > 5 and word.endswith("ing"):
            term = word[:-3]
            if len(term) > 2 and term[-1] == term[-2]:
                term = term[:-1]
        elif len(word) > 4 and word.endswith("ed"):
            term = word[:-2]
        if len(term) >= 4 and term not in _ACTIVITY_STOP:
            terms.add(term)
    return terms


def _state_and_structural_groups(index: V36Index) -> None:
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    state_rows: dict[
        tuple[str, str, str, str], list[RoleFrameNode]
    ] = defaultdict(list)
    for frame in index.frames:
        if (
            frame.frame_kind in {"state", "preference"}
            or frame.state_op != "none"
        ):
            state_rows[(
                frame.owner_key, frame.entity_key, frame.predicate_key,
                frame.context_key,
            )].append(frame)
    for key, rows in state_rows.items():
        rows.sort(key=lambda item: (
            item.temporal.event_time or item.temporal.observed_at or "",
            item.observation_order,
        ))
        current: list[RoleFrameNode] = []
        versions: list[StateVersionV36] = []
        for frame in rows:
            timestamp = frame.temporal.event_time or frame.temporal.observed_at
            if frame.state_op in {"set", "cancel"}:
                for prior in current:
                    for version in reversed(versions):
                        if version.frame_id == prior.frame_id:
                            version.valid_to = timestamp
                            break
                current = [] if frame.state_op == "cancel" else [frame]
            elif frame.state_op in {"remove", "decrement"}:
                current = [
                    item for item in current
                    if item.object_key != frame.object_key
                ]
            elif frame.state_op in {"add", "increment"}:
                current = [
                    item for item in current
                    if item.object_key != frame.object_key
                ] + [frame]
            elif (
                frame.modality == "planned"
                and any(item.lifecycle_status == "completed" for item in current)
            ):
                pass
            else:
                current = [frame]
            versions.append(StateVersionV36(
                frame_id=frame.frame_id,
                state_op=frame.state_op,
                object_key=frame.object_key,
                event_time=frame.temporal.event_time,
                observed_at=frame.temporal.observed_at,
                valid_from=timestamp,
                valid_to=None,
                source_turn_ids=frame.source_turn_ids,
            ))
        chain = StateChainV36(
            chain_id=f"{index.turns[0].question_id}:state:{len(index.state_chains)}",
            question_id=index.turns[0].question_id,
            owner_key=key[0],
            entity_key=key[1],
            attribute_key=key[2],
            context_key=key[3],
            versions=versions,
            current_frame_ids=[frame.frame_id for frame in current],
        )
        index.state_chains.append(chain)
        for old, new in zip(rows, rows[1:]):
            _edge(
                index, old.frame_id, new.frame_id, "state_transition",
                confidence=min(old.confidence, new.confidence),
                provenance={
                    "local_rule": "same_owner_entity_attribute_context",
                    "chain_id": chain.chain_id,
                    "source_turn_ids": list(dict.fromkeys(
                        [*old.source_turn_ids, *new.source_turn_ids]
                    )),
                },
            )
            _group(
                index, "state_transition", [old, new],
                required_roles=["previous_state", "current_state", "time", "source"],
                role_by_frame={
                    old.frame_id: "previous_state",
                    new.frame_id: "current_state",
                },
            )

    collections: dict[
        tuple[str, str, str, str], list[RoleFrameNode]
    ] = defaultdict(list)
    for frame in index.frames:
        collection_scope = frame.context_key or frame.event_identity_key
        if (
            collection_scope
            and frame.state_op in {
                "set", "add", "remove", "increment", "decrement", "cancel",
            }
        ):
            collections[(
                frame.owner_key, frame.entity_key, frame.predicate_key,
                collection_scope,
            )].append(frame)
    for rows in collections.values():
        if len(rows) < 2:
            continue
        collection_semantics = (
            "collection", "list", "inventory", "member", "item", "set",
        )
        has_collection_signal = any(
            frame.state_op in {
                "add", "remove", "increment", "decrement", "cancel",
            }
            or any(
                signal in semantic_type
                for semantic_type in frame.semantic_type_keys
                for signal in collection_semantics
            )
            for frame in rows
        )
        if not has_collection_signal:
            continue
        group = _group(
            index, "collection", rows,
            required_roles=["scope", "members", "operations", "source"],
        )
        if group:
            for frame in rows:
                _edge(
                    index, group.group_id, frame.frame_id, "collection_member",
                    provenance={
                        "local_rule": "exact_collection_key",
                        "source_turn_ids": frame.source_turn_ids,
                    },
                )

    activity_collections: dict[tuple[str, str], list[RoleFrameNode]] = defaultdict(list)
    for frame in index.frames:
        if (
            frame.frame_kind not in {"event", "fact", "state"}
            or frame.lifecycle_status != "completed"
            or frame.modality != "asserted"
            or frame.polarity != "positive"
            or not frame.owner_key
            or not frame.source_turn_ids
        ):
            continue
        for term in sorted(_activity_terms(frame)):
            activity_collections[(frame.owner_key, term)].append(frame)
    for (owner_key, activity_key), rows in activity_collections.items():
        rows = list({frame.frame_id: frame for frame in rows}.values())
        session_ids = {
            session_id for frame in rows for session_id in frame.session_ids
        }
        if len(session_ids) < 2 or not 2 <= len(rows) <= 8:
            continue
        member_ids = {frame.frame_id for frame in rows}
        if any(
            group.group_kind == "collection"
            and set(group.member_frame_ids) == member_ids
            for group in index.evidence_groups
        ):
            continue
        group = _group(
            index, "collection", rows,
            required_roles=["scope", "members", "event", "source"],
        )
        if group:
            for frame in rows:
                _edge(
                    index, group.group_id, frame.frame_id, "collection_member",
                    confidence=min(0.9, frame.confidence),
                    provenance={
                        "local_rule": "bounded_cross_session_activity_collection",
                        "owner_key": owner_key,
                        "activity_key": activity_key,
                        "source_turn_ids": frame.source_turn_ids,
                    },
                )

    events: dict[str, list[RoleFrameNode]] = defaultdict(list)
    for frame in index.frames:
        if frame.event_identity_key:
            events[frame.event_identity_key].append(frame)
    for identity, rows in events.items():
        rows.sort(key=lambda item: (
            item.temporal.event_time or item.temporal.observed_at or "",
            item.observation_order,
        ))
        for old, new in zip(rows, rows[1:]):
            _edge(
                index, old.frame_id, new.frame_id, "same_event",
                confidence=min(old.confidence, new.confidence),
                provenance={
                    "local_rule": "exact_event_identity",
                    "event_identity_key": identity,
                    "source_turn_ids": list(dict.fromkeys(
                        [*old.source_turn_ids, *new.source_turn_ids]
                    )),
                },
            )
        timed = [
            frame for frame in rows
            if frame.temporal.event_time or frame.temporal.start
        ]
        for old, new in zip(timed, timed[1:]):
            group = _group(
                index, "temporal_pair", [old, new],
                required_roles=["event_a", "event_b", "time_a", "time_b", "source"],
                role_by_frame={
                    old.frame_id: "event_a", new.frame_id: "event_b",
                },
            )
            if group:
                _edge(
                    index, group.group_id, old.frame_id, "temporal_endpoint",
                    provenance={
                        "local_rule": "same_event_identity_endpoint",
                        "source_turn_ids": old.source_turn_ids,
                    },
                    role="event_a",
                )
                _edge(
                    index, group.group_id, new.frame_id, "temporal_endpoint",
                    provenance={
                        "local_rule": "same_event_identity_endpoint",
                        "source_turn_ids": new.source_turn_ids,
                    },
                    role="event_b",
                )

    contrasts: dict[
        tuple[str, str, str, str, str], dict[str, RoleFrameNode]
    ] = defaultdict(dict)
    for frame in index.frames:
        contrasts[(
            frame.owner_key, frame.entity_key, frame.predicate_key,
            frame.object_key, frame.context_key,
        )][frame.polarity] = frame
    for rows in contrasts.values():
        if "positive" not in rows or "negative" not in rows:
            continue
        positive, negative = rows["positive"], rows["negative"]
        _edge(
            index, positive.frame_id, negative.frame_id, "contrast",
            confidence=min(positive.confidence, negative.confidence),
            provenance={
                "local_rule": "same_fact_opposite_polarity",
                "source_turn_ids": list(dict.fromkeys(
                    [*positive.source_turn_ids, *negative.source_turn_ids]
                )),
            },
        )
        _group(
            index, "contrast", [positive, negative],
            required_roles=["positive", "negative", "source"],
            role_by_frame={
                positive.frame_id: "positive",
                negative.frame_id: "negative",
            },
        )


def _dialogue_groups(index: V36Index) -> None:
    turns_by_session: dict[str, list[TurnNodeV36]] = defaultdict(list)
    frames_by_source: dict[str, list[RoleFrameNode]] = defaultdict(list)
    for turn in index.turns:
        turns_by_session[turn.session_id].append(turn)
    for frame in index.frames:
        for source in frame.source_turn_ids:
            frames_by_source[source].append(frame)
    for rows in turns_by_session.values():
        rows.sort(key=lambda item: item.turn_index)
        for left, right in zip(rows, rows[1:]):
            _edge(
                index, left.node_id, right.node_id, "next_turn",
                provenance={
                    "local_rule": "adjacent_turn",
                    "source_turn_ids": [left.node_id, right.node_id],
                },
            )
            if left.speaker_key == right.speaker_key:
                continue
            if not (
                _QUESTION_CUE.search(left.text)
                or _REQUEST_CUE.search(left.text)
            ):
                continue
            pair_ids = {left.node_id, right.node_id}
            response_frames = [
                frame for frame in frames_by_source.get(right.node_id, [])
                if set(frame.source_turn_ids) <= pair_ids
            ]
            prompt_frames = [
                frame for frame in frames_by_source.get(left.node_id, [])
                if set(frame.source_turn_ids) <= pair_ids
            ]
            member_frames = [*prompt_frames, *response_frames]
            group = _group(
                index, "dialogue_pair", member_frames,
                required_roles=["prompt_turn", "reply_turn", "reply_content", "source"],
                role_by_frame={
                    **{frame.frame_id: "prompt_turn" for frame in prompt_frames},
                    **{frame.frame_id: "reply_content" for frame in response_frames},
                },
                source_turn_ids=[left.node_id, right.node_id],
                confidence=0.95 if _QUESTION_CUE.search(left.text) else 0.8,
            )
            if group:
                group.retrieval_text = (
                    f"prompt {left.speaker}: {left.text} | "
                    f"reply {right.speaker}: {right.text}"
                )
                group.session_ids = [left.session_id]
                group.completeness_mask["prompt_turn"] = True
                group.completeness_mask["reply_turn"] = True
                group.completeness_mask["reply_content"] = bool(right.text)
                group.completeness_mask["source"] = True
                _edge(
                    index, left.node_id, right.node_id, "dialogue_pair",
                    confidence=group.confidence,
                    provenance={
                        "local_rule": "question_or_request_to_adjacent_reply",
                        "group_id": group.group_id,
                        "source_turn_ids": [left.node_id, right.node_id],
                    },
                )


def _source_and_routing_edges(index: V36Index) -> None:
    for frame in index.frames:
        for source in frame.source_turn_ids:
            _edge(
                index, frame.frame_id, source, "source",
                confidence=frame.confidence,
                provenance={
                    "local_rule": "extraction_source",
                    "source_turn_ids": [source],
                },
            )
    for card in index.routing_cards:
        for frame_id in card.frame_ids:
            _edge(
                index, card.card_id, frame_id, "routing_contains",
                provenance={
                    "local_rule": "session_card_contains",
                    "session_id": card.session_id,
                },
            )


def build_inverted_indexes(index: V36Index) -> None:
    fields: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set) for name in (
            "entity_key", "owner_key", "speaker_key", "predicate_key",
            "object_key", "event_identity_key", "date", "lifecycle_status",
            "polarity", "semantic_type",
        )
    }
    for turn in index.turns:
        if turn.speaker_key:
            fields["speaker_key"][turn.speaker_key].add(turn.node_id)
        if turn.session_date:
            fields["date"][turn.session_date].add(turn.node_id)
    for frame in index.frames:
        for name in (
            "entity_key", "owner_key", "predicate_key", "object_key",
            "event_identity_key", "lifecycle_status", "polarity",
        ):
            value = str(getattr(frame, name) or "")
            if value:
                fields[name][value].add(frame.frame_id)
        for semantic_type in frame.semantic_type_keys:
            fields["semantic_type"][semantic_type].add(frame.frame_id)
        for value in (
            frame.temporal.event_time, frame.temporal.start,
            frame.temporal.end, frame.temporal.observed_at,
        ):
            if value:
                fields["date"][value].add(frame.frame_id)
    index.inverted_indexes = {
        name: {key: sorted(ids) for key, ids in values.items()}
        for name, values in fields.items()
    }


def add_routing_semantic_edges(
    index: V36Index, *, k: int = 3, floor: float = 0.72, per_card_cap: int = 2
) -> None:
    cards = [card for card in index.routing_cards if card.embedding is not None]
    if len(cards) < 2:
        return

    neighbors: dict[str, list[tuple[float, str]]] = {}
    for card in cards:
        ranked = sorted(
            (
                (cosine_similarity(card.embedding or [], other.embedding or []), other.card_id)
                for other in cards if other.card_id != card.card_id
            ),
            reverse=True,
        )
        neighbors[card.card_id] = ranked[:k]
    existing = {(edge.src, edge.dst, edge.relation) for edge in index.edges}
    for card in cards:
        accepted = 0
        for score, other_id in neighbors[card.card_id]:
            if score < floor:
                continue
            if card.card_id not in {
                value for _score, value in neighbors.get(other_id, [])
            }:
                continue
            pair = tuple(sorted((card.card_id, other_id)))
            if (pair[0], pair[1], "semantic_neighbor") in existing:
                continue
            _edge(
                index, pair[0], pair[1], "semantic_neighbor",
                confidence=score,
                provenance={
                    "local_rule": "routing_card_mutual_knn",
                    "k": k, "floor": floor, "protected": False,
                },
            )
            existing.add((pair[0], pair[1], "semantic_neighbor"))
            accepted += 1
            if accepted >= per_card_cap:
                break


def build_index(
    *,
    question_id: str,
    turns: list[TurnNodeV36],
    frames: list[RoleFrameNode],
    routing_cards: list[RoutingCard],
    coverage: list[CoverageEntry],
) -> V36Index:
    # Reassign a stable global observation order without changing frame IDs.
    frames = [
        replace(frame, observation_order=order)
        for order, frame in enumerate(frames)
    ]
    index = V36Index(
        turns=turns,
        frames=frames,
        routing_cards=routing_cards,
        coverage=coverage,
    )
    _source_and_routing_edges(index)
    _dialogue_groups(index)
    _state_and_structural_groups(index)
    build_inverted_indexes(index)
    errors = validate_index(index)
    if errors:
        raise ValueError(f"V3.6 index validation failed: {errors[:12]}")
    return index


def validate_index(index: V36Index) -> list[str]:
    errors: list[str] = []
    if index.schema_version != GRAPHMEM_V36_SCHEMA:
        errors.append("schema_version")
    node_ids = {
        node.node_id for node in [
            *index.turns, *index.frames, *index.routing_cards,
            *index.evidence_groups,
        ]
    }
    if len(node_ids) != (
        len(index.turns) + len(index.frames) + len(index.routing_cards)
        + len(index.evidence_groups)
    ):
        errors.append("duplicate_node_id")
    turn_ids = {turn.node_id for turn in index.turns}
    frame_ids = {frame.frame_id for frame in index.frames}
    coverage_ids = {entry.turn_id for entry in index.coverage}
    if coverage_ids != turn_ids:
        errors.append("coverage_manifest")
    for entry in index.coverage:
        if not set(entry.frame_ids) <= frame_ids:
            errors.append(f"coverage_frame:{entry.turn_id}")
        if entry.coverage_class == "memory_frame" and not entry.frame_ids:
            errors.append(f"coverage_empty_memory:{entry.turn_id}")
    semantic_keys: set[tuple[Any, ...]] = set()
    for frame in index.frames:
        if not frame.source_turn_ids or not set(frame.source_turn_ids) <= turn_ids:
            errors.append(f"frame_source:{frame.frame_id}")
        key = (
            frame.owner_key, frame.entity_key, frame.predicate_key,
            frame.object_key, frame.context_key, frame.polarity,
            frame.quantity.value, canonical_key(frame.quantity.unit), frame.quantity.multiplier,
            frame.temporal.event_time, frame.temporal.start, frame.temporal.end,
            frame.event_identity_key, tuple(frame.source_turn_ids),
        )
        if key in semantic_keys:
            errors.append(f"duplicate_frame:{frame.frame_id}")
        semantic_keys.add(key)
    edge_ids: set[str] = set()
    for edge in index.edges:
        if edge.edge_id in edge_ids:
            errors.append(f"duplicate_edge:{edge.edge_id}")
        edge_ids.add(edge.edge_id)
        if edge.relation not in _ALLOWED_RELATIONS:
            errors.append(f"edge_relation:{edge.edge_id}")
        if edge.relation in _FORBIDDEN_RELATIONS:
            errors.append(f"forbidden_edge:{edge.edge_id}")
        if edge.src not in node_ids or edge.dst not in node_ids:
            errors.append(f"edge_endpoint:{edge.edge_id}")
        if not edge.directed:
            errors.append(f"undirected_edge:{edge.edge_id}")
        if not edge.provenance:
            errors.append(f"edge_provenance:{edge.edge_id}")
    for group in index.evidence_groups:
        if not set(group.member_frame_ids) <= frame_ids:
            errors.append(f"group_member:{group.group_id}")
        if not group.source_turn_ids or not set(group.source_turn_ids) <= turn_ids:
            errors.append(f"group_source:{group.group_id}")
        if set(group.required_roles) != set(group.completeness_mask):
            errors.append(f"group_roles:{group.group_id}")
        if group.provenance_complete != bool(group.source_turn_ids):
            errors.append(f"group_provenance:{group.group_id}")
    for chain in index.state_chains:
        if not {item.frame_id for item in chain.versions} <= frame_ids:
            errors.append(f"chain_member:{chain.chain_id}")
        if not set(chain.current_frame_ids) <= frame_ids:
            errors.append(f"chain_current:{chain.chain_id}")
    return errors


def clone_index(index: V36Index, question_id: str) -> V36Index:
    old_question = index.turns[0].question_id if index.turns else ""

    def remap(value: str) -> str:
        prefix = f"{old_question}:"
        return f"{question_id}:{value[len(prefix):]}" if value.startswith(prefix) else value

    payload = json.loads(json.dumps(index, default=lambda value: value.__dict__))
    raw = V36Index()
    from .schema import index_from_dict

    raw = index_from_dict(payload)
    raw.turns = [
        replace(item, node_id=remap(item.node_id), question_id=question_id)
        for item in raw.turns
    ]
    raw.frames = [
        replace(
            item, frame_id=remap(item.frame_id), question_id=question_id,
            source_turn_ids=[remap(value) for value in item.source_turn_ids],
        )
        for item in raw.frames
    ]
    raw.routing_cards = [
        replace(
            item, card_id=remap(item.card_id), question_id=question_id,
            frame_ids=[remap(value) for value in item.frame_ids],
            turn_ids=[remap(value) for value in item.turn_ids],
        )
        for item in raw.routing_cards
    ]
    raw.evidence_groups = [
        replace(
            item, group_id=remap(item.group_id), question_id=question_id,
            member_frame_ids=[remap(value) for value in item.member_frame_ids],
            source_turn_ids=[remap(value) for value in item.source_turn_ids],
        )
        for item in raw.evidence_groups
    ]
    raw.edges = [
        replace(
            item, edge_id=remap(item.edge_id), question_id=question_id,
            src=remap(item.src), dst=remap(item.dst),
        )
        for item in raw.edges
    ]
    raw.state_chains = [
        replace(
            item, chain_id=remap(item.chain_id), question_id=question_id,
            versions=[
                replace(
                    value, frame_id=remap(value.frame_id),
                    source_turn_ids=[
                        remap(source) for source in value.source_turn_ids
                    ],
                )
                for value in item.versions
            ],
            current_frame_ids=[
                remap(value) for value in item.current_frame_ids
            ],
        )
        for item in raw.state_chains
    ]
    raw.coverage = [
        replace(
            item, turn_id=remap(item.turn_id),
            frame_ids=[remap(value) for value in item.frame_ids],
        )
        for item in raw.coverage
    ]
    build_inverted_indexes(raw)
    return raw
