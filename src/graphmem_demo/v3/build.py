from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Iterable

from ..clients import cosine_similarity
from ..models import QuestionCase
from .catalog import ensure_catalog
from .extraction_normalize import normalize_extraction_payload
from .source_resolution import resolve_source_ids
from .schema import (
    GRAPHMEM_V3_SCHEMA,
    ClaimNode,
    EpisodeNode,
    EventNode,
    HyperEdge,
    HyperIncidence,
    StateChainV3,
    ThemeNode,
    TurnNode,
    V3Index,
)


V3_PROMPT_VERSION = "graphmem_v3_role_neutral_extract_20260727f"
V3_BUILD_VERSION = "graphmem_v3_hypergraph_build_20260727ab"

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_GENERIC_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "i", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "ours", "she", "so", "that", "the", "their", "theirs",
    "them", "they", "this", "to", "was", "we", "were", "with", "you", "your",
}


def canonical_key(value: str) -> str:
    words = [
        token.casefold()
        for token in _WORD_RE.findall(value.replace("_", " "))
        if token.casefold() not in _GENERIC_STOP
    ]
    return " ".join(words)

_PROGRESS_EVENT_CUE = re.compile(
    r"\b(?:currently|still)\b.{0,80}\b(?:work|build|develop|make|progress)\w*\b|"
    r"\b(?:work|build|develop|make)\w*\s+on\b|"
    r"\b(?:in\s+progress|taking\s+shape|(?:come|coming)\s+together)\b",
    re.IGNORECASE,
)
_COMPLETION_EVENT_CUE = re.compile(
    r"\b(?:finished|completed|finali[sz]ed|accomplished)\b|"
    r"\b(?:finally\s+)?wrapp?ed\b.{0,50}\bup\b|"
    r"\b(?:it|that|the\s+[\w'-]+)\s+(?:is|was|'s)\s+done\b",
    re.IGNORECASE,
)
_CANCEL_EVENT_CUE = re.compile(
    r"\b(?:cancelled|canceled|called\s+off|abandoned)\b", re.IGNORECASE,
)


def calibrate_event_status(status: str, source_text: str) -> str:
    """Correct explicit lifecycle conflicts without reclassifying ordinary past events."""
    if _CANCEL_EVENT_CUE.search(source_text):
        return "cancelled"
    if _COMPLETION_EVENT_CUE.search(source_text):
        return "complete"
    if status == "complete" and _PROGRESS_EVENT_CUE.search(source_text):
        return "asserted"
    return status



def _speaker(message: dict[str, Any], role: str) -> str:
    explicit = str(message.get("speaker") or "").strip()
    if explicit:
        return explicit
    return "participant_1" if role == "user" else "participant_2"


def build_turn_nodes(case: QuestionCase) -> list[TurnNode]:
    turns: list[TurnNode] = []
    for session_id, session_date, messages in zip(
        case.haystack_session_ids, case.haystack_dates, case.haystack_sessions
    ):
        for turn_index, message in enumerate(messages):
            role = str(message.get("role") or "unknown").casefold()
            speaker = _speaker(message, role)
            listener = str(message.get("listener") or "").strip()
            text = str(message.get("content") or "").strip()
            node_id = f"{case.question_id}:{session_id}:turn:{turn_index}"
            cues = [f"speaker {speaker}", text]
            if listener:
                cues.append(f"listener {listener}")
            turns.append(
                TurnNode(
                    node_id=node_id,
                    question_id=case.question_id,
                    session_id=session_id,
                    session_date=session_date,
                    turn_index=turn_index,
                    speaker=speaker,
                    speaker_key=canonical_key(speaker) or speaker.casefold(),
                    listener=listener,
                    transport_role=role,
                    text=text,
                    retrieval_text=" | ".join(cues),
                )
            )
    return turns


def session_extraction_messages(
    session_id: str,
    session_date: str | None,
    turns: list[TurnNode],
) -> list[dict[str, str]]:
    rows = [
        [turn.node_id, turn.speaker, turn.listener, turn.transport_role, turn.text]
        for turn in turns
    ]
    schema = {
        "claims": [
            [
                "subject", "predicate", "object", "kind", "polarity", "modality",
                "state_op", "context", "event_time", ["source_turn_ids"],
                "quantity_or_null", "unit", "confidence",
            ]
        ],
        "events": [
            [
                "label", "status", "event_time", ["participant_names"],
                ["claim_indices"], ["source_turn_ids"], "confidence",
                ["question_independent_semantic_type_keys"],
            ]
        ],
        "episodes": [
            {
                "label": "short neutral label",
                "turn_ids": ["source turn ids"],
                "claim_indices": [0],
                "event_indices": [0],
            }
        ],
    }
    system = (
        "Extract a role-neutral memory graph from a conversation. Treat every named "
        "speaker as an equal participant; transport roles such as user/assistant are "
        "not evidence of importance or truth. Preserve exact names, numbers, units, "
        "negation, modality, plans versus completed actions, and explicit or relative "
        "times anchored to the session date. Split independent facts and collection "
        "items. For coordinated lists, repeat every shared action, owner, source, and "
        "context relation on each emitted item; never replace that relation with a generic "
        "predicate such as item. Every claim and event must cite existing source turn IDs. Emit an event only when it joins multiple claims or spans multiple turns; do not duplicate a single claim as an event. Do not infer "
        "facts that were not said. Preserve participant-specific memory and exact assistant-provided "
        "answers, names, lists, and numbers; compress generic explanatory boilerplate. Before emitting a second claim from any one turn, cover every source turn that contains a durable participant state, resource, preference, plan, completed action, exact quantity, or explicitly named answer. For a long unaccepted explanatory or recommendation list, emit at most three representative claims and preserve the complete lossless turn through its episode and source pointers; such a list must never crowd out later participant-authored memory. Emit at most "
        "24 claims. Always emit exactly the top-level keys claims, events, and episodes. Claims and "
        "events must be positional arrays matching the schema, never objects. For each event, emit one to three question-independent semantic type keys from broad to specific. Types must describe the source event itself, not copy a benchmark question, dataset label, or requested answer. Return JSON only."
    )
    user = json.dumps(
        {
            "session_id": session_id,
            "session_date": session_date,
            "schema": schema,
            "allowed": {
                "kind": ["state", "event", "preference", "quantity", "general"],
                "polarity": ["positive", "negative", "unknown"],
                "modality": ["asserted", "planned", "possible", "conditional", "unknown"],
                "state_op": [
                    "assert", "retract", "replace", "add", "remove",
                    "complete", "cancel", "none",
                ],
                "event_status": [
                    "asserted", "planned", "possible", "complete", "cancelled", "unknown",
                ],
            },
            "turns": rows,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except Exception:
            return None


def _decode_named_array(text: str, name: str) -> list[Any] | None:
    match = re.search(rf"\"{re.escape(name)}\"\s*:\s*", text)
    if not match:
        return None
    start = text.find("[", match.end())
    if start < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except Exception:
        return None
    return value if isinstance(value, list) else None


def _enum(value: Any, allowed: set[str], default: str) -> str:
    key = str(value or "").strip().casefold()
    return key if key in allowed else default


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_ids(value: Any, valid: set[str]) -> list[str]:
    return resolve_source_ids(value, valid)


def _fallback_claims(
    question_id: str, session_id: str, turns: list[TurnNode]
) -> list[ClaimNode]:
    return [
        ClaimNode(
            node_id=f"{question_id}:{session_id}:claim:{index}",
            question_id=question_id,
            session_id=session_id,
            subject=turn.speaker,
            subject_key=turn.speaker_key,
            predicate="said",
            predicate_key="said",
            object=turn.text,
            object_key=canonical_key(turn.text),
            kind="general",
            source_turn_ids=[turn.node_id],
            observed_at=turn.session_date,
            confidence=0.4,
            retrieval_text=f"{turn.speaker} said {turn.text}",
            observation_order=index,
        )
        for index, turn in enumerate(turns)
        if turn.text
    ]


def parse_session_extraction(
    text: str,
    *,
    question_id: str,
    session_id: str,
    session_date: str | None,
    turns: list[TurnNode],
) -> tuple[list[ClaimNode], list[EventNode], list[dict[str, Any]], str | None]:
    payload = _json_object(text)
    parse_status: str | None = None
    if payload is None:
        salvaged_claims = _decode_named_array(text, "claims")
        if salvaged_claims is not None:
            payload = {
                "claims": salvaged_claims,
                "events": _decode_named_array(text, "events") or [],
                "episodes": _decode_named_array(text, "episodes") or [],
            }
            parse_status = "partial_json_salvaged"
    valid_turns = {turn.node_id for turn in turns}
    if payload is None:
        return _fallback_claims(question_id, session_id, turns), [], [], "invalid_json"
    payload = normalize_extraction_payload(payload)

    claims: list[ClaimNode] = []
    claim_rows = payload.get("claims") if isinstance(payload.get("claims"), list) else []
    for row in claim_rows:
        if not isinstance(row, list) or len(row) < 10:
            continue
        sources = _source_ids(row[9], valid_turns)
        if not sources:
            continue
        subject, predicate, obj = (str(row[index] or "").strip() for index in range(3))
        if not (subject and predicate and obj):
            continue
        kind = _enum(row[3], {"state", "event", "preference", "quantity", "general"}, "general")
        polarity = _enum(row[4], {"positive", "negative", "unknown"}, "unknown")
        modality = _enum(
            row[5], {"asserted", "planned", "possible", "conditional", "unknown"}, "unknown"
        )
        state_op = _enum(
            row[6],
            {"assert", "retract", "replace", "add", "remove", "complete", "cancel", "none"},
            "none",
        )
        context = canonical_key(str(row[7] or "")) or "default"
        event_time = str(row[8]).strip() if row[8] else None
        quantity = _float(row[10] if len(row) > 10 else None)
        unit = str(row[11] or "").strip() if len(row) > 11 else ""
        confidence = _float(row[12] if len(row) > 12 else None)
        claims.append(
            ClaimNode(
                node_id=f"{question_id}:{session_id}:claim:{len(claims)}",
                question_id=question_id,
                session_id=session_id,
                subject=subject,
                subject_key=canonical_key(subject) or subject.casefold(),
                predicate=predicate,
                predicate_key=canonical_key(predicate) or predicate.casefold(),
                object=obj,
                object_key=canonical_key(obj),
                kind=kind,  # type: ignore[arg-type]
                polarity=polarity,  # type: ignore[arg-type]
                modality=modality,  # type: ignore[arg-type]
                state_op=state_op,  # type: ignore[arg-type]
                context_key=context,
                event_time=event_time,
                observed_at=session_date,
                valid_from=event_time or session_date,
                quantity=quantity,
                unit=unit,
                source_turn_ids=sources,
                confidence=max(0.0, min(1.0, confidence if confidence is not None else 0.8)),
                retrieval_text=" | ".join(
                    part for part in (subject, predicate, obj, context, event_time or "", unit) if part
                ),
            )
        )
    covered_turn_ids = {
        source_id for claim in claims for source_id in claim.source_turn_ids
    }
    if (
        claims and turns
        and len(covered_turn_ids) / len(turns) < 0.35
    ):
        for fallback in _fallback_claims(question_id, session_id, turns):
            if fallback.source_turn_ids[0] in covered_turn_ids:
                continue
            claims.append(replace(
                fallback,
                node_id=f"{question_id}:{session_id}:claim:{len(claims)}",
                observation_order=len(claims),
            ))
        parse_status = "undercovered_claims_augmented"
    if not claims and claim_rows:
        return _fallback_claims(question_id, session_id, turns), [], [], "empty_claims"
    if not claims:
        parse_status = parse_status or "no_claims"
    for index, claim in enumerate(claims):
        claim.observation_order = index

    events: list[EventNode] = []
    event_rows = payload.get("events") if isinstance(payload.get("events"), list) else []
    for row in event_rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        sources = _source_ids(row[5], valid_turns)
        label = str(row[0] or "").strip()
        if not label or not sources:
            continue
        claim_ids = [
            claims[index].node_id
            for index in row[4] if isinstance(index, int) and 0 <= index < len(claims)
        ] if isinstance(row[4], list) else []
        participants = [
            canonical_key(str(value)) or str(value).casefold()
            for value in (row[3] if isinstance(row[3], list) else [])
            if str(value).strip()
        ]
        confidence = _float(row[6] if len(row) > 6 else None)
        semantic_type_keys = list(dict.fromkeys(
            canonical_key(str(value))
            for value in (row[7] if len(row) > 7 and isinstance(row[7], list) else [])
            if canonical_key(str(value))
        ))[:3]
        event_time = str(row[2]).strip() if row[2] else None
        raw_status = _enum(
            row[1],
            {"asserted", "planned", "possible", "complete", "cancelled", "unknown"},
            "unknown",
        )
        source_text = "\n".join(
            turn.text for turn in turns if turn.node_id in sources
        )
        status = calibrate_event_status(raw_status, source_text)
        events.append(
            EventNode(
                node_id=f"{question_id}:{session_id}:event:{len(events)}",
                question_id=question_id,
                session_id=session_id,
                label=label,
                label_key=canonical_key(label),
                status=status,  # type: ignore[arg-type]
                participant_keys=list(dict.fromkeys(participants)),
                event_time=event_time,
                claim_ids=claim_ids,
                source_turn_ids=sources,
                semantic_type_keys=semantic_type_keys,
                confidence=max(0.0, min(1.0, confidence if confidence is not None else 0.8)),
                retrieval_text=" | ".join(
                    [label, *semantic_type_keys, *participants, event_time or "", status]
                ),
            )
        )

    raw_episodes = payload.get("episodes")
    episodes = [row for row in raw_episodes if isinstance(row, dict)] if isinstance(raw_episodes, list) else []
    return claims, events, episodes, parse_status


def _mean(vectors: Iterable[list[float] | None]) -> list[float] | None:
    rows = [row for row in vectors if row]
    if not rows:
        return None
    size = min(len(row) for row in rows)
    return [sum(row[index] for row in rows) / len(rows) for index in range(size)]


def _top_terms(texts: Iterable[str], limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(
            token.casefold()
            for token in _WORD_RE.findall(text)
            if len(token) > 2 and token.casefold() not in _GENERIC_STOP
        )
    return [token for token, _count in counts.most_common(limit)]


def _episodes(
    question_id: str,
    sessions: list[str],
    session_dates: dict[str, str | None],
    turns: list[TurnNode],
    claims: list[ClaimNode],
    events: list[EventNode],
    proposals: dict[str, list[dict[str, Any]]],
    max_turns: int = 8,
) -> list[EpisodeNode]:
    turns_by_session: dict[str, list[TurnNode]] = defaultdict(list)
    for turn in turns:
        turns_by_session[turn.session_id].append(turn)
    claim_by_turn: dict[str, list[ClaimNode]] = defaultdict(list)
    event_by_turn: dict[str, list[EventNode]] = defaultdict(list)
    for claim in claims:
        for source in claim.source_turn_ids:
            claim_by_turn[source].append(claim)
    for event in events:
        for source in event.source_turn_ids:
            event_by_turn[source].append(event)

    result: list[EpisodeNode] = []
    for session_id in sessions:
        rows = sorted(turns_by_session[session_id], key=lambda item: item.turn_index)
        valid_ids = {turn.node_id for turn in rows}
        groups: list[tuple[str, list[str]]] = []
        used: set[str] = set()
        for proposal in proposals.get(session_id, []):
            ids = [str(value) for value in proposal.get("turn_ids") or [] if str(value) in valid_ids]
            ids = [value for value in ids if value not in used]
            if ids:
                groups.append((str(proposal.get("label") or "").strip(), ids))
                used.update(ids)
        remaining = [turn for turn in rows if turn.node_id not in used]
        for offset in range(0, len(remaining), max_turns):
            chunk = remaining[offset:offset + max_turns]
            if chunk:
                groups.append(("", [turn.node_id for turn in chunk]))
        order = {turn.node_id: turn.turn_index for turn in rows}
        groups.sort(key=lambda item: min(order[value] for value in item[1]))
        for label, turn_ids in groups:
            episode_claims = list(dict.fromkeys(
                claim.node_id for turn_id in turn_ids for claim in claim_by_turn[turn_id]
            ))
            episode_events = list(dict.fromkeys(
                event.node_id for turn_id in turn_ids for event in event_by_turn[turn_id]
            ))
            selected_turns = [turn for turn in rows if turn.node_id in set(turn_ids)]
            participants = list(dict.fromkeys(turn.speaker_key for turn in selected_turns))
            selected_claims = [claim for claim in claims if claim.node_id in set(episode_claims)]
            selected_events = [event for event in events if event.node_id in set(episode_events)]
            terms = _top_terms(
                [turn.text for turn in selected_turns]
                + [claim.retrieval_text for claim in selected_claims]
            )
            rendered_label = label or " ".join(terms[:5]) or f"episode {len(result) + 1}"
            date_values = [
                value for value in
                [*(claim.event_time for claim in selected_claims),
                 *(event.event_time for event in selected_events)]
                if value
            ]
            retrieval = "\n".join(
                [
                    f"Episode: {rendered_label}",
                    f"Participants: {', '.join(participants)}",
                    f"Time: {session_dates.get(session_id) or 'unknown'}",
                    *[f"Claim: {claim.retrieval_text}" for claim in selected_claims],
                    *[f"Event: {event.retrieval_text}" for event in selected_events],
                ]
            )
            result.append(
                EpisodeNode(
                    node_id=f"{question_id}:episode:{len(result)}",
                    question_id=question_id,
                    session_id=session_id,
                    session_date=session_dates.get(session_id),
                    label=rendered_label,
                    participant_keys=participants,
                    time_start=min(date_values) if date_values else session_dates.get(session_id),
                    time_end=max(date_values) if date_values else session_dates.get(session_id),
                    turn_ids=turn_ids,
                    claim_ids=episode_claims,
                    event_ids=episode_events,
                    retrieval_text=retrieval,
                    embedding=_mean(
                        [turn.embedding for turn in selected_turns]
                        + [claim.embedding for claim in selected_claims]
                        + [event.embedding for event in selected_events]
                    ),
                )
            )
    return result


def _themes(question_id: str, episodes: list[EpisodeNode], floor: float = 0.58) -> list[ThemeNode]:
    if not episodes:
        return []
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left in range(len(episodes)):
        scored = sorted(
            (
                (cosine_similarity(episodes[left].embedding, episodes[right].embedding), right)
                for right in range(len(episodes)) if right != left
            ),
            reverse=True,
        )[:2]
        for score, right in scored:
            shared = set(episodes[left].participant_keys) & set(episodes[right].participant_keys)
            if score >= floor or (score >= floor - 0.08 and shared):
                neighbors[left].add(right)
                neighbors[right].add(left)
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(episodes)):
        if start in seen:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(neighbors[current] - seen)
        components.append(sorted(component))

    # Session themes are deliberately overlapping with semantic components.
    by_session: dict[str, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        by_session[episode.session_id].append(index)
    groups = components + [indices for indices in by_session.values() if len(indices) > 1]
    unique: set[tuple[int, ...]] = set()
    result: list[ThemeNode] = []
    for indices in groups:
        key = tuple(sorted(indices))
        if not key or key in unique:
            continue
        unique.add(key)
        members = [episodes[index] for index in key]
        labels = _top_terms([member.retrieval_text for member in members], 8)
        participant_keys = list(dict.fromkeys(
            value for member in members for value in member.participant_keys
        ))
        claim_ids = list(dict.fromkeys(value for member in members for value in member.claim_ids))
        event_ids = list(dict.fromkeys(value for member in members for value in member.event_ids))
        turn_ids = list(dict.fromkeys(value for member in members for value in member.turn_ids))
        starts = [member.time_start for member in members if member.time_start]
        ends = [member.time_end for member in members if member.time_end]
        retrieval = "\n".join(
            [
                f"Theme labels: {', '.join(labels)}",
                f"Participants: {', '.join(participant_keys)}",
                *[f"Episode: {member.label}" for member in members],
            ]
        )
        result.append(
            ThemeNode(
                node_id=f"{question_id}:theme:{len(result)}",
                question_id=question_id,
                labels=labels,
                participant_keys=participant_keys,
                time_start=min(starts) if starts else None,
                time_end=max(ends) if ends else None,
                episode_ids=[member.node_id for member in members],
                claim_ids=claim_ids,
                event_ids=event_ids,
                source_turn_ids=turn_ids,
                retrieval_text=retrieval,
                embedding=_mean(member.embedding for member in members),
            )
        )
    return result


def _state_chains(question_id: str, claims: list[ClaimNode]) -> tuple[list[StateChainV3], list[HyperEdge]]:
    groups: dict[tuple[str, str, str], list[ClaimNode]] = defaultdict(list)
    for claim in claims:
        if claim.state_op != "none" or claim.kind in {"state", "preference"}:
            groups[(claim.subject_key, claim.predicate_key, claim.context_key)].append(claim)
    chains: list[StateChainV3] = []
    edges: list[HyperEdge] = []
    for key, rows in groups.items():
        rows.sort(key=lambda item: (item.event_time or item.observed_at or "", item.observation_order))
        current: list[ClaimNode] = []
        for claim in rows:
            if claim.state_op in {"replace", "retract", "cancel"}:
                for previous in current:
                    previous.valid_to = claim.event_time or claim.observed_at
                current = [] if claim.state_op in {"retract", "cancel"} else [claim]
            elif claim.state_op == "remove":
                current = [item for item in current if item.object_key != claim.object_key]
            elif claim.state_op == "add":
                current = [item for item in current if item.object_key != claim.object_key] + [claim]
            elif claim.modality == "planned" and any(item.state_op == "complete" for item in current):
                continue
            else:
                current = [claim]
        chain_id = f"{question_id}:state:{len(chains)}"
        chains.append(
            StateChainV3(
                chain_id=chain_id,
                question_id=question_id,
                subject_key=key[0],
                predicate_key=key[1],
                context_key=key[2],
                current_claim_ids=[item.node_id for item in current],
                history_claim_ids=[item.node_id for item in rows],
                update_order=[item.node_id for item in rows],
                valid_from=rows[0].valid_from if rows else None,
                valid_to=rows[-1].valid_to if rows else None,
            )
        )
        if len(rows) > 1:
            edges.append(
                HyperEdge(
                    edge_id=f"{question_id}:hyperedge:state:{len(edges)}",
                    question_id=question_id,
                    relation="state_history",
                    incidences=[
                        HyperIncidence(item.node_id, "current" if item in current else "history", order)
                        for order, item in enumerate(rows)
                    ],
                    directed=True,
                    confidence=min(item.confidence for item in rows),
                    provenance={"chain_id": chain_id, "local_rule": "ordered_state_transition"},
                    retrieval_text=" | ".join(
                        f"{item.subject} {item.predicate} {item.object}" for item in rows
                    ),
                    embedding=_mean(item.embedding for item in rows),
                )
            )
        positives = {item.object_key: item for item in rows if item.polarity == "positive"}
        negatives = {item.object_key: item for item in rows if item.polarity == "negative"}
        for object_key in positives.keys() & negatives.keys():
            pair = [positives[object_key], negatives[object_key]]
            edges.append(
                HyperEdge(
                    edge_id=f"{question_id}:hyperedge:contradiction:{len(edges)}",
                    question_id=question_id,
                    relation="contradiction",
                    incidences=[HyperIncidence(item.node_id, item.polarity) for item in pair],
                    confidence=min(item.confidence for item in pair),
                    provenance={"local_rule": "same_state_opposite_polarity"},
                    retrieval_text=" | ".join(item.retrieval_text for item in pair),
                    embedding=_mean(item.embedding for item in pair),
                )
            )
    return chains, edges


def _hyperedges(index: V3Index) -> list[HyperEdge]:
    edges: list[HyperEdge] = []

    def add(relation: str, incidences: list[HyperIncidence], *, directed: bool = False,
            confidence: float = 1.0, provenance: dict[str, Any] | None = None,
            retrieval_text: str = "", embedding: list[float] | None = None) -> None:
        if len({item.node_id for item in incidences}) < 2:
            return
        edges.append(
            HyperEdge(
                edge_id=f"{index.turns[0].question_id if index.turns else 'memory'}:hyperedge:{len(edges)}",
                question_id=index.turns[0].question_id if index.turns else "",
                relation=relation,  # type: ignore[arg-type]
                incidences=incidences,
                directed=directed,
                confidence=confidence,
                provenance=provenance or {"local_rule": relation},
                retrieval_text=retrieval_text,
                embedding=embedding,
            )
        )

    node_vectors: dict[str, list[float] | None] = {
        node.node_id: node.embedding
        for node in [
            *index.turns, *index.claims, *index.events, *index.episodes, *index.themes,
            *index.event_frames, *index.operands,
        ]
    }
    for claim in index.claims:
        add(
            "supports",
            [HyperIncidence(source, "source") for source in claim.source_turn_ids]
            + [HyperIncidence(claim.node_id, "claim")],
            directed=True,
            confidence=claim.confidence,
            retrieval_text=claim.retrieval_text,
            embedding=claim.embedding,
        )
    for episode in index.episodes:
        members = (
            [HyperIncidence(value, "turn") for value in episode.turn_ids]
            + [HyperIncidence(value, "claim") for value in episode.claim_ids]
            + [HyperIncidence(value, "event") for value in episode.event_ids]
            + [HyperIncidence(episode.node_id, "episode")]
        )
        add(
            "episode_member", members, directed=False,
            retrieval_text=episode.retrieval_text, embedding=episode.embedding,
        )
    for theme in index.themes:
        for offset in range(0, len(theme.episode_ids), 24):
            member_ids = theme.episode_ids[offset:offset + 24]
            add(
                "theme_member",
                [HyperIncidence(value, "episode") for value in member_ids]
                + [HyperIncidence(theme.node_id, "theme")],
                retrieval_text=theme.retrieval_text,
                embedding=theme.embedding,
            )
    participant_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    participant_coarse: dict[str, set[str]] = defaultdict(set)
    for turn in index.turns:
        participant_groups[(turn.speaker_key, turn.session_id)].add(turn.node_id)
    for claim in index.claims:
        participant_groups[(claim.subject_key, claim.session_id)].add(claim.node_id)
    for event in index.events:
        for participant in event.participant_keys:
            participant_groups[(participant, event.session_id)].add(event.node_id)
    for episode in index.episodes:
        for participant in episode.participant_keys:
            participant_groups[(participant, episode.session_id)].add(episode.node_id)
            participant_coarse[participant].add(episode.node_id)
    for theme in index.themes:
        for participant in theme.participant_keys:
            participant_coarse[participant].add(theme.node_id)
    for (participant, session_id), node_ids in participant_groups.items():
        selected = sorted(node_ids)
        for offset in range(0, len(selected), 24):
            chunk = selected[offset:offset + 24]
            add(
                "participant",
                [HyperIncidence(value, "local_mention") for value in chunk],
                retrieval_text=f"{participant} | session {session_id}",
                embedding=_mean(node_vectors.get(value) for value in chunk),
            )
    for participant, node_ids in participant_coarse.items():
        selected = sorted(node_ids)
        for offset in range(0, len(selected), 24):
            chunk = selected[offset:offset + 24]
            add(
                "participant",
                [HyperIncidence(value, "coarse_mention") for value in chunk],
                retrieval_text=participant,
                embedding=_mean(node_vectors.get(value) for value in chunk),
            )
    event_groups: dict[str, list[EventNode]] = defaultdict(list)
    for event in index.events:
        if event.label_key:
            event_groups[event.label_key].append(event)
    for label, rows in event_groups.items():
        add(
            "same_event",
            [HyperIncidence(item.node_id, item.status) for item in rows],
            directed=True,
            confidence=min(item.confidence for item in rows),
            retrieval_text=label,
            embedding=_mean(item.embedding for item in rows),
        )
    quantity_groups: dict[tuple[str, str, str], list[ClaimNode]] = defaultdict(list)
    for claim in index.claims:
        if claim.kind == "quantity" or claim.quantity is not None:
            quantity_groups[(claim.subject_key, claim.predicate_key, claim.context_key)].append(claim)
    for key, rows in quantity_groups.items():
        add(
            "quantity_collection",
            [HyperIncidence(item.node_id, "operand", order) for order, item in enumerate(rows)],
            retrieval_text=" | ".join(key),
            embedding=_mean(item.embedding for item in rows),
        )
    temporal_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for claim in index.claims:
        if claim.event_time:
            temporal_groups[(claim.subject_key, claim.predicate_key)].add(claim.node_id)
    for event in index.events:
        if event.event_time:
            for participant in event.participant_keys:
                temporal_groups[(participant, event.label_key)].add(event.node_id)
    for key, node_ids in temporal_groups.items():
        selected = sorted(node_ids)
        add(
            "temporal_scope",
            [HyperIncidence(value, "timed") for value in selected],
            directed=True,
            retrieval_text=" | ".join(key),
            embedding=_mean(node_vectors.get(value) for value in selected),
        )
    return edges


def build_hypergraph(
    *,
    question_id: str,
    session_ids: list[str],
    session_dates: dict[str, str | None],
    turns: list[TurnNode],
    claims: list[ClaimNode],
    events: list[EventNode],
    episode_proposals: dict[str, list[dict[str, Any]]],
) -> V3Index:
    episodes = _episodes(
        question_id, session_ids, session_dates, turns, claims, events, episode_proposals
    )
    themes = _themes(question_id, episodes)
    chains, state_edges = _state_chains(question_id, claims)
    index = V3Index(
        turns=turns,
        claims=claims,
        events=events,
        episodes=episodes,
        themes=themes,
        state_chains=chains,
    )
    index.hyperedges = [*_hyperedges(index), *state_edges]
    ensure_catalog(index)
    errors = validate_hypergraph(index)
    if errors:
        raise ValueError(f"V3 hypergraph validation failed: {errors[:8]}")
    return index


def validate_hypergraph(index: V3Index) -> list[str]:
    errors: list[str] = []
    nodes = {
        node.node_id: node
        for node in [
            *index.turns, *index.claims, *index.events, *index.event_entities,
            *index.episodes, *index.themes,
            *index.event_frames, *index.operands,
        ]
    }
    turn_ids = {turn.node_id for turn in index.turns}
    for claim in index.claims:
        if not claim.source_turn_ids or not set(claim.source_turn_ids) <= turn_ids:
            errors.append(f"claim_source:{claim.node_id}")
    for event in index.events:
        if not event.source_turn_ids or not set(event.source_turn_ids) <= turn_ids:
            errors.append(f"event_source:{event.node_id}")
    event_ids = {event.node_id for event in index.events}
    for entity in index.event_entities:
        if not entity.source_turn_ids or not set(entity.source_turn_ids) <= turn_ids:
            errors.append(f"event_entity_source:{entity.node_id}")
        if not set(entity.member_event_ids) <= event_ids:
            errors.append(f"event_entity_member:{entity.node_id}")
        if entity.current_event_id not in set(entity.member_event_ids):
            errors.append(f"event_entity_current:{entity.node_id}")
    for episode in index.episodes:
        if not episode.turn_ids or not set(episode.turn_ids) <= turn_ids:
            errors.append(f"episode_source:{episode.node_id}")
    edge_ids: set[str] = set()
    for edge in index.hyperedges:
        if edge.edge_id in edge_ids:
            errors.append(f"duplicate_edge:{edge.edge_id}")
        edge_ids.add(edge.edge_id)
        if len(edge.incidences) < 2:
            errors.append(f"edge_arity:{edge.edge_id}")
        for incidence in edge.incidences:
            if incidence.node_id not in nodes:
                errors.append(f"edge_endpoint:{edge.edge_id}:{incidence.node_id}")
    for theme in index.themes:
        if not theme.source_turn_ids or not set(theme.source_turn_ids) <= turn_ids:
            errors.append(f"theme_source:{theme.node_id}")
    claim_ids = {claim.node_id for claim in index.claims}
    event_ids = {event.node_id for event in index.events}
    frame_ids = {frame.frame_id for frame in index.event_frames}
    for frame in index.event_frames:
        if not frame.source_turn_ids or not set(frame.source_turn_ids) <= turn_ids:
            errors.append(f"event_frame_source:{frame.frame_id}")
        if not set(frame.claim_ids) <= claim_ids or not set(frame.event_ids) <= event_ids:
            errors.append(f"event_frame_member:{frame.frame_id}")
    for operand in index.operands:
        if not operand.source_turn_ids or not set(operand.source_turn_ids) <= turn_ids:
            errors.append(f"operand_source:{operand.operand_id}")
        if not operand.source_claim_ids or not set(operand.source_claim_ids) <= claim_ids:
            errors.append(f"operand_claim:{operand.operand_id}")
        if operand.event_frame_id and operand.event_frame_id not in frame_ids:
            errors.append(f"operand_frame:{operand.operand_id}")
    return errors


def prompt_hash() -> str:
    return hashlib.sha256(
        f"{V3_PROMPT_VERSION}\n{V3_BUILD_VERSION}".encode()
    ).hexdigest()


def clone_index(index: V3Index, question_id: str) -> V3Index:
    ensure_catalog(index)
    id_map: dict[str, str] = {}

    def remap(value: str) -> str:
        if value not in id_map:
            suffix = value.split(":", 1)[1] if ":" in value else value
            id_map[value] = f"{question_id}:{suffix}"
        return id_map[value]

    turns = [
        replace(turn, node_id=remap(turn.node_id), question_id=question_id)
        for turn in index.turns
    ]
    claims = [
        replace(
            claim, node_id=remap(claim.node_id), question_id=question_id,
            source_turn_ids=[remap(value) for value in claim.source_turn_ids],
        )
        for claim in index.claims
    ]
    events = [
        replace(
            event, node_id=remap(event.node_id), question_id=question_id,
            claim_ids=[remap(value) for value in event.claim_ids],
            source_turn_ids=[remap(value) for value in event.source_turn_ids],
        )
        for event in index.events
    ]
    event_entities = [
        replace(
            entity, node_id=remap(entity.node_id), question_id=question_id,
            member_event_ids=[remap(value) for value in entity.member_event_ids],
            current_event_id=remap(entity.current_event_id)
            if entity.current_event_id else None,
            source_turn_ids=[remap(value) for value in entity.source_turn_ids],
        )
        for entity in index.event_entities
    ]
    episodes = [
        replace(
            episode, node_id=remap(episode.node_id), question_id=question_id,
            turn_ids=[remap(value) for value in episode.turn_ids],
            claim_ids=[remap(value) for value in episode.claim_ids],
            event_ids=[remap(value) for value in episode.event_ids],
        )
        for episode in index.episodes
    ]
    themes = [
        replace(
            theme, node_id=remap(theme.node_id), question_id=question_id,
            episode_ids=[remap(value) for value in theme.episode_ids],
            claim_ids=[remap(value) for value in theme.claim_ids],
            event_ids=[remap(value) for value in theme.event_ids],
            source_turn_ids=[remap(value) for value in theme.source_turn_ids],
        )
        for theme in index.themes
    ]
    hyperedges = [
        replace(
            edge, edge_id=remap(edge.edge_id), question_id=question_id,
            incidences=[
                replace(incidence, node_id=remap(incidence.node_id))
                for incidence in edge.incidences
            ],
            provenance=dict(edge.provenance),
        )
        for edge in index.hyperedges
    ]
    chains = [
        replace(
            chain, chain_id=remap(chain.chain_id), question_id=question_id,
            current_claim_ids=[remap(value) for value in chain.current_claim_ids],
            history_claim_ids=[remap(value) for value in chain.history_claim_ids],
            update_order=[remap(value) for value in chain.update_order],
        )
        for chain in index.state_chains
    ]
    frames = [
        replace(
            frame, frame_id=remap(frame.frame_id), question_id=question_id,
            claim_ids=[remap(value) for value in frame.claim_ids],
            event_ids=[remap(value) for value in frame.event_ids],
            source_turn_ids=[remap(value) for value in frame.source_turn_ids],
        )
        for frame in index.event_frames
    ]
    operands = [
        replace(
            operand, operand_id=remap(operand.operand_id), question_id=question_id,
            event_frame_id=remap(operand.event_frame_id) if operand.event_frame_id else None,
            source_claim_ids=[remap(value) for value in operand.source_claim_ids],
            source_turn_ids=[remap(value) for value in operand.source_turn_ids],
        )
        for operand in index.operands
    ]
    return V3Index(
        turns=turns, claims=claims, events=events,
        event_entities=event_entities, episodes=episodes,
        themes=themes, hyperedges=hyperedges, state_chains=chains,
        event_frames=frames, operands=operands,
    )
