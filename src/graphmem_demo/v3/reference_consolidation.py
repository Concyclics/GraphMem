from __future__ import annotations

import json
import re
from typing import Any

from ..clients import cosine_similarity
from .action_semantics import action_family_overlap
from .schema import EventNode, HyperEdge, HyperIncidence, TurnNode


_REFERENCE_CUE = re.compile(
    r"\b(?:that|this|the|it|one)\b.{0,80}\b(?:you|your)\b.{0,60}"
    r"\b(?:recommend\w*|suggest\w*|advise\w*|mention\w*|share\w*|"
    r"send\w*|give\w*|show\w*|promise\w*|tell\w*)\b|"
    r"\b(?:you|your)\b.{0,60}\b(?:recommendation|suggestion|advice|"
    r"message|photo|link|name|idea|tip)\b",
    re.IGNORECASE,
)
_EXPLICIT_REFERENCE_CUE = re.compile(
    r"\b(?:that|this|the)\s+[\w'-]+(?:\s+[\w'-]+){0,5}\s+you\s+"
    r"(?:recommend\w*|suggest\w*|advise\w*|mention\w*|share\w*|"
    r"send\w*|give\w*|show\w*|promise\w*|tell\w*)\b",
    re.IGNORECASE,
)
_ANTECEDENT_ACTION_CUE = re.compile(
    r"\b(?:recommend\w*|suggest\w*|advise\w*|mention\w*|share\w*|"
    r"send\w*|give\w*|show\w*|promise\w*|tell\w*)\b",
    re.IGNORECASE,
)
_NAMED_VALUE_CUE = re.compile(
    r'"[^"\n]{2,100}"|(?<!\w)\'[^\'\n]{2,100}\'(?!\w)|'
    r"\b[A-Z][\w'-]+\s+(?:by|from)\s+"
    r"[A-Z][\w'-]+",
)
_GENERIC_EVENT_TERMS = {
    "activity", "complete", "completed", "conversation", "discuss",
    "discussed", "discussion", "event", "exchange", "interaction",
    "plan", "planned", "planning", "share", "shared", "sharing",
    "practice", "routine", "social", "support", "supported", "talk",
    "talked", "update", "wellbeing",
    "about", "and", "for", "from", "her", "his", "into", "its",
    "our", "the", "their", "this", "through", "with", "your",
}



def reference_candidate_payload(
    turns: list[TurnNode],
    *,
    max_anchors: int = 10,
    candidates_per_anchor: int = 3,
) -> dict[str, Any] | None:
    """Build a bounded, query-independent candidate set for discourse references."""
    anchor_rows: list[tuple[float, int, dict[str, Any]]] = []
    for position, anchor in enumerate(turns):
        if not _REFERENCE_CUE.search(anchor.text):
            continue
        ranked: list[tuple[float, int, TurnNode]] = []
        for candidate_position, candidate in enumerate(turns[:position]):
            if candidate.speaker_key == anchor.speaker_key:
                continue
            participant_bonus = 0.0
            if anchor.listener and anchor.listener.casefold() in candidate.speaker.casefold():
                participant_bonus += 0.18
            if candidate.listener and anchor.speaker.casefold() in candidate.listener.casefold():
                participant_bonus += 0.12
            semantic = max(
                0.0, cosine_similarity(anchor.embedding, candidate.embedding)
            )
            action_bonus = (
                0.55
                if _EXPLICIT_REFERENCE_CUE.search(anchor.text)
                and _ANTECEDENT_ACTION_CUE.search(candidate.text)
                and action_family_overlap(anchor.text, candidate.text) > 0
                else 0.0
            )
            value_bonus = 1.00 if _NAMED_VALUE_CUE.search(candidate.text) else 0.0
            recency = candidate_position / max(1, position)
            ranked.append((0.72 * semantic + participant_bonus + action_bonus
                           + value_bonus + 0.10 * recency,
                           candidate_position, candidate))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        candidates = []
        for _score, candidate_position, candidate in ranked[:candidates_per_anchor]:
            context_turns = [
                turns[index]
                for index in (candidate_position - 1, candidate_position, candidate_position + 1)
                if 0 <= index < len(turns)
                and turns[index].session_id == candidate.session_id
            ]
            candidates.append({
                "node_id": candidate.node_id,
                "speaker": candidate.speaker,
                "listener": candidate.listener,
                "date": candidate.session_date,
                "text": candidate.text[:260],
                "local_context": "\n".join(
                    f"{item.speaker}: {item.text}" for item in context_turns
                )[:380],
            })
        if candidates:
            cue_priority = 2.0 if _EXPLICIT_REFERENCE_CUE.search(anchor.text) else 1.0
            resolvability = ranked[0][0] if ranked else 0.0
            anchor_rows.append((cue_priority + 0.35 * resolvability, position, {
                "node_id": anchor.node_id,
                "speaker": anchor.speaker,
                "listener": anchor.listener,
                "date": anchor.session_date,
                "text": anchor.text[:260],
                "candidates": candidates,
            }))
    anchor_rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    anchors = [row for _score, _position, row in anchor_rows[:max_anchors]]
    anchors.sort(key=lambda row: next(
        index for index, turn in enumerate(turns) if turn.node_id == row["node_id"]
    ))
    return {"anchors": anchors} if anchors else None


def _term_key(value: str) -> str:
    key = value.casefold().strip("'")
    return key[:-2] if key.endswith("'s") else key


def _specific_terms(value: str, participant_keys: list[str]) -> set[str]:
    participant_terms = {
        _term_key(token)
        for participant in participant_keys
        for token in re.findall(r"[\w'-]+", participant)
    }
    terms: set[str] = set()
    for token in re.findall(r"[\w'-]+", value):
        key = _term_key(token)
        if (
            len(key) >= 3
            and key not in _GENERIC_EVENT_TERMS
            and key not in participant_terms
        ):
            terms.add(key)
    return terms


def _event_terms(event: EventNode) -> set[str]:
    return _specific_terms(event.label, event.participant_keys)


def _specific_type_keys(event: EventNode) -> set[str]:
    return {
        token for value in event.semantic_type_keys
        for token in _specific_terms(value, event.participant_keys)
    }


def event_identity_candidate_rows(
    events: list[EventNode],
    turns: list[TurnNode],
    *,
    max_pairs: int = 8,
    candidates_per_event: int = 2,
) -> list[dict[str, Any]]:
    """Propose bounded cross-session pairs; the LLM must still verify identity."""
    turn_by_id = {turn.node_id: turn for turn in turns}
    position = {event.node_id: index for index, event in enumerate(events)}
    proposed: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for later_index, later in enumerate(events):
        ranked: list[tuple[float, EventNode]] = []
        if later.status not in {"complete", "cancelled"}:
            continue
        later_types = _specific_type_keys(later)
        later_terms = _event_terms(later)
        for earlier in events[:later_index]:
            if earlier.session_id == later.session_id:
                continue
            if earlier.status in {"complete", "cancelled"}:
                continue
            earlier_types = _specific_type_keys(earlier)
            earlier_terms = _event_terms(earlier)
            type_overlap = len(later_types & earlier_types)
            term_union = later_terms | earlier_terms
            term_overlap = (
                len(later_terms & earlier_terms) / len(term_union)
                if term_union else 0.0
            )
            semantic = max(0.0, cosine_similarity(later.embedding, earlier.embedding))
            if not ((type_overlap and later_terms & earlier_terms) or len(later_terms & earlier_terms) >= 2):
                continue
            participant_union = set(later.participant_keys) | set(earlier.participant_keys)
            participant_overlap = (
                len(set(later.participant_keys) & set(earlier.participant_keys))
                / len(participant_union)
                if participant_union else 0.0
            )
            score = (
                0.48 * semantic
                + 0.20 * min(1.0, type_overlap)
                + 0.16 * min(1.0, len(later_terms & earlier_terms))
                + 0.10 * term_overlap
                + 0.06 * participant_overlap
            )
            ranked.append((score, earlier))
        ranked.sort(key=lambda row: (row[0], position[row[1].node_id]), reverse=True)
        for score, earlier in ranked[:candidates_per_event]:
            key = (earlier.node_id, later.node_id)
            source_ids = list(dict.fromkeys([
                *earlier.source_turn_ids[:2], *later.source_turn_ids[:2]
            ]))
            source_context = "\n".join(
                f"{turn_by_id[source_id].speaker}: {turn_by_id[source_id].text}"
                for source_id in source_ids if source_id in turn_by_id
            )[:520]
            proposed[key] = (score, {
                "earlier_event_id": earlier.node_id,
                "later_event_id": later.node_id,
                "earlier": [
                    earlier.session_id, earlier.label, earlier.status,
                    earlier.event_time, earlier.participant_keys,
                    earlier.semantic_type_keys,
                ],
                "later": [
                    later.session_id, later.label, later.status,
                    later.event_time, later.participant_keys,
                    later.semantic_type_keys,
                ],
                "source_context": source_context,
            })
    ranked_pairs = sorted(
        proposed.values(), key=lambda row: row[0], reverse=True
    )[:max_pairs]
    return [row for _score, row in ranked_pairs]


def consolidation_candidate_payload(
    turns: list[TurnNode], events: list[EventNode]
) -> dict[str, Any] | None:
    references = reference_candidate_payload(turns) or {}
    event_pairs = event_identity_candidate_rows(events, turns)
    payload = {
        "anchors": references.get("anchors", []),
        "event_pairs": event_pairs,
    }
    return payload if payload["anchors"] or payload["event_pairs"] else None


def reference_consolidation_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "Consolidate a query-independent memory graph. For each discourse anchor, "
        "link it to at most one listed antecedent only when that antecedent explicitly supplies "
        "the value referred to by the anchor. Do not link merely topical or same-speaker facts. "
        "For each listed event pair, emit a link only when both records denote the same "
        "real-world event or a clear plan-to-outcome lifecycle of that one event. Shared topic, "
        "participants, activity type, or recurring activity alone is insufficient. Respect "
        "speaker/listener roles, event times, status, and chronology. identity_basis must name "
        "a concrete shared identity anchor present on both sides, such as the same named project, "
        "object, place, or single occurrence. Omit ongoing conversations, support relationships, "
        "and merely similar activities when no concrete shared anchor exists. Return JSON only as "
        '{"links":[["anchor_node_id","antecedent_node_id",confidence,"resolved_value"]],'
        '"event_links":[["earlier_event_id","later_event_id",confidence,"identity_basis"]],'
        '"event_clusters":[{"member_event_ids":["event_id"],"canonical_label":"specific '
        'event","identity_anchors":["grounded anchor"],"confidence":0.0}],'
        '"event_decisions":[{"target_event_id":"terminal_event_id",'
        '"predecessor_event_ids":["candidate_id"],"canonical_label":"specific event",'
        '"identity_anchors":["grounded anchor"],"confidence":0.0}]}. '
        "When event_candidates are provided, cluster 2-6 listed events only when they are mentions "
        "of one real-world event; preserve a specific early identity through generic later mentions. "
        "Prefer clusters that span a lifecycle state change such as plan or progress to one outcome. "
        "Never cluster two separately completed or cancelled records merely because their participant, "
        "profession, event type, or generic object label is similar. "
        "For every event_neighborhood, emit exactly one event_decisions row. Select only listed "
        "candidate_predecessor_ids that are earlier mentions of that exact terminal event; use an "
        "empty predecessor_event_ids list when none qualifies. Do not omit a neighborhood. "
        "Confidence must be between 0 and 1. Omit ambiguous references."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_reference_edges(
    text: str,
    *,
    question_id: str,
    turns: list[TurnNode],
) -> list[HyperEdge]:
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return []
    rows = payload.get("links", []) if isinstance(payload, dict) else []
    turn_by_id = {turn.node_id: turn for turn in turns}
    position = {turn.node_id: index for index, turn in enumerate(turns)}
    edges: list[HyperEdge] = []
    seen: set[tuple[str, str]] = set()
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            anchor_id = str(row.get("anchor_node_id") or row.get("anchor") or "")
            antecedent_id = str(
                row.get("antecedent_node_id") or row.get("antecedent") or ""
            )
            confidence_value = row.get("confidence")
            resolved_value = str(row.get("resolved_value") or row.get("value") or "")
        elif isinstance(row, list) and len(row) >= 3:
            anchor_id, antecedent_id = str(row[0]), str(row[1])
            confidence_value = row[2]
            resolved_value = str(row[3] if len(row) > 3 else "")
        else:
            continue
        if (
            anchor_id not in turn_by_id
            or antecedent_id not in turn_by_id
            or position[antecedent_id] >= position[anchor_id]
            or turn_by_id[anchor_id].speaker_key == turn_by_id[antecedent_id].speaker_key
            or (anchor_id, antecedent_id) in seen
        ):
            continue
        if (
            _EXPLICIT_REFERENCE_CUE.search(turn_by_id[anchor_id].text)
            and action_family_overlap(
                turn_by_id[anchor_id].text, turn_by_id[antecedent_id].text
            ) <= 0
        ):
            continue
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            continue
        if confidence < 0.70:
            continue
        seen.add((anchor_id, antecedent_id))
        edge_index = len(edges)
        edges.append(HyperEdge(
            edge_id=f"{question_id}:reference:{edge_index}",
            question_id=question_id,
            relation="refers_to",
            incidences=[
                HyperIncidence(antecedent_id, "antecedent", 0),
                HyperIncidence(anchor_id, "anaphor", 1),
            ],
            directed=True,
            confidence=max(0.0, min(1.0, confidence)),
            provenance={
                "generator": "bounded_reference_consolidation",
                "anchor_id": anchor_id,
                "antecedent_id": antecedent_id,
                "resolved_value": resolved_value[:240],
            },
            retrieval_text=" | ".join(part for part in (
                resolved_value,
                turn_by_id[antecedent_id].retrieval_text,
                turn_by_id[anchor_id].retrieval_text,
            ) if part),
        ))
    return edges


def parse_event_identity_edges(
    text: str,
    *,
    question_id: str,
    events: list[EventNode],
    candidate_pairs: list[dict[str, Any]],
) -> list[HyperEdge]:
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return []
    rows = payload.get("event_links", []) if isinstance(payload, dict) else []
    event_by_id = {event.node_id: event for event in events}
    position = {event.node_id: index for index, event in enumerate(events)}
    allowed = {
        (str(row.get("earlier_event_id") or ""), str(row.get("later_event_id") or ""))
        for row in candidate_pairs
    }
    edges: list[HyperEdge] = []
    seen: set[tuple[str, str]] = set()
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            earlier_id = str(row.get("earlier_event_id") or row.get("earlier") or "")
            later_id = str(row.get("later_event_id") or row.get("later") or "")
            confidence_value = row.get("confidence")
            identity_basis = str(row.get("identity_basis") or row.get("basis") or "")
        elif isinstance(row, list) and len(row) >= 3:
            earlier_id, later_id = str(row[0]), str(row[1])
            confidence_value = row[2]
            identity_basis = str(row[3] if len(row) > 3 else "")
        else:
            continue
        if (
            (earlier_id, later_id) not in allowed
            or earlier_id not in event_by_id
            or later_id not in event_by_id
            or event_by_id[earlier_id].session_id == event_by_id[later_id].session_id
            or position[earlier_id] >= position[later_id]
            or (earlier_id, later_id) in seen
        ):
            continue
        earlier = event_by_id[earlier_id]
        later = event_by_id[later_id]
        type_overlap = _specific_type_keys(earlier) & _specific_type_keys(later)
        term_overlap = _event_terms(earlier) & _event_terms(later)
        if (
            earlier.status in {"complete", "cancelled"}
            or later.status not in {"complete", "cancelled"}
            or not ((type_overlap and term_overlap) or len(term_overlap) >= 2)
        ):
            continue
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            continue
        if confidence < 0.85:
            continue
        seen.add((earlier_id, later_id))
        edges.append(HyperEdge(
            edge_id=f"{question_id}:event_identity:{len(edges)}",
            question_id=question_id,
            relation="same_event",
            incidences=[
                HyperIncidence(earlier_id, f"earlier_{earlier.status}", 0),
                HyperIncidence(later_id, f"later_{later.status}", 1),
            ],
            directed=True,
            confidence=max(0.0, min(1.0, confidence)),
            provenance={
                "generator": "bounded_event_identity_consolidation",
                "earlier_event_id": earlier_id,
                "later_event_id": later_id,
                "identity_basis": identity_basis[:240],
            },
            retrieval_text=" | ".join(part for part in (
                earlier.retrieval_text, later.retrieval_text, identity_basis
            ) if part),
        ))
    return edges
