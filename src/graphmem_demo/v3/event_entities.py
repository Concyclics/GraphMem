from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from ..clients import cosine_similarity
from .reference_consolidation import _event_terms, _specific_type_keys
from .schema import EventEntityNode, EventNode, HyperEdge, HyperIncidence, TurnNode


_EXPLICIT_TERMINAL_CUE = re.compile(
    r"\b(?:complet\w*|finish\w*|finaliz\w*|cancel\w*|accomplish\w*|"
    r"achiev\w*|graduat\w*|ended|closed|adopted|won|done|conclud\w*|wrapped\s+up)\b",
    re.IGNORECASE,
)


def _pair_profile(
    earlier: EventNode, later: EventNode
) -> tuple[float, list[str], list[str]] | None:
    # Two terminal records are not a lifecycle transition. Merging them from
    # type similarity alone is unsafe because they are often sibling events.
    if (
        earlier.status in {"complete", "cancelled"}
        and later.status in {"complete", "cancelled"}
    ):
        return None
    participants = set(earlier.participant_keys) & set(later.participant_keys)
    terms = sorted(_event_terms(earlier) & _event_terms(later))
    types = sorted(_specific_type_keys(earlier) & _specific_type_keys(later))
    if not participants or not (terms or types):
        return None
    semantic = max(0.0, cosine_similarity(earlier.embedding, later.embedding))
    lifecycle_transition = (
        earlier.status not in {"complete", "cancelled"}
        and later.status in {"complete", "cancelled"}
    )
    if semantic < 0.42 and not (
        (terms and types) or (lifecycle_transition and terms)
    ):
        return None
    lifecycle = float(lifecycle_transition)
    participant_union = set(earlier.participant_keys) | set(later.participant_keys)
    score = (
        0.42 * semantic
        + 0.18 * min(1.0, len(terms))
        + 0.16 * min(1.0, len(types))
        + 0.12 * len(participants) / max(1, len(participant_union))
        + 0.12 * lifecycle
    )
    return score, terms, types


def event_entity_candidate_payload(
    events: list[EventNode],
    turns: list[TurnNode],
    *,
    candidates_per_event: int = 5,
    max_links: int = 48,
    max_events: int = 28,
) -> dict[str, Any] | None:
    """Build a bounded, query-independent candidate graph for event identity."""
    turn_by_id = {turn.node_id: turn for turn in turns}
    explicit_terminal_ids = {
        event.node_id
        for event in events
        if event.status in {"complete", "cancelled"}
        and _EXPLICIT_TERMINAL_CUE.search("\n".join([
            event.label,
            *(turn_by_id[value].text for value in event.source_turn_ids
              if value in turn_by_id),
        ]))
    }
    buckets: list[tuple[EventNode, list[dict[str, Any]]]] = []
    for later_index, later in enumerate(events):
        ranked = []
        for earlier in events[:later_index]:
            if earlier.session_id == later.session_id:
                continue
            profile = _pair_profile(earlier, later)
            if profile is not None:
                ranked.append((*profile, earlier))
        ranked.sort(key=lambda row: (row[0], row[3].node_id), reverse=True)
        rows = [
            {
                "earlier_event_id": earlier.node_id,
                "later_event_id": later.node_id,
                "score": round(score, 6),
                "shared_terms": terms,
                "shared_types": types,
            }
            for score, terms, types, earlier
            in ranked[:candidates_per_event]
        ]
        if rows:
            buckets.append((later, rows))

    links: list[dict[str, Any]] = []
    allowed: set[str] = set()

    def take(row: dict[str, Any]) -> None:
        if row in links or len(links) >= max_links:
            return
        endpoints = {
            str(row["earlier_event_id"]), str(row["later_event_id"])
        }
        if len(allowed | endpoints) > max_events:
            return
        links.append(row)
        allowed.update(endpoints)

    # Explicit outcomes receive their full local neighborhood before noisy
    # status labels can consume the bounded prompt.
    for rank in range(candidates_per_event):
        for later, rows in buckets:
            if (
                later.node_id in explicit_terminal_ids
                and rank < len(rows)
            ):
                take(rows[rank])
    # Then give every remaining terminal one candidate and fill timeline-wide.
    for later, rows in buckets:
        if later.status in {"complete", "cancelled"}:
            take(rows[0])
    for rank in range(candidates_per_event):
        for _later, rows in buckets:
            if rank < len(rows):
                take(rows[rank])
    if len(links) < max_links:
        remaining = sorted(
            (
                row for _later, rows in buckets for row in rows
                if row not in links
            ),
            key=lambda row: (float(row["score"]), str(row)),
            reverse=True,
        )
        for row in remaining:
            take(row)
    if not links:
        return None
    profiles = []
    for event in events:
        if event.node_id not in allowed:
            continue
        context = "\n".join(
            f"{turn_by_id[value].speaker}: {turn_by_id[value].text}"
            for value in event.source_turn_ids[:2] if value in turn_by_id
        )[:420]
        profiles.append({
            "event_id": event.node_id,
            "session_id": event.session_id,
            "label": event.label,
            "status": event.status,
            "event_time": event.event_time,
            "participant_keys": event.participant_keys,
            "semantic_type_keys": event.semantic_type_keys,
            "source_context": context,
        })
    profile_by_id = {row["event_id"]: row for row in profiles}
    neighborhoods = []
    for target in events:
        if (
            target.node_id not in profile_by_id
            or target.node_id not in explicit_terminal_ids
        ):
            continue
        incoming = [
            row for row in links if row["later_event_id"] == target.node_id
        ]
        if incoming:
            neighborhoods.append({
                "target_event_id": target.node_id,
                "candidate_predecessor_ids": [
                    row["earlier_event_id"] for row in incoming
                ],
            })

    return {
        "event_candidates": profiles,
        "event_candidate_links": links,
        "event_neighborhoods": neighborhoods,
    }


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _decode_cluster(row: Any) -> tuple[list[str], str, list[str], float] | None:
    if isinstance(row, dict):
        members = row.get("member_event_ids") or row.get("members") or []
        label = str(row.get("canonical_label") or row.get("label") or "").strip()
        anchors = row.get("identity_anchors") or row.get("anchors") or []
        confidence = row.get("confidence")
    elif isinstance(row, list) and len(row) >= 4:
        members, label, anchors, confidence = row[:4]
        label = str(label).strip()
    else:
        return None
    if not isinstance(members, list):
        return None
    anchors = [anchors] if isinstance(anchors, str) else anchors
    if not isinstance(anchors, list):
        return None
    try:
        return (
            list(dict.fromkeys(str(value) for value in members if str(value))),
            label,
            [str(value) for value in anchors if str(value).strip()],
            float(confidence),
        )
    except (TypeError, ValueError):
        return None


def _connected(member_ids: list[str], links: set[frozenset[str]]) -> bool:
    members = set(member_ids)
    reached = {member_ids[0]}
    while True:
        updated = reached | {
            value for pair in links if pair & reached
            for value in pair if value in members
        }
        if updated == reached:
            return reached == members
        reached = updated


def parse_event_entities(
    text: str,
    *,
    question_id: str,
    events: list[EventNode],
    candidate_payload: dict[str, Any] | None,
) -> tuple[list[EventEntityNode], list[HyperEdge]]:
    """Parse and locally verify multi-mention event entities."""
    response = _json_object(text)
    rows = response.get("event_clusters", [])
    if not isinstance(rows, list) or not candidate_payload:
        return [], []
    rows = list(rows)
    neighborhoods = {
        str(row.get("target_event_id") or ""): {
            str(value)
            for value in row.get("candidate_predecessor_ids", [])
        }
        for row in candidate_payload.get("event_neighborhoods", [])
        if isinstance(row, dict)
        and isinstance(row.get("candidate_predecessor_ids", []), list)
    }
    decisions = response.get("event_decisions", [])
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, dict):
            continue
        target_id = str(decision.get("target_event_id") or "")
        predecessor_ids = decision.get("predecessor_event_ids") or []
        if (
            target_id not in neighborhoods
            or not isinstance(predecessor_ids, list)
        ):
            continue
        predecessor_ids = list(dict.fromkeys(
            str(value) for value in predecessor_ids if str(value)
        ))
        if (
            not predecessor_ids
            or not set(predecessor_ids) <= neighborhoods[target_id]
        ):
            continue
        rows.append({
            "member_event_ids": [*predecessor_ids, target_id],
            "canonical_label": decision.get("canonical_label") or "",
            "identity_anchors": decision.get("identity_anchors") or [],
            "confidence": decision.get("confidence"),
        })
    event_by_id = {event.node_id: event for event in events}
    position = {event.node_id: index for index, event in enumerate(events)}
    allowed_ids = {
        str(row.get("event_id") or "")
        for row in candidate_payload.get("event_candidates", [])
        if isinstance(row, dict)
    }
    links = {
        frozenset((str(row.get("earlier_event_id") or ""),
                   str(row.get("later_event_id") or "")))
        for row in candidate_payload.get("event_candidate_links", [])
        if isinstance(row, dict)
    }
    accepted = []
    for row in rows:
        decoded = _decode_cluster(row)
        if decoded is None:
            continue
        member_ids, label, anchors, confidence = decoded
        if (
            confidence < 0.88 or not label or not 2 <= len(member_ids) <= 6
            or not set(member_ids) <= allowed_ids
            or not set(member_ids) <= set(event_by_id)
            or not _connected(member_ids, links)
        ):
            continue
        member_ids.sort(key=position.__getitem__)
        members = [event_by_id[value] for value in member_ids]
        common_participants = set(members[0].participant_keys)
        for event in members[1:]:
            common_participants &= set(event.participant_keys)
        term_counts = Counter(term for event in members for term in _event_terms(event))
        type_counts = Counter(
            term for event in members for term in _specific_type_keys(event)
        )
        repeated = {
            term for term, count in (term_counts + type_counts).items() if count >= 2
        }
        grounded = set(term_counts) | set(type_counts)
        declared = {
            token.casefold() for value in [label, *anchors]
            for token in re.findall(r"[\w'-]+", value) if len(token) >= 3
        }
        terminal_members = [
            event for event in members
            if event.status in {"complete", "cancelled"}
        ]
        if (
            len({event.session_id for event in members}) < 2
            or not common_participants or not repeated
            or not declared & grounded or len(terminal_members) > 1
        ):
            continue
        accepted.append((
            confidence, len(declared & repeated), member_ids, label,
            sorted(declared & grounded),
        ))
    accepted.sort(key=lambda row: (row[0], row[1], len(row[2])), reverse=True)
    entities: list[EventEntityNode] = []
    edges: list[HyperEdge] = []
    claimed: set[str] = set()
    for confidence, _strength, member_ids, label, anchors in accepted:
        if claimed & set(member_ids):
            continue
        members = [event_by_id[value] for value in member_ids]
        current = next((
            event for event in reversed(members)
            if event.status in {"complete", "cancelled"}
        ), members[-1])
        times = [event.event_time for event in members if event.event_time]
        sources = list(dict.fromkeys(
            value for event in members for value in event.source_turn_ids
        ))
        participants = sorted(set.intersection(*[
            set(event.participant_keys) for event in members
        ]))
        types = sorted(set(
            value for event in members for value in event.semantic_type_keys
        ))
        entity_id = f"{question_id}:event_entity:{len(entities)}"
        retrieval_text = " | ".join([
            label, f"identity anchors {' '.join(anchors)}",
            f"participants {' '.join(participants)}", f"types {' '.join(types)}",
            f"lifecycle {current.status}",
            *[
                f"{event.label} status {event.status} time {event.event_time or 'unknown'}"
                for event in members
            ],
        ])
        entity = EventEntityNode(
            node_id=entity_id, question_id=question_id,
            canonical_label=label,
            canonical_key=" ".join(sorted(set(re.findall(
                r"[\w'-]+", label.casefold()
            )))),
            member_event_ids=member_ids, anchor_terms=anchors,
            participant_keys=participants, semantic_type_keys=types,
            lifecycle_status=current.status, current_event_id=current.node_id,
            time_start=times[0] if times else None,
            time_end=times[-1] if times else None,
            source_turn_ids=sources, confidence=max(0.0, min(1.0, confidence)),
            retrieval_text=retrieval_text,
        )
        edge = HyperEdge(
            edge_id=f"{question_id}:event_entity_member:{len(edges)}",
            question_id=question_id, relation="event_entity_member",
            incidences=[
                HyperIncidence(entity_id, "identity", 0),
                *[
                    HyperIncidence(event.node_id, f"mention_{event.status}", order + 1)
                    for order, event in enumerate(members)
                ],
            ],
            directed=True, confidence=entity.confidence,
            provenance={
                "generator": "bounded_event_entity_consolidation",
                "event_entity_id": entity_id,
                "identity_anchors": anchors,
                "member_event_ids": member_ids,
            },
            retrieval_text=retrieval_text,
        )
        entities.append(entity)
        edges.append(edge)
        claimed.update(member_ids)
    return entities, edges
