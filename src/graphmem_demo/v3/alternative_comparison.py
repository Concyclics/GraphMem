from __future__ import annotations

import re
from typing import Any

from .action_semantics import action_families, action_family_overlap
from .catalog_schema import OperandRecordV3
from .schema import QueryFrame
from .temporal_normalize import resolve_evidence_time


_WORD_RE = re.compile(r"[a-z0-9]+")
_RELATION_STOP = {
    "a", "an", "are", "did", "do", "does", "event", "first", "happen",
    "happened", "has", "have", "had", "i", "is", "me", "my", "occur",
    "occurred", "the", "was", "were", "what", "which", "who",
}
_ENTITY_STOP = {
    "a", "an", "at", "did", "for", "from", "i", "in", "of", "or",
    "the", "to", "was", "were", "who",
}


def _token_key(value: str) -> str:
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _terms(value: str, ignored: set[str]) -> set[str]:
    return {
        key for token in _WORD_RE.findall(value.casefold())
        if token not in ignored and (key := _token_key(token)) and len(key) > 1
    }


def _alternatives(question: str) -> tuple[str, str] | None:
    match = re.search(
        r",\s*(?:the\s+)?(.+?)\s+or\s+(?:the\s+)?(.+?)[?]?$",
        question.casefold(),
    )
    return (match.group(1).strip(), match.group(2).strip()) if match else None


def earliest_alternative_from_sources(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    turns: list[Any],
) -> dict[str, object] | None:
    """Compare explicit alternatives using entity, relation, time and sources."""

    if frame.requested_operation != "earliest":
        return None
    alternatives = _alternatives(frame.raw_question)
    if alternatives is None:
        return None

    relation_prefix = frame.raw_question.casefold().split(",", 1)[0]
    relation_terms = _terms(relation_prefix, _RELATION_STOP)
    relation_families = action_families(relation_prefix)
    turn_by_id = {
        str(getattr(turn, "node_id", "")): turn
        for turn in turns
    }

    bound: list[tuple[Any, str, OperandRecordV3, str]] = []
    for raw in alternatives:
        entity_terms = _terms(raw, _ENTITY_STOP)
        candidates: list[tuple[float, Any, OperandRecordV3, str]] = []
        for item in operands:
            if item.modality in {"planned", "possible", "hypothetical"}:
                continue
            source_text = " ".join(
                str(getattr(turn_by_id.get(source_id), "text", ""))
                for source_id in item.source_turn_ids
            )
            entity_text = " ".join((
                item.subject_key, item.predicate_key, item.object_text,
                item.context_key,
            ))
            item_entity_terms = _terms(entity_text, _ENTITY_STOP)
            source_entity_terms = _terms(source_text, _ENTITY_STOP)
            direct_shared = entity_terms & item_entity_terms
            source_shared = entity_terms & source_entity_terms
            direct_relation_score = action_family_overlap(
                relation_prefix, item.predicate_key
            )
            alternative_action_match = action_family_overlap(
                raw, item.predicate_key
            )
            if len(entity_terms) >= 2:
                identity_complete = (
                    len(direct_shared) >= 2
                    or (alternative_action_match > 0 and len(direct_shared) >= 1)
                    or (
                        len(direct_shared) >= 1
                        and len(direct_shared | source_shared) >= 2
                        and direct_relation_score > 0
                    )
                )
            else:
                identity_complete = bool(direct_shared)
            if not identity_complete:
                continue
            entity_score = (
                len(direct_shared) / max(1, min(len(entity_terms), len(item_entity_terms)))
                + 0.12 * len(source_shared - direct_shared)
            )
            evidence_text = " ".join((
                item.subject_key, item.predicate_key, item.object_text,
                item.context_key, source_text,
            ))
            if relation_families:
                relation_score = action_family_overlap(relation_prefix, evidence_text)
            elif relation_terms:
                relation_score = len(relation_terms & _terms(evidence_text, set()))
            else:
                relation_score = 1
            if relation_score <= 0:
                continue
            resolved, basis = resolve_evidence_time(item.event_time, item.observed_at)
            # Extraction may normalize a relative event to the observation
            # date. A relation-bound lossless source is more specific; prefer
            # its explicit or relative expression over that coarse fallback.
            source_anchor = next((
                str(getattr(turn_by_id.get(source_id), "session_date", "") or "")
                for source_id in item.source_turn_ids
                if getattr(turn_by_id.get(source_id), "session_date", None)
            ), item.observed_at)
            source_resolved, source_basis = resolve_evidence_time(
                source_text, source_anchor
            )
            if source_resolved is not None and source_basis != "observed_fallback":
                resolved, basis = source_resolved, f"lossless_{source_basis}"
            if resolved is None:
                continue
            candidates.append((
                entity_score + min(relation_score, 2) * 0.35
                + min(direct_relation_score, 1) * 0.55
                + 0.05 * item.confidence,
                resolved,
                item,
                basis,
            ))
        if not candidates:
            return {
                "operation": "named_alternative_incomplete",
                "value": "insufficient evidence for every named alternative",
                "missing_alternatives": [raw],
                "operand_ids": [],
                "source_turn_ids": [],
                "complete": True,
            }
        _score, resolved, item, basis = max(
            candidates,
            key=lambda row: (row[0], row[1], row[2].operand_id),
        )
        bound.append((resolved, raw, item, basis))

    if bound[0][2].operand_id == bound[1][2].operand_id:
        return {
            "operation": "named_alternative_incomplete",
            "value": "alternatives do not have distinct relation-bound evidence",
            "missing_alternatives": list(alternatives),
            "operand_ids": [],
            "source_turn_ids": [],
            "complete": True,
        }

    resolved, raw, winner, _basis = min(
        bound, key=lambda row: (row[0], row[1])
    )
    sources = list(dict.fromkeys(
        source_id
        for _time, _raw, item, _time_basis in bound
        for source_id in item.source_turn_ids
    ))
    return {
        "operation": "earliest_named_alternative",
        "value": raw,
        "date": resolved.date().isoformat(),
        "operand_ids": [item.operand_id for _time, _raw, item, _basis in bound],
        "source_turn_ids": sources,
        "proofs": [
            {
                "alternative": alternative,
                "resolved_date": time.date().isoformat(),
                "time_basis": basis,
                "operand_id": item.operand_id,
                "source_turn_ids": list(item.source_turn_ids),
            }
            for time, alternative, item, basis in bound
        ],
        "complete": True,
        "completion_basis": "distinct_entity_relation_time_source_closure",
    }
