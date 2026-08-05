from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..domain import FactBinding, OperandSpec, TemporalKey, stable_id
from ..text import content_terms, normalize_key


def _text(attributes, key: str) -> str | None:
    """Read an optional string attribute without inventing the literal 'None'.

    ``str(attrs.get(key)) or None`` yields the four-character string ``"None"``
    whenever the attribute is absent, which then reads downstream as a real
    owner id, event instance or interval.
    """
    value = attributes.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _binding(operand_id: str, node, score: float, path: tuple[str, ...]) -> FactBinding:
    attrs = node.attributes
    return FactBinding(
        stable_id("binding", operand_id, node.node_id), operand_id, node.node_id,
        _text(attrs, "owner_id"), str(attrs.get("predicate", "")), str(attrs.get("scope", "")),
        normalize_key(str(attrs.get("value_key", attrs.get("value", "")))),
        _text(attrs, "event_instance_id"),
        TemporalKey.from_attribute(attrs.get("time_interval")),
        tuple(node.all_evidence_group_ids), path, score,
        value=str(attrs.get("value", "")),
        value_type=str(attrs.get("value_type", "text") or "text"),
        polarity=str(attrs.get("polarity", "positive") or "positive"),
        modality=str(attrs.get("modality", "asserted") or "asserted"),
        session_id=str(attrs.get("session_id", "") or ""),
        turn_index=int(attrs.get("turn_index", -1) or -1),
        collection_key=normalize_key(str(attrs.get("collection_key", "") or "")),
        node_type=node.node_type.value,
    )


def binding_score(node, operand: OperandSpec, owner_ids: set[str]) -> float:
    attrs = node.attributes
    score = 0.0
    owner = str(attrs.get("owner_id", ""))
    if not operand.owner_aliases or owner in owner_ids:
        score += 0.55
    predicate = normalize_key(str(attrs.get("predicate", "")))
    if operand.predicate_candidates:
        overlap = max((len(content_terms(predicate) & content_terms(value))
                       for value in operand.predicate_candidates), default=0)
        score += min(0.35, overlap * 0.18)
    if operand.scope_candidates and normalize_key(str(attrs.get("scope", ""))) in {
        normalize_key(value) for value in operand.scope_candidates
    }:
        score += 0.10
    return score


def bind_facts(view, operand_owners: dict[str, set[str]], operands: Iterable[OperandSpec],
               fact_node_ids: Iterable[str], paths: dict[str, tuple[str, ...]] | None = None) -> tuple[FactBinding, ...]:
    paths = paths or {}
    rows: list[FactBinding] = []
    for operand in operands:
        owners = operand_owners.get(operand.operand_id, set())
        for node_id in fact_node_ids:
            node = view.nodes.get(node_id)
            if not node or node.node_type.value != "canonical_fact":
                continue
            score = binding_score(node, operand, owners)
            if score <= 0.0:
                continue
            rows.append(_binding(operand.operand_id, node, score, paths.get(node_id, ())))
    dedup = {row.binding_id: row for row in rows}
    return tuple(sorted(dedup.values(), key=lambda row: (-row.confidence, row.binding_id)))


def by_operand(rows: Iterable[FactBinding]) -> dict[str, tuple[FactBinding, ...]]:
    grouped: dict[str, list[FactBinding]] = defaultdict(list)
    for row in rows:
        grouped[row.operand_id].append(row)
    return {key: tuple(sorted(value, key=lambda row: (-row.confidence, row.binding_id)))
            for key, value in grouped.items()}
