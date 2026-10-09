"""Compact, deterministic relation-route labels for source evidence.

The labels expose how a source turn entered the query plan without copying a
generated fact summary into the answer context.  They are presentation
metadata only; the original source turn remains the factual authority.
"""
from __future__ import annotations

from collections.abc import Iterable


RELATION_ROUTE_FAMILY = {
    "shared_entity": "entity",
    "same_entity_state": "entity",
    "resolves_to": "entity",
    "owned_by": "entity",
    "has_actor": "entity",
    "has_object": "entity",
    "has_state": "state",
    "state_transition": "state",
    "state_next": "state",
    "replacement": "state",
    "contradiction_update": "state",
    "temporal_before": "time",
    "temporal_after": "time",
    "temporal_continuation": "time",
    "at_time": "time",
    "member_of": "collection",
    "collection_co_member": "collection",
    "contains": "collection",
    "in_scope": "collection",
    "scene_contains": "hierarchy",
    "refines_to": "hierarchy",
    "portal": "hierarchy",
    "coarse_related": "semantic",
    "same_event": "semantic",
    "same_activity": "semantic",
    "same_preference_domain": "semantic",
    "shared_value": "semantic",
    "shared_referent": "lexical",
    "causal": "causal",
    "dialogue_pair": "dialogue",
    "coreference": "coreference",
}


def relation_route_hint(
    source_channels: Iterable[str] = (),
    relation_contributions: Iterable[str] = (),
) -> str:
    """Return a bounded ``{via=...}`` hint from non-generative metadata."""

    channels = frozenset(str(value) for value in source_channels)
    families: list[str] = []
    if "semantic_fact" in channels:
        families.append("fact")
    if "semantic_predicate" in channels:
        families.append("relation")
    for relation in relation_contributions:
        family = RELATION_ROUTE_FAMILY.get(str(relation), "")
        if family and family not in families:
            families.append(family)
    return f" {{via={'>'.join(families[:4])}}}" if families else ""


__all__ = ["RELATION_ROUTE_FAMILY", "relation_route_hint"]
