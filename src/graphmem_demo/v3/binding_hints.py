from __future__ import annotations

import re
from typing import Any, Callable

from .schema import QueryFrame


def missing_count_target_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    node_text: Callable[[Any], str],
    tokenize: Callable[[str], list[str]],
) -> dict[str, object] | None:
    """Refuse sibling-category substitution when an exact count target is absent."""
    if frame.requested_operation != "count":
        return None
    match = re.search(
        r"\bhow many\s+(.+?)\s+"
        r"(?:am|are|did|do|does|had|has|have|is|was|were|will|would)\b",
        frame.raw_question.casefold(),
    )
    if match is None:
        return None
    ignored = {
        "amount", "event", "item", "kind", "number", "occurrence",
        "thing", "time", "type",
    }
    heads: set[str] = set()
    for alternative in re.split(r"\bor\b", match.group(1)):
        terms = [term for term in tokenize(alternative) if term not in ignored]
        if terms:
            heads.add(terms[-1])
    if not heads:
        return None
    evidence_terms = {
        term
        for _kind, node, _score, _source in kept
        for term in tokenize(node_text(node))
    }
    missing = sorted(heads - evidence_terms)
    # An alternative target (A or B) is answerable when any named head is present.
    if heads & evidence_terms:
        return None
    return {
        "operation": "missing_count_target",
        "target_heads": sorted(heads),
        "missing_heads": missing,
        "value": "insufficient evidence for the exact requested count target",
        "supporting_node_ids": [],
        "complete": True,
    }


def missing_possessive_anchor_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    node_text: Callable[[Any], str],
    tokenize: Callable[[str], list[str]],
) -> dict[str, object] | None:
    """Detect a missing explicit possessor without equating sibling relations."""
    match = re.search(
        r"\bmy\s+([a-z][\w-]+)'s\b", frame.raw_question.casefold()
    )
    if match is None:
        return None
    anchor = match.group(1)
    evidence_terms = {
        term
        for _kind, node, _score, _source in kept
        for term in tokenize(node_text(node))
    }
    if anchor in evidence_terms:
        return None
    return {
        "operation": "missing_possessive_anchor",
        "anchor": anchor,
        "value": "insufficient evidence for the exact requested relationship",
        "supporting_node_ids": [],
        "complete": True,
    }
