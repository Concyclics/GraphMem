from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

from ..clients import rough_token_count


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def conservative_prompt_tokens(system: str, user: str) -> int:
    text = system + "\n" + user
    return max(rough_token_count(text), math.ceil(len(text) / 3.0))



_NESTED_TEXT_LIMIT = 720
_NESTED_LIST_LIMIT = 24
_ID_LIST_LIMIT = 16


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + " …"


def _compact_structured_value(value: Any, *, key: str = "") -> Any:
    """Bound routing/operator diagnostics while preserving answer operands.

    Structured hints may cite the same lossless turns that are already present in
    the evidence pack.  Their nested evidence prose and graph traversal ID lists
    are audit metadata, not additional answer evidence, so they need their own
    size policy instead of competing with the evidence blocks.
    """
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key == "evidence" and isinstance(child_value, str):
                compacted[child_key] = _bounded_text(child_value, _NESTED_TEXT_LIMIT)
                if len(child_value) > _NESTED_TEXT_LIMIT:
                    compacted["evidence_original_chars"] = len(child_value)
                continue
            compacted[child_key] = _compact_structured_value(
                child_value, key=child_key
            )
        return compacted
    if isinstance(value, list):
        limit = (
            _ID_LIST_LIMIT
            if key.endswith("_ids") or key.endswith("_node_ids")
            else _NESTED_LIST_LIMIT
        )
        return [
            _compact_structured_value(item, key=key)
            for item in value[:limit]
        ]
    return value


def _compact_structured_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    compacted = deepcopy(payload)
    changed: list[str] = []
    for key, value in list(compacted.items()):
        if key in {"evidence", "query_focused_evidence"}:
            continue
        bounded = _compact_structured_value(value, key=key)
        if bounded != value:
            compacted[key] = bounded
            changed.append(key)

    focused = compacted.get("query_focused_evidence")
    if isinstance(focused, str) and len(focused) > 6000:
        compacted["query_focused_evidence"] = _bounded_text(focused, 6000)
        changed.append("query_focused_evidence")
    return compacted, changed


def contract_evidence_to_ids(evidence: str, node_ids: set[str]) -> str:
    """Keep only evidence blocks grounded in a complete local closure."""

    if not evidence or not node_ids:
        return ""
    selected: list[str] = []
    for block in evidence.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        header = block.split("\n", 1)[0]
        if any(node_id in header for node_id in node_ids):
            selected.append(block)
    return "\n\n".join(selected)


def fit_answer_payload(
    system: str,
    payload: dict[str, Any],
    *,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Fit the complete request, not merely its evidence field."""
    fitted, compacted_structured_fields = _compact_structured_payload(payload)
    evidence = str(fitted.get("evidence") or "")
    blocks = [value for value in evidence.split("\n\n") if value.strip()]
    initial_blocks = len(blocks)

    # Routing diagnostics are useful but never more important than cited evidence.
    fitted["scope_posteriors"] = list(fitted.get("scope_posteriors") or [])[:2]
    fitted["temporal_evidence_ledger_newest_first"] = list(
        fitted.get("temporal_evidence_ledger_newest_first") or []
    )[:4]
    fitted["recommendation_constraints"] = list(
        fitted.get("recommendation_constraints") or []
    )[:4]

    user = _serialize(fitted)
    initial_estimate = conservative_prompt_tokens(system, user)
    while len(blocks) > 1 and conservative_prompt_tokens(system, user) > max_prompt_tokens:
        blocks.pop()
        fitted["evidence"] = "\n\n".join(blocks)
        user = _serialize(fitted)

    optional = (
        "scope_posteriors",
        "primary_scope_hint",
        "temporal_evidence_ledger_newest_first",
        "recommendation_constraints",
        "newest_scalar_hint",
        "catalog_candidate_hint",
        "reference_chain_hints",
        "event_identity_chain_hints",
        "event_entity_hints",
        "query_focused_evidence",
    )
    removed_optional: list[str] = []
    for key in optional:
        if conservative_prompt_tokens(system, user) <= max_prompt_tokens:
            break
        if fitted.get(key):
            fitted[key] = [] if isinstance(fitted[key], list) else None
            removed_optional.append(key)
            user = _serialize(fitted)

    if conservative_prompt_tokens(system, user) > max_prompt_tokens and blocks:
        # Last-resort bounded truncation keeps the highest-priority first block.
        excess = conservative_prompt_tokens(system, user) - max_prompt_tokens
        keep_chars = max(320, len(fitted["evidence"]) - excess * 4)
        fitted["evidence"] = fitted["evidence"][:keep_chars]
        user = _serialize(fitted)

    final_estimate = conservative_prompt_tokens(system, user)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], {
        "max_prompt_tokens": max_prompt_tokens,
        "initial_estimated_prompt_tokens": initial_estimate,
        "final_estimated_prompt_tokens": final_estimate,
        "initial_evidence_blocks": initial_blocks,
        "final_evidence_blocks": len(blocks),
        "removed_optional_fields": removed_optional,
        "compacted_structured_fields": compacted_structured_fields,
        "fit_pass": final_estimate <= max_prompt_tokens,
    }
