from __future__ import annotations

import json
import re
from typing import Any

from .models import GraphEdge, LeafNode

_LEAF_EDGE_RELATIONS = {
    "temporal_neighbor",
    "entity_neighbor",
    "event_neighbor",
    "update_neighbor",
    "state_neighbor",
    "keyword_neighbor",
}

_RELATION_ALIASES = {
    "temporal": "temporal_neighbor",
    "entity": "entity_neighbor",
    "entity_link": "entity_neighbor",
    "event": "event_neighbor",
    "same_event": "event_neighbor",
    "update": "update_neighbor",
    "state": "state_neighbor",
    "keyword": "keyword_neighbor",
}


def llm_session_leaf_edge_messages(
    session_id: str,
    leaves: list[LeafNode],
    *,
    max_snippet_chars: int = 1024,
) -> list[dict[str, str]]:
    """One LLM call per session: propose semantic links between its leaves."""
    lines: list[str] = []
    for leaf in sorted(leaves, key=lambda item: item.turn_index):
        text = (leaf.retrieval_text or leaf.user_text or leaf.raw_text or "").strip()
        if len(text) > max_snippet_chars:
            text = text[: max_snippet_chars - 3] + "..."
        lines.append(
            f'- leaf_id="{leaf.node_id}" turn={leaf.turn_index}: {text.replace(chr(10), " ")}'
        )
    allowed = ", ".join(sorted(_LEAF_EDGE_RELATIONS))
    system_content = (
        "You identify high-confidence semantic links between conversation turns (leaves) "
        "within ONE session. Return STRICT JSON only:\n"
        '{"edges": [{"src": "<leaf_id>", "dst": "<leaf_id>", "relation": "<relation>", '
        '"confidence": <0.0-1.0>}]}\n'
        f"Allowed relation values: {allowed}.\n"
        "Only link turns with clear, QA-relevant support: same concrete event/entity from "
        "different angles, explicit before/after updates on the same entity, or strong "
        "non-adjacent continuation of the same fact thread.\n"
        "Precision over recall: assign confidence >= 0.8 only when the link is clearly "
        "supported by the text. Use 0.9+ for unambiguous entity/event/update links. "
        "Do NOT link turns that merely share a broad topic. Avoid keyword_neighbor unless "
        "the same named entity appears in both turns. Use only leaf_id values from the list. "
        "If no strong links exist, return {\"edges\": []}. JSON only."
    )
    user_content = (
        f"Session id: {session_id}\n"
        f"Leaves ({len(leaves)}):\n" + "\n".join(lines)
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parse_llm_leaf_edges(
    text: str,
    *,
    valid_leaf_ids: set[str],
    min_confidence: float,
    max_edges_per_leaf: int = 3,
    max_edges_per_session: int = 16,
) -> list[GraphEdge]:
    payload = _extract_first_json_object(text)
    if payload is None:
        return []
    raw_edges = payload.get("edges")
    if not isinstance(raw_edges, list):
        return []
    candidates: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src") or "").strip()
        dst = str(item.get("dst") or "").strip()
        if not src or not dst or src == dst:
            continue
        if src not in valid_leaf_ids or dst not in valid_leaf_ids:
            continue
        relation = _normalize_relation(item.get("relation"))
        if relation is None:
            continue
        confidence = _coerce_confidence(item.get("confidence"))
        if confidence < min_confidence:
            continue
        pair = tuple(sorted((src, dst)))
        key = (pair[0], pair[1], relation)
        if key in seen:
            continue
        candidates.append(GraphEdge(src=pair[0], dst=pair[1], score=confidence, relation=relation))
        seen.add(key)
    return prune_llm_leaf_edges(
        candidates,
        max_edges_per_leaf=max_edges_per_leaf,
        max_edges_per_session=max_edges_per_session,
    )


def prune_llm_leaf_edges(
    edges: list[GraphEdge],
    *,
    max_edges_per_leaf: int,
    max_edges_per_session: int,
) -> list[GraphEdge]:
    """Keep highest-confidence edges under per-leaf and per-session caps."""
    if not edges or max_edges_per_session <= 0:
        return []
    if max_edges_per_leaf <= 0:
        return []
    ranked = sorted(
        edges,
        key=lambda edge: (-edge.score, edge.relation, edge.src, edge.dst),
    )
    kept: list[GraphEdge] = []
    leaf_degree: dict[str, int] = {}
    for edge in ranked:
        if len(kept) >= max_edges_per_session:
            break
        if leaf_degree.get(edge.src, 0) >= max_edges_per_leaf:
            continue
        if leaf_degree.get(edge.dst, 0) >= max_edges_per_leaf:
            continue
        kept.append(edge)
        leaf_degree[edge.src] = leaf_degree.get(edge.src, 0) + 1
        leaf_degree[edge.dst] = leaf_degree.get(edge.dst, 0) + 1
    return kept


def build_deterministic_session_leaf_edges(leaves: list[LeafNode]) -> list[GraphEdge]:
    """Adjacent-turn links within a session (zero LLM cost)."""
    if len(leaves) < 2:
        return []
    ordered = sorted(leaves, key=lambda item: (item.turn_index, item.node_id))
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for left, right in zip(ordered, ordered[1:]):
        if left.session_id != right.session_id:
            continue
        pair = tuple(sorted((left.node_id, right.node_id)))
        key = (pair[0], pair[1], "temporal_neighbor")
        if key in seen:
            continue
        edges.append(
            GraphEdge(src=pair[0], dst=pair[1], score=0.95, relation="temporal_neighbor")
        )
        seen.add(key)
    return edges


def build_session_leaf_edges(
    leaves_by_session: dict[str, list[LeafNode]],
    *,
    enable_deterministic: bool,
    llm_edges_by_session: dict[str, list[GraphEdge]] | None = None,
) -> list[GraphEdge]:
    merged: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    llm_edges_by_session = llm_edges_by_session or {}

    def add(edge: GraphEdge) -> None:
        key = (edge.src, edge.dst, edge.relation)
        if key in seen:
            return
        merged.append(edge)
        seen.add(key)

    for session_id, session_leaves in leaves_by_session.items():
        if enable_deterministic:
            for edge in build_deterministic_session_leaf_edges(session_leaves):
                add(edge)
        for edge in llm_edges_by_session.get(session_id, []):
            add(edge)
    return merged


def _normalize_relation(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _LEAF_EDGE_RELATIONS:
        return text  # type: ignore[return-value]
    mapped = _RELATION_ALIASES.get(text)
    if mapped in _LEAF_EDGE_RELATIONS:
        return mapped  # type: ignore[return-value]
    return None


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        match = re.search(r"0?\.\d+|1\.0|1|0", value)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(0))))
            except ValueError:
                return 0.0
    return 0.0


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
