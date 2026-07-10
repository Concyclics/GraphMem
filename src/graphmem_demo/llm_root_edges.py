from __future__ import annotations

import json
import re
from typing import Any

from .models import GraphEdge, SummaryNode

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "how",
    "many",
    "much",
    "your",
    "user",
    "session",
    "summary",
}


def llm_root_anchor_messages(root: SummaryNode) -> list[dict[str, str]]:
    text = root.retrieval_text or root.summary or root.raw_summary_text
    prompt = (
        "Extract normalized anchor terms from this session summary.\n"
        "Return STRICT JSON object with keys: entities, events, times, state_phrases, keywords.\n"
        "Each value must be a short string list, no explanations.\n"
        "Use canonical forms and deduplicate semantically equivalent mentions.\n\n"
        f"Session id: {root.session_id}\n"
        f"Session date: {root.session_date or 'unknown'}\n"
        "Summary text:\n"
        f"{text}\n"
    )
    return [{"role": "user", "content": prompt}]


def parse_llm_root_anchors(text: str, *, max_items_per_key: int = 8) -> dict[str, list[str]]:
    payload: dict[str, Any] | None = None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            payload = loaded
    except json.JSONDecodeError:
        payload = _extract_first_json_object(text)
    if payload is None:
        return {}

    parsed: dict[str, list[str]] = {}
    for key in ("entities", "events", "times", "state_phrases", "keywords"):
        values = payload.get(key)
        if not isinstance(values, list):
            parsed[key] = []
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = _normalize_term(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
            if len(cleaned) >= max_items_per_key:
                break
        parsed[key] = cleaned
    return parsed


def build_llm_anchor_edges(
    roots: list[SummaryNode],
    llm_anchors: dict[str, dict[str, list[str]]],
    *,
    neighbors_per_relation: int,
    min_shared: int,
) -> list[GraphEdge]:
    if not roots:
        return []
    anchors_by_id = {root.node_id: llm_anchors.get(root.node_id, {}) for root in roots}
    neighbors: dict[str, list[tuple[float, SummaryNode, str]]] = {root.node_id: [] for root in roots}

    relation_specs = (
        ("entities", "entity_neighbor", 0.88, 0.03),
        ("times", "time_neighbor", 0.78, 0.03),
        ("events", "event_neighbor", 0.70, 0.03),
        ("state_phrases", "state_neighbor", 0.74, 0.02),
        ("keywords", "keyword_neighbor", 0.62, 0.02),
    )
    for index, root in enumerate(roots):
        left = anchors_by_id[root.node_id]
        for candidate in roots[index + 1 :]:
            right = anchors_by_id[candidate.node_id]
            for key, relation, base, step in relation_specs:
                shared_count = len(set(left.get(key, [])) & set(right.get(key, [])))
                if shared_count < min_shared:
                    continue
                score = min(0.98, base + shared_count * step)
                neighbors[root.node_id].append((score, candidate, relation))
                neighbors[candidate.node_id].append((score, root, relation))
            if _is_update_neighbor(left, right):
                neighbors[root.node_id].append((0.84, candidate, "update_neighbor"))
                neighbors[candidate.node_id].append((0.84, root, "update_neighbor"))

    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for root in roots:
        relation_budget: dict[str, int] = {}
        for score, candidate, relation in sorted(
            neighbors[root.node_id], key=lambda item: (item[0], item[2], item[1].node_id), reverse=True
        ):
            if relation_budget.get(relation, 0) >= neighbors_per_relation:
                continue
            pair = tuple(sorted((root.node_id, candidate.node_id)))
            key = (pair[0], pair[1], relation)
            if key in seen:
                continue
            edges.append(
                GraphEdge(
                    src=pair[0],
                    dst=pair[1],
                    score=score,
                    relation=relation,  # type: ignore[arg-type]
                )
            )
            seen.add(key)
            relation_budget[relation] = relation_budget.get(relation, 0) + 1
    return edges


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


def _normalize_term(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" \t\r\n.,;:")).casefold()
    if len(text) < 2:
        return ""
    if text in _STOPWORDS:
        return ""
    return text


def _is_update_neighbor(left: dict[str, list[str]], right: dict[str, list[str]]) -> bool:
    left_events = set(left.get("events", []))
    right_events = set(right.get("events", []))
    if not (left_events & right_events):
        return False
    left_entities = set(left.get("entities", [])) | set(left.get("keywords", []))
    right_entities = set(right.get("entities", [])) | set(right.get("keywords", []))
    return bool(left_entities & right_entities)
