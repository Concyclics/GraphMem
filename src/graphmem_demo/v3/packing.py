from __future__ import annotations

import re
from typing import Any

from ..clients import rough_token_count
from .schema import ClaimNode, EpisodeNode, EventNode, QueryFrame, ThemeNode, TurnNode


_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {value.casefold() for value in _WORD_RE.findall(text)}


def _overlap(frame: QueryFrame, text: str) -> float:
    query = set(
        frame.content_terms + frame.participant_terms + frame.temporal_terms
    )
    return len(query & _tokens(text)) / max(1, len(query))


def _focused_text(text: str, frame: QueryFrame, limit: int) -> str:
    if len(text) <= limit:
        return text
    segments = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    ]
    ranked = sorted(
        enumerate(segments),
        key=lambda item: (
            _overlap(frame, item[1]),
            len(set(frame.content_terms) & _tokens(item[1])),
            -item[0],
        ),
        reverse=True,
    )
    chosen = sorted(index for index, _value in ranked[:4])
    return " ".join(segments[index] for index in chosen)[:limit]


def render_block(frame: QueryFrame, kind: str, node: Any) -> str:
    if isinstance(node, TurnNode):
        return (
            f"[TURN {node.node_id} | session={node.session_id} | "
            f"date={node.session_date or 'unknown'} | speaker={node.speaker}]\n"
            f"{_focused_text(node.text, frame, 1400)}"
        )
    if isinstance(node, ClaimNode):
        return (
            f"[CLAIM {node.node_id} | "
            f"time={node.event_time or node.observed_at or 'unknown'} | "
            f"modality={node.modality} | polarity={node.polarity} | "
            f"sources={','.join(node.source_turn_ids)}]\n"
            f"{node.subject} | {node.predicate} | "
            f"{_focused_text(node.object, frame, 900)}"
        )
    if isinstance(node, EventNode):
        return (
            f"[EVENT {node.node_id} | time={node.event_time or 'unknown'} | "
            f"status={node.status} | sources={','.join(node.source_turn_ids)}]\n"
            f"{_focused_text(node.label, frame, 700)}"
        )
    if isinstance(node, EpisodeNode):
        return (
            f"[EPISODE {node.node_id} | session={node.session_id} | "
            f"turns={','.join(node.turn_ids)}]\n"
            f"{_focused_text(node.retrieval_text, frame, 1000)}"
        )
    if isinstance(node, ThemeNode):
        return (
            f"[THEME {node.node_id} | episodes={','.join(node.episode_ids)}]\n"
            f"{_focused_text(node.retrieval_text, frame, 800)}"
        )
    return f"[{kind.upper()}]\n{_focused_text(str(node), frame, 800)}"


def pack_context(
    frame: QueryFrame,
    ordered: list[tuple[str, Any, float, str]],
    budget: int,
) -> tuple[list[tuple[str, Any, float, str]], str, list[dict[str, Any]]]:
    prepared = []
    for kind, node, score, source in ordered:
        block = render_block(frame, kind, node)
        cost = rough_token_count(block)
        priority = (
            1.20 if source == "protected_direct" else
            0.82 if source == "protected_graph_rescue" else
            0.26 if source == "provenance_expansion" else 0.0
        )
        priority += 0.55 * _overlap(frame, getattr(node, "retrieval_text", ""))
        if kind in {"claim", "event"}:
            priority += 0.18
        elif kind in {"episode", "theme"}:
            priority += 0.10
        priority += min(score, 2.0) * 0.12
        priority -= min(cost, 1200) / 6000.0
        prepared.append((priority, kind, node, score, source, block, cost))
    prepared.sort(
        key=lambda item: (
            item[0],
            -item[6],
            getattr(item[2], "node_id", ""),
        ),
        reverse=True,
    )
    kept: list[tuple[str, Any, float, str]] = []
    blocks: list[str] = []
    decisions: list[dict[str, Any]] = []
    used = 0
    for priority, kind, node, score, source, block, cost in prepared:
        decision = "keep" if used + cost <= budget else "drop_budget"
        decisions.append(
            {
                "node_id": getattr(node, "node_id", ""),
                "decision": decision,
                "rough_tokens": cost,
                "source": source,
                "pack_priority": round(priority, 6),
            }
        )
        if decision != "keep":
            continue
        kept.append((kind, node, score, source))
        blocks.append(block)
        used += cost
    return kept, "\n\n".join(blocks), decisions
