from __future__ import annotations

from typing import Any, Callable

from .schema import ClaimNode, EpisodeNode


def project_reached_episodes(
    *,
    nodes: dict[str, Any],
    expanded_scores: dict[str, float],
    primary_similarity: Callable[[Any], float],
    slot_similarity: Callable[[Any], float],
    query_overlap: Callable[[Any], float],
    episode_limit: int = 6,
    per_episode_limit: int = 2,
    total_limit: int = 12,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Project reached coarse episodes to a small set of their fine members.

    Graph traversal depth controls semantic exploration.  Descending from an
    already reached episode to its own provenance is a representation
    operation, not another semantic hop, so it is handled here under explicit
    per-episode and global quotas.
    """
    episode_rows: list[tuple[float, str, EpisodeNode]] = []
    max_expanded = max(expanded_scores.values(), default=1.0) or 1.0
    for node_id, graph_score in expanded_scores.items():
        node = nodes.get(node_id)
        if not isinstance(node, EpisodeNode):
            continue
        score = (
            0.35 * graph_score / max_expanded
            + 0.25 * max(0.0, primary_similarity(node))
            + 0.40 * max(0.0, slot_similarity(node))
        )
        episode_rows.append((score, node_id, node))
    episode_rows.sort(reverse=True, key=lambda row: (row[0], row[1]))

    selected: list[str] = []
    trace: list[dict[str, Any]] = []
    seen: set[str] = set()
    for episode_score, episode_id, episode in episode_rows[:episode_limit]:
        fine_ids = [
            node_id
            for node_id in [*episode.claim_ids, *episode.event_ids, *episode.turn_ids]
            if node_id in nodes
        ]
        fine_rows = []
        for node_id in fine_ids:
            node = nodes[node_id]
            kind_bonus = 0.06 if hasattr(node, "source_turn_ids") else 0.0
            score = (
                0.30 * max(0.0, primary_similarity(node))
                + 0.45 * max(0.0, slot_similarity(node))
                + 0.25 * max(0.0, query_overlap(node))
                + kind_bonus
            )
            fine_rows.append((score, node_id))
        fine_rows.sort(reverse=True, key=lambda row: (row[0], row[1]))

        episode_selected: list[str] = []
        for fine_score, node_id in fine_rows:
            if node_id in seen:
                continue
            seen.add(node_id)
            selected.append(node_id)
            episode_selected.append(node_id)
            trace.append(
                {
                    "episode_id": episode_id,
                    "episode_score": round(episode_score, 6),
                    "node_id": node_id,
                    "fine_score": round(fine_score, 6),
                }
            )
            # A selected atomic claim/event must remain losslessly grounded.
            # The source turn is charged to the same small projection quota.
            source_ids = [
                source_id
                for source_id in getattr(nodes[node_id], "source_turn_ids", [])
                if source_id in nodes and source_id not in seen
            ]
            if (
                source_ids
                and len(episode_selected) < per_episode_limit
                and len(selected) < total_limit
            ):
                source_id = max(
                    source_ids,
                    key=lambda value: (
                        slot_similarity(nodes[value]),
                        primary_similarity(nodes[value]),
                        value,
                    ),
                )
                seen.add(source_id)
                selected.append(source_id)
                episode_selected.append(source_id)
                trace.append(
                    {
                        "episode_id": episode_id,
                        "episode_score": round(episode_score, 6),
                        "node_id": source_id,
                        "fine_score": round(
                            0.55 * slot_similarity(nodes[source_id])
                            + 0.45 * primary_similarity(nodes[source_id]),
                            6,
                        ),
                        "via_source_node_id": node_id,
                    }
                )
            if (
                len(episode_selected) >= per_episode_limit
                or len(selected) >= total_limit
            ):
                break
        if len(selected) >= total_limit:
            break
    return selected, trace


def project_routed_claim_sources(
    *,
    nodes: dict[str, Any],
    scope_session_ids: list[str],
    primary_similarity: Callable[[Any], float],
    slot_similarity: Callable[[Any], float],
    query_overlap: Callable[[Any], float],
    session_limit: int = 6,
    per_session_limit: int = 2,
    total_limit: int = 10,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Project answer-compatible claims in routed sessions to source turns."""
    claims_by_session: dict[str, list[ClaimNode]] = {}
    for node in nodes.values():
        if not isinstance(node, ClaimNode):
            continue
        if node.predicate_key in {"said", "mentioned", "discussed"}:
            continue
        claims_by_session.setdefault(node.session_id, []).append(node)

    selected: list[str] = []
    trace: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for session_rank, session_id in enumerate(scope_session_ids[:session_limit]):
        ranked: list[tuple[float, ClaimNode, float, float, float]] = []
        for claim in claims_by_session.get(session_id, []):
            primary = max(0.0, float(primary_similarity(claim)))
            slot = max(0.0, float(slot_similarity(claim)))
            overlap = max(0.0, float(query_overlap(claim)))
            if max(primary, slot, overlap) <= 0.0:
                continue
            score = 0.30 * primary + 0.45 * slot + 0.25 * overlap
            score *= 0.75 + 0.25 * max(0.0, min(1.0, float(claim.confidence)))
            ranked.append((score, claim, primary, slot, overlap))
        ranked.sort(key=lambda row: (-row[0], row[1].node_id))

        session_quota = per_session_limit if session_rank < 3 else 1
        accepted = 0
        for score, claim, primary, slot, overlap in ranked:
            candidates = [
                source_id
                for source_id in claim.source_turn_ids
                if source_id in nodes and source_id not in seen_sources
            ]
            if not candidates:
                continue
            source_id = max(
                candidates,
                key=lambda item: (
                    slot_similarity(nodes[item]),
                    primary_similarity(nodes[item]),
                    query_overlap(nodes[item]),
                    item,
                ),
            )
            seen_sources.add(source_id)
            selected.append(source_id)
            trace.append(
                {
                    "session_id": session_id,
                    "session_rank": session_rank,
                    "claim_id": claim.node_id,
                    "source_id": source_id,
                    "score": round(score, 6),
                    "primary_similarity": round(primary, 6),
                    "slot_similarity": round(slot, 6),
                    "query_overlap": round(overlap, 6),
                }
            )
            accepted += 1
            if accepted >= session_quota or len(selected) >= total_limit:
                break
        if len(selected) >= total_limit:
            break
    return selected, trace
