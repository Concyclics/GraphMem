from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Callable

from .schema import ClaimNode, QueryFrame


def score_scope_posteriors(
    frame: QueryFrame,
    nodes: dict[str, Any],
    channels: dict[str, list[str]],
    rrf_scores: dict[str, float],
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
    query_overlap: Callable[[QueryFrame, str], float],
) -> list[dict[str, Any]]:
    """Score sessions by grounded claim fit, with lexical evidence as fallback.

    The score is deliberately continuous: it changes routing priority but never
    filters a session.  Self-reference is resolved against the memory owner's
    canonical participant key, while named questions match the named subject.
    """

    query_terms = set(frame.content_terms)
    raw_query_terms = set(tokenize(frame.raw_question))
    self_query = bool(
        raw_query_terms & {"i", "me", "my", "mine", "we", "our", "ours"}
    )
    named_subjects = set(frame.participant_terms)

    def subject_compatibility(node: ClaimNode) -> float:
        subject_key = re.sub(
            r"[^\w]+", "_", (node.subject_key or node.subject).casefold()
        ).strip("_")
        if self_query:
            return 1.0 if subject_key in {"participant_1", "user"} else 0.0
        if named_subjects:
            return 1.0 if named_subjects & set(tokenize(subject_key)) else 0.0
        return 0.65

    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for node_id, node in nodes.items():
        session_id = getattr(node, "session_id", None)
        if session_id:
            grouped[str(session_id)].append((node_id, node))

    channel_top = {name: set(values[:80]) for name, values in channels.items()}
    max_rrf = max(rrf_scores.values(), default=1.0)
    rows: list[dict[str, Any]] = []
    for session_id, values in grouped.items():
        covered: set[str] = set()
        best_overlap = 0.0
        relevant_count = 0
        best_rrf = 0.0
        best_claim_joint = 0.0
        best_claim_predicate = 0.0
        node_ids = {node_id for node_id, _node in values}
        for node_id, node in values:
            text = node_text(node)
            overlap_terms = query_terms & set(tokenize(text))
            if overlap_terms:
                covered.update(overlap_terms)
                relevant_count += 1
            best_overlap = max(best_overlap, query_overlap(frame, text))
            best_rrf = max(best_rrf, rrf_scores.get(node_id, 0.0))
            if isinstance(node, ClaimNode) and node.predicate_key != "said":
                compatibility = subject_compatibility(node)
                claim_terms = set(tokenize(
                    f"{node.predicate} {node.predicate_key} {node.object}"
                ))
                predicate_terms = set(tokenize(
                    f"{node.predicate} {node.predicate_key}"
                ))
                best_claim_joint = max(
                    best_claim_joint,
                    compatibility
                    * len(query_terms & claim_terms)
                    / max(1, len(query_terms)),
                )
                best_claim_predicate = max(
                    best_claim_predicate,
                    compatibility
                    * len(query_terms & predicate_terms)
                    / max(1, len(query_terms)),
                )

        channel_hits = sum(bool(node_ids & ids) for ids in channel_top.values())
        coverage = len(covered) / max(1, len(query_terms))
        density = min(1.0, math.log1p(relevant_count) / math.log(8.0))
        posterior = (
            0.38 * best_claim_joint
            + 0.18 * best_claim_predicate
            + 0.14 * coverage
            + 0.10 * best_overlap
            + 0.09 * (channel_hits / max(1, len(channel_top)))
            + 0.06 * (best_rrf / max_rrf)
            + 0.05 * density
        )
        rows.append({
            "session_id": session_id,
            "posterior": round(posterior, 6),
            "query_coverage": round(coverage, 6),
            "covered_terms": sorted(covered),
            "claim_joint_match": round(best_claim_joint, 6),
            "claim_predicate_match": round(best_claim_predicate, 6),
            "channel_hits": channel_hits,
            "relevant_node_count": relevant_count,
        })

    rows.sort(
        key=lambda row: (
            row["posterior"],
            row["claim_joint_match"],
            row["query_coverage"],
            row["session_id"],
        ),
        reverse=True,
    )
    return rows
