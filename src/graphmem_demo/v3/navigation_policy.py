from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_RELATIONAL_QUERY_KINDS = frozenset(
    {
        "latest",
        "location",
        "lookup",
        "ordering",
        "recommendation",
    }
)


@dataclass(frozen=True)
class NavigationDecision:
    use_graph_navigation: bool
    query_kind: str
    reason: str


def navigation_decision(retrieval_trace: dict[str, Any]) -> NavigationDecision:
    """Choose graph navigation by query algebra, never benchmark or topic.

    Relational lookup/state/order questions benefit from graph-path recovery.
    Closed-form aggregate and temporal arithmetic questions should retain their
    deterministic operator proof instead of asking an LLM to reselect operands.
    """

    query_frame = retrieval_trace.get("query_frame") or {}
    query_kind = str(
        query_frame.get("query_kind")
        or retrieval_trace.get("query_kind")
        or query_frame.get("requested_operation")
        or ""
    ).casefold()
    if query_kind in _RELATIONAL_QUERY_KINDS:
        return NavigationDecision(
            True,
            query_kind,
            "relational_query_requires_path_validation",
        )
    return NavigationDecision(
        False,
        query_kind,
        "deterministic_or_direct_path_preserved",
    )
