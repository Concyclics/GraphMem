"""Recovery helpers for resumable full-corpus graph builds.

Graph publication is atomic, but semantic calls are cached as soon as each
request completes.  An interruption can therefore leave LLM call rows for a
Memory that has no published graph.  Replaying those rows inside a fresh token
ledger double-counts cost and can change which late batch is admitted.  Audit
and reset such unpublished attempts while preserving every published graph and
all deterministic embeddings.
"""
from __future__ import annotations

from typing import Any, Mapping


def reset_unpublished_llm_attempts(store: Any) -> tuple[Mapping[str, object], ...]:
    """Remove partial LLM ledgers with no matching graph publication."""

    rows = store._read(
        "SELECT c.memory_id,COUNT(*) calls,"
        "SUM(CASE WHEN c.cached=0 THEN 1 ELSE 0 END) api_calls,"
        "SUM(COALESCE(CAST(json_extract(c.usage_json,'$.uncached_input_tokens') "
        "AS INTEGER),0)) input_tokens,"
        "SUM(COALESCE(CAST(json_extract(c.usage_json,'$.cached_input_tokens') "
        "AS INTEGER),0)) cached_input_tokens,"
        "SUM(COALESCE(CAST(json_extract(c.usage_json,'$.output_tokens') "
        "AS INTEGER),0)) output_tokens,"
        "SUM(c.retry_count) retry_count "
        "FROM llm_calls c LEFT JOIN graph_versions g "
        "ON g.memory_id=c.memory_id WHERE g.memory_id IS NULL "
        "GROUP BY c.memory_id ORDER BY c.memory_id"
    )
    audit = tuple({
        "memory_id": str(row["memory_id"]),
        "calls": int(row["calls"] or 0),
        "api_calls": int(row["api_calls"] or 0),
        "input_tokens": int(row["input_tokens"] or 0),
        "cached_input_tokens": int(row["cached_input_tokens"] or 0),
        "output_tokens": int(row["output_tokens"] or 0),
        "total_api_tokens": int(row["input_tokens"] or 0)
                            + int(row["output_tokens"] or 0),
        "retry_count": int(row["retry_count"] or 0),
        "reason": "unpublished_attempt_reset",
    } for row in rows)
    if not audit:
        return ()

    with store.transaction() as db:
        db.execute(
            "DELETE FROM llm_cache WHERE cache_key IN ("
            "SELECT c.cache_key FROM llm_calls c LEFT JOIN graph_versions g "
            "ON g.memory_id=c.memory_id WHERE g.memory_id IS NULL) "
            "AND cache_key NOT IN ("
            "SELECT c.cache_key FROM llm_calls c JOIN graph_versions g "
            "ON g.memory_id=c.memory_id)"
        )
        db.execute(
            "DELETE FROM llm_calls WHERE memory_id NOT IN "
            "(SELECT memory_id FROM graph_versions)"
        )
    return audit
