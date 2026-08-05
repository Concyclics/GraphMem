from graphmem_demo.v3.navigation_policy import navigation_decision


def _trace(kind: str) -> dict:
    return {"query_frame": {"query_kind": kind}}


def test_relational_queries_use_graph_navigation() -> None:
    for kind in ("lookup", "location", "latest", "ordering", "recommendation"):
        assert navigation_decision(_trace(kind)).use_graph_navigation


def test_closed_form_queries_preserve_deterministic_path() -> None:
    for kind in ("count", "list", "date", "duration", "arithmetic"):
        assert not navigation_decision(_trace(kind)).use_graph_navigation


def test_policy_does_not_read_benchmark_or_topic_fields() -> None:
    trace = _trace("lookup")
    trace.update({"benchmark": "unknown", "topic": "special-case-value"})
    decision = navigation_decision(trace)
    assert decision.use_graph_navigation
    assert "special-case" not in decision.reason
