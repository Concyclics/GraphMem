from graphmem_demo.v3.retrieval import _evidence_time, _node_text, _tokens, build_query_frame
from graphmem_demo.v3.schema import TurnNode
from graphmem_demo.v3.weekday_operator import weekday_scope_hint


def _turn(node_id, day, text):
    return TurnNode(
        node_id, "q", "s", day, 0, "A", "a", "B", "user", text, text,
    )


def test_last_weekday_is_anchored_before_question_date() -> None:
    frame = build_query_frame("What material did I start using last Friday?")
    target = _turn("target", "2026-03-06", "I started using cobalt material.")
    distractor = _turn("other", "2026-03-10", "I discussed material storage.")
    hint = weekday_scope_hint(
        frame,
        [("turn", distractor, 2.0, "test"), ("turn", target, 1.0, "test")],
        "2026-03-11",
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["target_date"] == "2026-03-06"
    assert hint["selected_evidence_date"] == "2026-03-06"
    assert hint["within_tolerance"]


def test_same_day_events_bind_requested_companion_slot() -> None:
    frame = build_query_frame("Who did I attend the event with last Friday?")
    no_companion = _turn("plain", "2026-03-06", "I attended a public lecture.")
    companion = _turn("bound", "2026-03-06", "I attended a concert with my neighbors.")
    hint = weekday_scope_hint(
        frame,
        [("turn", no_companion, 2.0, "test"), ("turn", companion, 1.0, "test")],
        "2026-03-11",
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["supporting_node_ids"] == ["bound"]
