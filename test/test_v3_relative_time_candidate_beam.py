from __future__ import annotations

from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.state_temporal_operators import relative_time_hint


def _operand(node_id: str, text: str, score: float) -> tuple:
    node = OperandRecordV3(
        operand_id=node_id,
        question_id="q",
        subject_key="participant 1",
        predicate_key="has",
        object_key=text.casefold(),
        object_text=text,
        event_time="2023-03-15",
        observed_at="2023-03-15",
        source_turn_ids=[f"{node_id}:turn"],
        retrieval_text=text,
    )
    return ("operand", node, score, "date_index")


def test_relative_time_preserves_a_bounded_same_date_candidate_beam() -> None:
    frame = build_query_frame("What kitchen appliance did I buy 10 days ago?")
    hint = relative_time_hint(
        frame,
        [
            _operand("suggestion", "energy efficient appliance suggestions", 0.9),
            _operand("purchase", "new smoker", 0.7),
        ],
        "2023-03-25",
        tokenize=lambda text: text.casefold().split(),
        node_text=lambda node: node.retrieval_text,
        evidence_time=lambda node: node.event_time,
    )
    assert hint is not None
    assert set(hint["candidate_node_ids"]) == {"suggestion", "purchase"}
    assert hint["within_tolerance"] is True
