from __future__ import annotations

from dataclasses import replace

from graphmem_demo.models import RetrievedContext
from graphmem_demo.v36.schema import QuantityValue
from graphmem_demo.v4 import build_capability_view
from graphmem_demo.v4.capability_retrieval import supplement_capability_gaps
from test_v36_role_graph import _parsed


def test_projection_supplement_is_routed_provenanced_and_budgeted() -> None:
    _case, index, _embedder = _parsed("How many cameras were recommended?")
    index.frames[1] = replace(
        index.frames[1],
        frame_kind="quantity",
        quantity=QuantityValue(value=1, unit="camera"),
    )
    view = build_capability_view(index)
    result = RetrievedContext(
        question_id="q",
        variant="hierarchical_hybrid_graph_v4_0",
        summary_node_ids=[],
        leaf_node_ids=[],
        edge_count=0,
        context_text="seed",
        answer_session_hit=False,
        retrieved_session_ids=["s"],
        latency_sec=0.0,
        packed_rough_tokens=1,
    )
    rows = supplement_capability_gaps(
        result=result,
        index=index,
        capability_view=view,
        requested=["quantity"],
        query_vectors=[index.frames[1].embedding or []],
        question="How many cameras were recommended?",
        token_budget=500,
    )
    assert len(rows) == 1
    assert rows[0]["provenance_complete"] is True
    assert index.frames[1].frame_id in result.fact_node_ids
    assert index.frames[1].source_turn_ids[0] in result.evidence_leaf_ids
    assert result.packed_rough_tokens <= 500


def test_projection_supplement_never_widens_selected_sessions() -> None:
    _case, index, _embedder = _parsed()
    view = build_capability_view(index)
    result = RetrievedContext(
        question_id="q",
        variant="hierarchical_hybrid_graph_v4_0",
        summary_node_ids=[],
        leaf_node_ids=[],
        edge_count=0,
        context_text="",
        answer_session_hit=False,
        retrieved_session_ids=["different-session"],
        latency_sec=0.0,
    )
    rows = supplement_capability_gaps(
        result=result,
        index=index,
        capability_view=view,
        requested=["dialogue_answer"],
        query_vectors=[index.frames[1].embedding or []],
        question="What was the answer?",
        token_budget=500,
    )
    assert rows == []
    assert result.context_text == ""
