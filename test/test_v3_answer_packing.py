from __future__ import annotations

import json

from graphmem_demo.models import QuestionCase, RetrievedContext
from graphmem_demo.v3.answer_packing import (
    conservative_prompt_tokens, contract_evidence_to_ids,
    fit_answer_payload,
)
from graphmem_demo.v3.retrieval import answer_messages, build_query_frame


def test_complete_answer_request_is_fitted_not_only_evidence() -> None:
    system = "Use evidence only. " * 120
    blocks = [
        f"[TURN q:session:{index}:turn:0]\n" + ("evidence " * 220)
        for index in range(12)
    ]
    payload = {
        "question": "Which events occurred?",
        "scope_posteriors": [{"session_id": str(index), "score": 0.5} for index in range(8)],
        "temporal_evidence_ledger_newest_first": [{"text": "x" * 500} for _ in range(8)],
        "recommendation_constraints": [{"text": "y" * 500} for _ in range(8)],
        "evidence": "\n\n".join(blocks),
    }
    messages, trace = fit_answer_payload(system, payload, max_prompt_tokens=1800)
    assert trace["fit_pass"] is True
    assert trace["final_evidence_blocks"] < trace["initial_evidence_blocks"]
    assert conservative_prompt_tokens(
        messages[0]["content"], messages[1]["content"]
    ) <= 1800


def test_complete_local_closure_contracts_to_node_and_provenance_blocks() -> None:
    evidence = "\n\n".join([
        "[EVENT_FRAME frame:anchor | sources=turn:anchor]\nAnchor event",
        "[TURN turn:anchor | session=s1]\nAnchor source",
        "[EVENT_FRAME frame:target | sources=turn:target]\nTarget event",
        "[TURN turn:target | session=s0]\nTarget source",
        "[TURN turn:noise | session=s2]\nDistractor",
    ])
    contracted = contract_evidence_to_ids(
        evidence, {"frame:anchor", "turn:anchor", "frame:target", "turn:target"}
    )
    assert "Anchor event" in contracted and "Target event" in contracted
    assert "Anchor source" in contracted and "Target source" in contracted
    assert "Distractor" not in contracted


def test_empty_or_incomplete_closure_does_not_contract() -> None:
    assert contract_evidence_to_ids("[TURN t]\nvalue", set()) == ""


def test_lookup_keeps_full_evidence_and_exposes_focused_capsule() -> None:
    case = QuestionCase(
        question_id="q", question_type="fact",
        question="What item did Mira recommend to Lee?", answer="Northstar",
        question_date=None, haystack_sessions=[], haystack_session_ids=[],
        haystack_dates=[], answer_session_ids=[],
    )
    relevant = "[TURN q:s1:turn:0 | speaker=Mira]\nI recommended Northstar to Lee."
    distractors = [
        f"[TURN q:s2:turn:{index} | speaker=Omar]\nUnrelated archive note {index}."
        for index in range(12)
    ]
    retrieval = RetrievedContext(
        question_id="q", variant="hierarchical_hypergraph_v3",
        summary_node_ids=[], leaf_node_ids=[], edge_count=0,
        context_text="\n\n".join([relevant, *distractors]),
        answer_session_hit=True, retrieved_session_ids=["s1"], latency_sec=0.0,
        retrieval_trace={"query_frame": build_query_frame(case.question).__dict__},
    )
    messages = answer_messages(case, retrieval)
    payload = json.loads(messages[1]["content"])
    assert "Northstar" in payload["evidence"]
    assert "Unrelated archive note 11" in payload["evidence"]
    assert "Northstar" in payload["query_focused_evidence"]
    assert "Unrelated archive note 11" not in payload["query_focused_evidence"]
    assert retrieval.retrieval_trace["answer_focused_only"] is False
    assert retrieval.retrieval_trace["answer_evidence_block_ids"][0] == (
        "q:s1:turn:0"
    )
    assert len(retrieval.retrieval_trace["answer_evidence_block_ids"]) == 13
    assert relevant in retrieval.retrieval_trace["answer_evidence_text"]



def test_nested_operator_prose_and_graph_audit_ids_have_independent_bounds() -> None:
    payload = {
        "question": "Which event came first?",
        "before_after_relation_hint": {
            "complete": True,
            "anchor_event": {
                "label": "Harbor trip",
                "source_turn_ids": [f"turn:{index}" for index in range(80)],
                "evidence": "long assistant projection " * 900,
            },
            "nearest_qualifying_event": {
                "label": "Cedar trip",
                "evidence": "another long projection " * 900,
            },
        },
        "closure_certificate": {
            "complete": True,
            "visited_hyperedge_ids": [f"edge:{index}" for index in range(100)],
        },
        "query_focused_evidence": "focused " * 2000,
        "evidence": "[TURN turn:0]\nAlex visited Cedar before Harbor.",
    }
    messages, trace = fit_answer_payload(
        "Use cited evidence only.", payload, max_prompt_tokens=1800
    )
    fitted = json.loads(messages[1]["content"])
    assert trace["fit_pass"] is True
    assert "before_after_relation_hint" in trace["compacted_structured_fields"]
    assert len(fitted["before_after_relation_hint"]["anchor_event"]["evidence"]) < 800
    assert len(fitted["closure_certificate"]["visited_hyperedge_ids"]) == 16
    assert conservative_prompt_tokens(
        messages[0]["content"], messages[1]["content"]
    ) <= 1800
