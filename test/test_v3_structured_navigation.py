from graphmem_demo.v3.llm_navigation import deterministic_navigation_plan
from graphmem_demo.v3.structured_navigation import (
    authoritative_trace_answer,
    build_query_ir,
    canonical_operation,
    certified_trace_hint,
    merged_relations,
    multiview_frontier_ids,
)


def test_count_query_has_collection_contract_without_topic_rules() -> None:
    ir = build_query_ir("How many different places did Alex visit?")
    assert ir.intent == "count"
    assert ir.set_wide is True
    assert "collection_closure" in ir.required_slots
    assert "same_entity" in ir.allowed_relations
    assert "quantity_collection" in ir.allowed_relations


def test_temporal_query_requests_event_and_anchor_relations() -> None:
    ir = build_query_ir("When did Morgan attend the meeting?")
    assert ir.intent == "date"
    assert ir.temporal is True
    assert "event_identity" in ir.required_slots
    assert "temporal_anchor" in ir.required_slots
    assert "same_event" in ir.allowed_relations
    assert "temporal_scope" in ir.allowed_relations


def test_free_form_operation_cannot_replace_fixed_intent() -> None:
    ir = build_query_ir("What do Casey's children like?")
    operation = canonical_operation(ir, "pottery_topic_answer")
    assert operation.startswith(f"{ir.intent}:")
    assert operation.split(":", 1)[0] == "preference_list"


def test_proposed_relations_are_filtered_to_schema_allowlist() -> None:
    ir = build_query_ir("What is Robin's latest address?")
    relations = merged_relations(ir, ["state_history", "invented_topic_edge"])
    assert "state_history" in relations
    assert "invented_topic_edge" not in relations


def test_multiview_frontier_is_bounded_and_deduplicated() -> None:
    ir = build_query_ir("What did Taylor buy?")
    trace = {
        "channels": {
            "exact": ["q:session_1:turn:1", "q:session_1:turn:2"],
            "bm25": ["q:session_1:turn:1", "q:session_2:turn:1"],
            "dense": ["q:session_3:turn:1"],
        },
        "rrf_top": [
            {"node_id": "q:session_2:turn:1", "score": 0.2},
            {"node_id": "q:session_4:turn:1", "score": 0.1},
        ],
    }
    rows = multiview_frontier_ids(trace, ir)
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids))
    assert ids == [
        "q:session_1:turn:1",
        "q:session_1:turn:2",
        "q:session_2:turn:1",
        "q:session_3:turn:1",
        "q:session_4:turn:1",
    ]


def test_authoritative_absence_requires_complete_global_lossless_scan() -> None:
    trace = {
        "catalog_operator_hint": {
            "operation": "exact_entity_mismatch",
            "required_entity": "vintage films",
            "complete": True,
            "global_lossless_scan_complete": True,
            "contrast_proof_complete": True,
        }
    }
    answer = authoritative_trace_answer(trace)
    assert answer is not None
    assert "vintage films" in answer


def test_deterministic_navigation_uses_query_contract_without_answer_proposal() -> None:
    question = "How many different places did Alex visit?"
    ir = build_query_ir(question)
    ledger = [
        {
            "node_id": "q:session_1:turn:1",
            "node_type": "turn",
            "selection_source": "multiview_bm25",
            "score": 1.0,
            "text": "Alex visited Lisbon.",
            "session_id": "session_1",
        },
        {
            "node_id": "q:session_2:turn:1",
            "node_type": "turn",
            "selection_source": "multiview_dense",
            "score": 0.8,
            "text": "Alex also visited Porto.",
            "session_id": "session_2",
        },
    ]
    plan = deterministic_navigation_plan(
        question=question,
        evidence_ledger=ledger,
        query_ir=ir,
    )
    assert set(plan.selected_ids) == {
        "q:session_1:turn:1", "q:session_2:turn:1"
    }
    assert plan.operation == "count"
    assert plan.candidate_answer == ""
    assert plan.needed_relations == ir.allowed_relations


def test_ratio_query_precedes_incidental_current_state_word() -> None:
    ir = build_query_ir(
        "What percentage of the property price is work on my current house?"
    )
    assert ir.intent == "count"
    assert ir.answer_form == "number"


def test_final_choice_and_previous_now_queries_open_state_relations() -> None:
    final_ir = build_query_ir("What did we finally decide to name it?")
    assert final_ir.intent == "latest"
    assert final_ir.state_sensitive is True
    recurrence_ir = build_query_ir(
        "How often did I go previously, and how often do I go now?"
    )
    assert recurrence_ir.intent == "recurrence"
    assert recurrence_ir.state_sensitive is True
    assert "supersedes" in recurrence_ir.allowed_relations


def test_count_hint_rejects_unproven_state_cardinality_slot() -> None:
    ir = build_query_ir("How many more units must I acquire to reach the target?")
    trace = {
        "closure_certificate": {"provenance_complete": True},
        "catalog_operator_hint": {
            "operation": "latest_cardinality_state",
            "value": 300,
            "complete": True,
            "packed_provenance_complete": True,
        },
    }
    assert certified_trace_hint(trace, ir) is None


def test_count_hint_keeps_closed_dimensional_total_proposal() -> None:
    ir = build_query_ir("What is the total mass of the two deliveries?")
    trace = {
        "closure_certificate": {"provenance_complete": True},
        "catalog_operator_hint": {
            "operation": "dimensional_quantity_total",
            "value": 70,
            "unit": "kg",
            "complete": True,
            "packed_provenance_complete": True,
        },
    }
    assert certified_trace_hint(trace, ir) == {
        "catalog_operator_proposal": trace["catalog_operator_hint"]
    }


def test_complete_graph_operator_suppresses_conflicting_local_catalog() -> None:
    ir = build_query_ir("Which event happened first, launch or arrival?")
    trace = {
        "closure_certificate": {
            "complete": True,
            "provenance_complete": True,
            "missing_requirements": [],
            "operand_node_ids": ["q:operand:1", "q:operand:2"],
        },
        "operator_result": {"operation": "earliest", "value": "arrival"},
        "catalog_operator_hint": {
            "operation": "earliest_named_alternative",
            "value": "launch",
            "complete": True,
            "packed_provenance_complete": True,
        },
    }
    hint = certified_trace_hint(trace, ir)
    assert hint is not None
    assert hint["operator_result"]["value"] == "arrival"
    assert "catalog_operator_proposal" not in hint


def test_complete_operator_from_wrong_query_family_is_not_exposed() -> None:
    ir = build_query_ir("What is the total cost of the two repairs?")
    trace = {
        "closure_certificate": {
            "complete": True,
            "provenance_complete": True,
            "missing_requirements": [],
        },
        "operator_result": {"operation": "latest", "value": "unrelated state"},
        "catalog_operator_hint": {
            "operation": "dimensional_quantity_total",
            "value": 200,
            "complete": True,
            "packed_provenance_complete": True,
        },
    }
    assert certified_trace_hint(trace, ir) == {
        "catalog_operator_proposal": trace["catalog_operator_hint"]
    }
