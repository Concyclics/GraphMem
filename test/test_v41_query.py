from __future__ import annotations

import inspect
import sqlite3
from dataclasses import asdict, replace

from graphmem_demo.v4 import build_capability_view, build_query_ir
from graphmem_demo.v41 import (
    GRAPHMEM_V41_SCHEMA,
    QueryPolicyV41,
    answer_messages,
    build_query_plan,
    build_sidecar,
    parse_planner_result,
    persist_sidecar,
    query_views,
    retrieve,
    sidecar_matches, trim_latest_addition,
)
from test_v36_role_graph import _parsed


def test_v41_sidecar_is_read_only_versioned_and_persisted(tmp_path) -> None:
    _case, index, _embedder = _parsed()
    before = asdict(index)
    sidecar = build_sidecar(index)
    assert asdict(index) == before
    assert sidecar.documents
    assert sidecar.index_hash
    assert sidecar.adjacency

    path = tmp_path / "retrieval_v41.sqlite"
    persist_sidecar(path, sidecar)
    assert sidecar_matches(path, index)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT count(*) FROM documents_fts"
        ).fetchone()[0] == len(sidecar.documents)
    finally:
        connection.close()


def test_v41_domain_query_plan_is_generic_and_role_based() -> None:
    ir = build_query_ir(
        "Which medical device does my sibling currently prefer?"
    )
    plan = build_query_plan(ir)
    assert "health_device" in plan.domain_hints
    assert "profile_relationship" in plan.domain_hints
    assert plan.answer_algebra in {"state_update", "preference_recommendation"}
    assert {"owner", "source"}.issubset(plan.required_roles)
    assert query_views(ir, plan)


def test_v41_temporal_deictic_reply_keeps_preceding_event_identity() -> None:
    from graphmem_demo.v41.retrieval import _scene_window_nodes
    from graphmem_demo.v41.schema import (
        QuerySidecarV41, SidecarDocumentV41,
    )

    ir = build_query_ir("When did Dana take a walk after the trip?")
    assert ir.requested_value_type == "date"
    assert build_query_plan(ir).answer_algebra == "temporal_lookup"
    documents = {
        "q:s:turn:2": SidecarDocumentV41(
            node_id="q:s:turn:2", node_type="turn",
            session_ids=["s"], source_turn_ids=["q:s:turn:2"],
            text="speaker Dana | [Media shared; caption: walking on a trail]",
        ),
        "q:s:turn:3": SidecarDocumentV41(
            node_id="q:s:turn:3", node_type="turn",
            session_ids=["s"], source_turn_ids=["q:s:turn:3"],
            text="speaker Lee | Is that recent?",
        ),
        "q:s:turn:4": SidecarDocumentV41(
            node_id="q:s:turn:4", node_type="turn",
            session_ids=["s"], source_turn_ids=["q:s:turn:4"],
            text="speaker Dana | We did it yesterday after the trip.",
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", documents=documents,
        inverted={}, adjacency={},
    )
    selected = _scene_window_nodes(
        ["q:s:turn:4"], ir, sidecar, limit=4,
    )
    assert selected[:2] == ["q:s:turn:2", "q:s:turn:3"]


def test_v41_reference_identity_is_generic_and_planner_bound() -> None:
    from graphmem_demo.v41 import planner_messages

    question = (
        "What is the device with rotating blades where air is moved "
        "across a room?"
    )
    ir = build_query_ir(question)
    plan = build_query_plan(ir)
    assert ir.requested_value_type == "span"
    assert {"reference", "identity", "source"}.issubset(ir.required_roles)
    assert plan.answer_algebra == "reference_identity"
    assert plan.planner_required is True
    prompt = planner_messages(
        type("Case", (), {"question": question, "question_date": "2023-01-01"})(),
        ir, plan, {"present_roles": [], "missing_roles": ["identity"]}, [],
    )[0]["content"]
    assert "every distinctive descriptive clue" in prompt
    assert "lookup table" in prompt
    from graphmem_demo.clients import rough_token_count
    messages = planner_messages(
        type("Case", (), {"question": question, "question_date": "2023-01-01"})(),
        ir, plan, {"present_roles": [], "missing_roles": ["identity"]}, [],
    )
    assert rough_token_count("\n".join(row["content"] for row in messages)) <= 700


def test_v41_reference_identity_planner_keeps_same_session_descriptions() -> None:
    from graphmem_demo.v41.retrieval import _planner_evidence_candidates
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    distractor = "q:devices:turn:0"
    descriptive = "q:devices:turn:2"
    documents = {
        distractor: SidecarDocumentV41(
            node_id=distractor, node_type="turn", session_ids=["devices"],
            source_turn_ids=[distractor],
            text="We also used the named kitchen device Turbo Mixer.",
        ),
        descriptive: SidecarDocumentV41(
            node_id=descriptive, node_type="turn", session_ids=["devices"],
            source_turn_ids=[descriptive],
            text=("It has rotating blades that move air across a room and cool people."),
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", documents=documents,
        inverted={}, adjacency={},
    )
    ir = build_query_ir(
        "What is the device with rotating blades where air is moved "
        "across a room?"
    )
    rows = _planner_evidence_candidates(
        [distractor, descriptive], ir, sidecar, limit=2,
        preferred_source_ids=[], session_diverse=False,
    )
    assert rows[0]["source_turn_id"] == descriptive
    assert {row["source_turn_id"] for row in rows} == {distractor, descriptive}


def test_v41_retrieval_only_appends_sources_and_respects_budget() -> None:
    case, index, embedder = _parsed("Which camera did Bob recommend?")
    view = build_capability_view(index)
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    vectors = embedder.embed(
        query_views(ir, plan), question_id=case.question_id,
        variant="hierarchical_hybrid_graph_v4_1_query",
    )
    sidecar = build_sidecar(index)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=view, sidecar=sidecar,
        query_ir=ir, query_vectors=vectors, token_budget=9200,
    )
    trace = result.retrieval_trace
    assert result.schema_version == GRAPHMEM_V41_SCHEMA
    assert trace["v41_source_deletions"] == []
    assert trace["v41_frame_deletions"] == []
    assert result.packed_rough_tokens <= 9200
    assert trace["v41_evidence_certificate"]["source_turn_ids"]
    assert "v41_candidate_trace" in trace
    assert "v41_typed_expansion" in trace
    assert trace["v41_optional_stage_order"][:2] == [
        "typed_expansion", "answer_bearing",
    ]
    assert trace["planner_required"] is (
        not trace["v41_evidence_certificate"]["complete"]
    )



def test_v41_semantic_turn_rank_uses_domain_scene_bridges() -> None:
    from graphmem_demo.v41.retrieval import _semantic_turn_rank
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    question = "What should I serve with my homegrown ingredients?"
    ir = build_query_ir(question)
    plan = build_query_plan(ir)
    good = "q:garden:turn:0"
    noise = "q:party:turn:0"
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test",
        documents={
            good: SidecarDocumentV41(
                node_id=good, node_type="turn", session_ids=["garden"],
                source_turn_ids=[good],
                text="I harvested cherry tomatoes and herbs from my garden.",
            ),
            noise: SidecarDocumentV41(
                node_id=noise, node_type="turn", session_ids=["party"],
                source_turn_ids=[noise],
                text="I served drinks at a party this weekend.",
            ),
        },
        inverted={}, adjacency={},
    )
    ranked = _semantic_turn_rank(ir, plan, sidecar)
    assert ranked[0][0] == good


def test_v41_semantic_overlap_ignores_conversational_scaffolding() -> None:
    from graphmem_demo.v41.retrieval import _semantic_overlap, _tokens

    ir = build_query_ir(
        "I've been having trouble with my phone battery lately. Any tips?"
    )
    noise = _tokens(
        "I've been having trouble deciding lately. Can you give me any tips?"
    )
    relevant = _tokens(
        "My portable power bank can charge the phone when its battery is low."
    )
    assert not _semantic_overlap(ir, noise)
    assert {"phone", "battery"}.issubset(_semantic_overlap(ir, relevant))


def test_v41_dialogue_pair_completion_precedes_optional_anchors() -> None:
    from graphmem_demo.v41.retrieval import _dialogue_pair_completion_nodes
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    prefix = "memory:session_9:turn:"
    documents = {
        prefix + "11": SidecarDocumentV41(
            node_id=prefix + "11", node_type="turn", session_ids=["session_9"],
            source_turn_ids=[prefix + "11"], text="Here is an earlier remark.",
        ),
        prefix + "12": SidecarDocumentV41(
            node_id=prefix + "12", node_type="turn", session_ids=["session_9"],
            source_turn_ids=[prefix + "12"], text="What books do you enjoy?",
        ),
        prefix + "13": SidecarDocumentV41(
            node_id=prefix + "13", node_type="turn", session_ids=["session_9"],
            source_turn_ids=[prefix + "13"],
            text="I enjoy stories with adventures and magic.",
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", documents=documents,
        inverted={}, adjacency={
            prefix + "12": {"next_turn": [prefix + "11", prefix + "13"]},
        },
    )
    ir = build_query_ir("What kind of books does Nate enjoy?")
    assert _dialogue_pair_completion_nodes(
        [prefix + "12"], ir, sidecar,
    ) == [prefix + "13"]

def test_v41_planner_parser_fails_closed() -> None:
    invalid = parse_planner_result("not-json")
    assert invalid.valid is False
    valid = parse_planner_result(
        '{"alternative_entities":["relative"],"event_aliases":["arrival"],'
        '"relations":["visited"],"temporal_constraints":["last week"],'
        '"missing_roles":["time"]}'
    )
    assert valid.valid is True
    assert valid.event_aliases == ["arrival"]


def test_v41_collection_head_boundary_excludes_named_endpoint() -> None:
    from graphmem_demo.v41.retrieval import (
        _collection_head_and_boundary,
        _verified_planner_collection_members,
    )
    from graphmem_demo.v41.schema import PlannerResultV41

    question = (
        "How many properties did I view before making an offer on the "
        "townhouse in the Brookside neighborhood?"
    )
    assert _collection_head_and_boundary(question) == (
        "properties",
        "making an offer on the townhouse in the Brookside neighborhood",
    )
    ir = build_query_ir(question)
    planner = PlannerResultV41(
        member_candidates=[{
            "value": "3-bedroom townhouse in the Brookside neighborhood",
            "source_turn_id": "q:boundary:turn:0",
        }],
        valid=True,
    )
    verified = _verified_planner_collection_members(
        planner,
        [{"source_turn_id": "q:boundary:turn:0",
          "text": "I offered on the 3-bedroom townhouse in the Brookside neighborhood."},
         {"source_turn_id": "q:prior:turn:0",
          "text": "I viewed a 2-bedroom condo before that offer."}],
        ir,
    )
    assert [row["value"] for row in verified] == ["2-bedroom condo"]
    media_ir = build_query_ir(
        "How many albums or EPs have I purchased or downloaded?"
    )
    media = _verified_planner_collection_members(
        PlannerResultV41(valid=True),
        [{"source_turn_id": "q:music:turn:0",
          "text": "I got my Tame Impala vinyl signed after the show."}],
        media_ir,
    )
    assert [row["value"] for row in media] == ["Tame Impala vinyl"]
    care_ir = build_query_ir(
        "How many different doctors did I visit?"
    )
    care = _verified_planner_collection_members(
        PlannerResultV41(member_candidates=[{
            "value": "my doctor", "source_turn_id": "q:care:turn:0",
        }], valid=True),
        [{"source_turn_id": "q:care:turn:0",
          "text": "Dr. Patel diagnosed me after my visit."},
         {"source_turn_id": "q:care:turn:1",
          "text": "Dr. Smith prescribed antibiotics."},
         {"source_turn_id": "q:care:turn:2",
          "text": "I had a follow-up with Dr. Patel and Dr. Lee."}],
        care_ir,
    )
    assert [row["value"] for row in care] == [
        "Dr. Patel", "Dr. Smith", "Dr. Lee"
    ]
    estate_ir = build_query_ir(
        "How many properties did I view?"
    )
    estate = _verified_planner_collection_members(
        PlannerResultV41(member_candidates=[{
            "value": "3-bedroom bungalow in the Oakwood neighborhood",
            "source_turn_id": "q:estate:turn:0",
        }], valid=True),
        [{"source_turn_id": "q:estate:turn:0",
          "text": "I viewed a 3-bedroom bungalow in the Oakwood neighborhood."}],
        estate_ir,
    )
    assert [row["value"] for row in estate] == [
        "3-bedroom bungalow in the Oakwood neighborhood"
    ]



def test_v41_current_owned_instruments_are_source_bound_and_distinct() -> None:
    from graphmem_demo.v41.retrieval import (
        _planner_collection_exact_binding_safe,
        _verified_planner_collection_members,
    )
    from graphmem_demo.v41.schema import PlannerResultV41

    question = "How many musical instruments do I currently own?"
    evidence = [
        {"source_turn_id": "q:s1:turn:0",
         "text": "I've had my black Fender Stratocaster electric guitar for five years.",
         "selection_features": ["possession_bound"]},
        {"source_turn_id": "q:s2:turn:0",
         "text": "I've had my acoustic guitar, a Yamaha FG800, for eight years.",
         "selection_features": ["possession_bound"]},
        {"source_turn_id": "q:s3:turn:0",
         "text": "I'm selling my old drum set, a 5-piece Pearl Export.",
         "selection_features": ["possession_bound"]},
        {"source_turn_id": "q:s3:turn:2",
         "text": "I'm maintaining my piano, a Korg B1, which I've had for three years.",
         "selection_features": ["possession_bound"]},
        {"source_turn_id": "q:s4:turn:0",
         "text": "My niece just got her new violin.",
         "selection_features": []},
        {"source_turn_id": "q:s5:turn:0",
         "text": "I'm thinking of buying a new ukulele.",
         "selection_features": []},
    ]
    members = _verified_planner_collection_members(
        PlannerResultV41(valid=True), evidence, build_query_ir(question),
    )
    assert [row["value"] for row in members] == [
        "black Fender Stratocaster electric guitar",
        "Yamaha FG800 acoustic guitar",
        "5-piece Pearl Export drum set",
        "Korg B1 piano",
    ]
    assert _planner_collection_exact_binding_safe(
        question, members, evidence,
    )


def test_v41_planner_collection_members_are_source_validated() -> None:
    from graphmem_demo.v41.retrieval import (
        _verified_planner_collection_members,
    )
    from graphmem_demo.v41.schema import PlannerResultV41

    source_id = "q:models:turn:0"
    parsed = parse_planner_result(
        '{"member_candidates":['
        '["B-29 bomber","q:models:turn:0"],'
        '["meal kit","fabricated:turn:9"]],'
        '"selected_source_ids":["q:models:turn:0"]}'
    )
    assert parsed.member_candidates[0]["value"] == "B-29 bomber"
    ir = build_query_ir("How many model kits have I worked on?")
    verified = _verified_planner_collection_members(
        parsed,
        [{"source_turn_id": source_id,
          "text": "I worked on a B-29 bomber model kit."}],
        ir,
    )
    assert verified == [{
        "value": "B-29 bomber",
        "source_turn_id": source_id,
        "provenance_complete": True,
    }]
    generic = PlannerResultV41(
        member_candidates=[{
            "value": "model kits", "source_turn_id": source_id,
        }],
        valid=True,
    )
    assert _verified_planner_collection_members(
        generic,
        [{"source_turn_id": source_id,
          "text": "I worked on several model kits."}],
        ir,
    ) == []
    dinner = PlannerResultV41(
        member_candidates=[{
            "value": "dinner party at Sarah's place",
            "source_turn_id": "q:dinner:turn:0",
        }, {
            "value": "lovely feast at Sarah's place",
            "source_turn_id": "q:dinner:turn:1",
        }],
        valid=True,
    )
    dinner_ir = build_query_ir(
        "How many dinner parties did I attend?"
    )
    assert _verified_planner_collection_members(
        dinner,
        [{"source_turn_id": "q:dinner:turn:0",
          "text": "Speaking of dinner parties, I had a feast at Sarah's place."},
         {"source_turn_id": "q:dinner:turn:1",
          "text": "It was a lovely feast at Sarah's place."}],
        dinner_ir,
    ) == [{
        "value": "dinner party at Sarah's place",
        "source_turn_id": "q:dinner:turn:0",
        "provenance_complete": True,
    }]



def test_v41_answer_constraint_keeps_llm_as_final_reader() -> None:
    case, index, embedder = _parsed("Which camera did Bob recommend?")
    view = build_capability_view(index)
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=view, sidecar=build_sidecar(index),
        query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ),
        token_budget=9200,
    )
    result.retrieval_trace["v41_evidence_certificate"].update({
        "entity_match": True, "relation_match": True,
        "scope_match": True, "provenance_complete": True,
    })
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "latest_valid_state", "value": "camera-a",
        "certified": True,
    }]
    prompt = "\n".join(
        message["content"] for message in answer_messages(case, result)
    )
    assert "ANSWER_CONSTRAINT" in prompt
    assert "camera-a" in prompt
    assert "final response must be produced by this model" in prompt


def test_v41_source_has_no_benchmark_ids_or_instance_rules() -> None:
    import graphmem_demo.v41.domains as domains_module
    import graphmem_demo.v41.retrieval as retrieval_module
    import graphmem_demo.v41.sidecar as sidecar_module

    source = "\n".join(
        inspect.getsource(module)
        for module in (domains_module, retrieval_module, sidecar_module)
    ).casefold()
    banned = (
        "longmemeval", "locomo", "answer_session_ids", "gold_answer",
        "question_type", "ibotta", "guitar amp", "egg_sales",
    )
    assert all(value not in source for value in banned)


def test_v41_budget_trim_never_deletes_baseline_evidence() -> None:
    case, index, embedder = _parsed("Which camera did Bob recommend?")
    view = build_capability_view(index)
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=view, sidecar=build_sidecar(index),
        query_ir=ir, query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    baseline = set(result.retrieval_trace["v41_original_source_ids"])
    if result.retrieval_trace["v41_source_additions"]:
        assert trim_latest_addition(result) is not None
    assert baseline.issubset(result.retrieval_trace["packed_source_turn_ids"])
    assert result.retrieval_trace["v41_source_deletions"] == []


def test_v41_policy_defaults_match_budget_contract() -> None:
    policy = QueryPolicyV41()
    assert policy.normal_context_target == 8400
    assert policy.complex_context_target == 9200
    assert policy.query_target == 10_000
    assert policy.query_hard_limit == 13_000
    assert policy.planner_output_max == 256


def test_v41_scalar_type_lookup_is_not_misclassified_as_collection() -> None:
    ir = build_query_ir(
        "What type of meal does Alex often cook using a slow cooker?"
    )
    plan = build_query_plan(ir)
    assert plan.answer_algebra == "direct_fact"
    assert plan.planner_required is False


def test_v41_scalar_threshold_lookup_is_not_misclassified_as_collection() -> None:
    ir = build_query_ir(
        "How many points do I need to reach the gold level?"
    )
    plan = build_query_plan(ir)
    assert plan.answer_algebra == "direct_fact"
    assert plan.planner_required is False


def test_v41_current_metric_is_state_not_member_collection() -> None:
    ir = build_query_ir("How many followers do I currently have?")
    plan = build_query_plan(ir)
    assert plan.answer_algebra == "state_update"


def test_v41_single_event_time_is_lookup_without_planner() -> None:
    ir = build_query_ir("When did Alex attend the workshop?")
    plan = build_query_plan(ir)
    assert plan.answer_algebra == "temporal_lookup"
    assert {"event", "time", "source"}.issubset(plan.required_roles)
    assert "event_b" not in plan.required_roles
    assert plan.planner_required is False


def test_v41_two_endpoint_time_is_comparison() -> None:
    ir = build_query_ir(
        "How many days elapsed between the workshop and the concert?"
    )
    plan = build_query_plan(ir)
    assert plan.answer_algebra == "temporal_comparison"
    assert {"event_a", "event_b", "time_a", "time_b"}.issubset(
        plan.required_roles
    )


def test_v41_single_event_duration_uses_bounded_planner() -> None:
    ir = build_query_ir("How long have I been using the fitness tracker?")
    plan = build_query_plan(ir)
    assert plan.answer_algebra == "temporal_lookup"
    assert plan.planner_required is True


def test_v41_direct_dialogue_highlight_binds_three_turn_scene() -> None:
    from graphmem_demo.v41.retrieval import _direct_dialogue_highlights
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    prefix = "memory:session_3:turn:"
    texts = {
        "3": "I designed the store space to feel cozy.",
        "4": "What made you choose the furniture and decor for the design?",
        "5": "I chose comfortable furniture and matching decor.",
    }
    documents = {
        prefix + ordinal: SidecarDocumentV41(
            node_id=prefix + ordinal, node_type="turn",
            session_ids=["session_3"], source_turn_ids=[prefix + ordinal],
            text=text,
        )
        for ordinal, text in texts.items()
    }
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", documents=documents,
        inverted={}, adjacency={},
    )
    ir = build_query_ir("What did Gina design for her store?")
    rows = _direct_dialogue_highlights(list(documents), ir, sidecar)
    assert rows[0]["context_source_id"] == prefix + "3"
    assert rows[0]["prompt_source_id"] == prefix + "4"
    assert rows[0]["reply_source_id"] == prefix + "5"
    assert {"design", "store"}.issubset(rows[0]["matched_terms"])


def test_v41_inference_planner_candidate_is_source_validated() -> None:
    from dataclasses import replace

    from graphmem_demo.clients import rough_token_count
    from graphmem_demo.v41.retrieval import (
        _verified_inference_candidates, planner_messages,
    )
    from graphmem_demo.v41.schema import (
        PlannerResultV41, QuerySidecarV41, SidecarDocumentV41,
    )

    case, _index, _embedder = _parsed()
    case = replace(
        case,
        question="Which outdoor company likely signed Alex for a deal?",
    )
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    source_id = "memory:session_1:turn:7"
    evidence = [{
        "source_turn_id": source_id,
        "text": "I have always liked North Ridge; working with them would be cool.",
    }]
    messages = planner_messages(case, ir, plan, {}, evidence)
    payload = messages[-1]["content"]
    assert "source_named_candidates" in payload
    assert "North Ridge" in payload
    assert rough_token_count("\n".join(row["content"] for row in messages)) <= 700

    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test",
        documents={source_id: SidecarDocumentV41(
            node_id=source_id, node_type="turn", session_ids=["session_1"],
            source_turn_ids=[source_id], text=evidence[0]["text"],
        )}, inverted={}, adjacency={},
    )
    verified = _verified_inference_candidates(
        PlannerResultV41(alternative_entities=["North Ridge"], valid=True),
        [source_id], ir, sidecar,
    )
    assert verified == [{
        "candidate": "North Ridge",
        "source_turn_ids": [source_id],
        "provenance_complete": True,
    }]


def test_v41_followup_endorsement_is_generic_and_source_bound() -> None:
    from graphmem_demo.v41.retrieval import _followup_endorsements

    assert _followup_endorsements(
        "The North Stars are solid and their captain is impressive."
    ) == ["North Stars"]
    assert _followup_endorsements(
        "I heard somebody mention a team, but gave no opinion."
    ) == []


def test_v41_focused_source_evidence_is_provenance_bound_and_deduplicated() -> None:
    from graphmem_demo.v41.retrieval import _focused_source_evidence

    candidate = {
        "source_turn_id": "memory:s1:turn:2",
        "text": "I had 15 signed balls during the first three months.",
        "event_time_text": "2023-04-12",
        "speaker_key": "participant 1",
        "lifecycle_status": "completed",
        "polarity": "positive",
        "target_terms": ["signed ball"],
        "relation_terms": ["first three months"],
        "provenance_complete": True,
    }
    trace = {"source_span_closure": {"candidates": [
        candidate, dict(candidate), {
            **candidate,
            "source_turn_id": "memory:s1:turn:3",
            "text": "A summary without valid provenance.",
            "provenance_complete": False,
        },
    ]}}
    rows = _focused_source_evidence(trace, algebra="collection")
    assert len(rows) == 1
    assert rows[0]["source_turn_id"] == "memory:s1:turn:2"
    assert rows[0]["event_time"] == "2023-04-12"
    assert _focused_source_evidence(trace, algebra="direct_fact") == []


def test_v41_lexical_candidates_ignore_count_words_and_expand_relation_terms() -> None:
    from graphmem_demo.v41.retrieval import _candidate_nodes
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    math_id = "memory:math:turn:1"
    family_id = "memory:family:turn:1"
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", inverted={}, adjacency={},
        documents={
            math_id: SidecarDocumentV41(
                node_id=math_id, node_type="turn", session_ids=["math"],
                source_turn_ids=[math_id],
                text="Half a number plus five is eleven. What is the number?",
            ),
            family_id: SidecarDocumentV41(
                node_id=family_id, node_type="turn", session_ids=["family"],
                source_turn_ids=[family_id],
                text="I have two brothers and two sisters.",
            ),
        },
    )
    ir = build_query_ir("What is the total number of siblings I have?")
    candidates, trace = _candidate_nodes(
        ir, build_query_plan(ir), sidecar, planner=None, limit=10,
    )
    assert candidates[0] == family_id
    assert math_id not in trace["protected_ids"]



def test_v41_planner_source_selector_is_bounded_and_source_validated() -> None:
    from dataclasses import replace

    from graphmem_demo.clients import rough_token_count
    from graphmem_demo.v41.retrieval import _candidate_nodes, planner_messages
    from graphmem_demo.v41.schema import (
        PlannerResultV41, QuerySidecarV41, SidecarDocumentV41,
    )

    case, _index, _embedder = _parsed()
    case = replace(case, question="How many workshops did I attend last month?")
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    source_ids = [f"memory:s1:turn:{index}" for index in range(10)]
    evidence = [
        {"source_turn_id": source_id, "text": f"I attended workshop {index}."}
        for index, source_id in enumerate(source_ids)
    ]
    messages = planner_messages(case, ir, plan, {
        "present_roles": ["source"], "missing_roles": ["members"],
    }, evidence)
    payload = messages[-1]["content"]
    assert source_ids[7] in payload
    assert source_ids[8] not in payload
    assert rough_token_count("\n".join(row["content"] for row in messages)) <= 700

    parsed = parse_planner_result(
        '{"selected_source_ids":'
        + str(source_ids).replace("'", '"')
        + ',"alternative_entities":[]}'
    )
    assert parsed.selected_source_ids == source_ids[:8]

    valid_id = source_ids[0]
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", inverted={}, adjacency={},
        documents={valid_id: SidecarDocumentV41(
            node_id=valid_id, node_type="turn", session_ids=["s1"],
            source_turn_ids=[valid_id], text="I attended a workshop.",
        )},
    )
    planner = PlannerResultV41(
        selected_source_ids=[valid_id, "fabricated:turn:99"], valid=True,
    )
    candidates, trace = _candidate_nodes(
        ir, plan, sidecar, planner=planner, limit=8,
    )
    assert candidates[0] == valid_id
    assert "fabricated:turn:99" not in candidates
    assert trace["channels"][valid_id]["planner_selected_source"] == 1



def test_v41_labeled_collection_subtotals_are_cross_scope_and_deduplicated() -> None:
    from graphmem_demo.v36.schema import TurnNodeV36
    from graphmem_demo.v41.retrieval import _labeled_collection_subtotals_hint

    _case_value, index, _embedder = _parsed()
    def turn(node_id: str, session: str, date: str, text: str) -> TurnNodeV36:
        return TurnNodeV36(
            node_id=node_id, question_id="q", session_id=session,
            session_date=date, turn_index=0, speaker="Alice",
            speaker_key="alice", listener="Bob", transport_role="user",
            text=text, retrieval_text=text,
        )
    index.turns = [
        turn("q:s1:turn:0", "s1", "2026-01-01",
             "I previously completed 8 Acme courses."),
        turn("q:s2:turn:0", "s2", "2026-01-02",
             "I have already completed 12 courses on BetaLearn."),
        # The same labelled subtotal is an update, not an extra operand.
        turn("q:s3:turn:0", "s3", "2025-12-01",
             "I previously completed 7 Acme courses."),
    ]
    ir = build_query_ir(
        "What is the total number of online courses I've completed?"
    )
    hint = _labeled_collection_subtotals_hint(index, ir)
    assert hint is not None
    assert hint["operation"] == "labeled_collection_subtotal_sum"
    assert hint["value"] == 20
    assert {row["label"] for row in hint["operands"]} == {
        "acme", "betalearn",
    }
    assert hint["certified"] is True


def test_v41_event_collection_uses_graph_identity_and_time_scope() -> None:
    from graphmem_demo.v36.schema import (
        RoleFrameNode, RoutingCard, TurnNodeV36,
    )
    from graphmem_demo.v41.retrieval import _event_collection_members_hint

    _case_value, index, _embedder = _parsed()
    def turn(node_id: str, session: str, text: str) -> TurnNodeV36:
        return TurnNodeV36(
            node_id=node_id, question_id="q", session_id=session,
            session_date="2026-01-30", turn_index=0, speaker="Alice",
            speaker_key="alice", listener="Bob", transport_role="user",
            text=text, retrieval_text=text,
        )
    snake = turn(
        "q:s1:turn:0", "s1",
        "My snake plant, which I got from my sister last month, is thriving.",
    )
    pair = turn(
        "q:s2:turn:0", "s2",
        "I bought the peace lily and a succulent plant two weeks ago.",
    )
    old = turn(
        "q:s3:turn:0", "s3",
        "I bought a cactus plant two months ago.",
    )
    index.turns = [snake, pair, old]
    index.frames = [RoleFrameNode(
        frame_id="q:s1:frame:0", question_id="q", session_ids=["s1"],
        frame_kind="event", owner_key="alice", entity_key="snake plant",
        predicate_key="received", object_key="sister",
        lifecycle_status="completed", state_op="complete",
        source_turn_ids=[snake.node_id], retrieval_text="received snake plant",
    )]
    index.routing_cards = [
        RoutingCard(
            card_id="q:s2:card", question_id="q", session_id="s2",
            speaker_keys=["alice"], canonical_entities=[
                "peace lily", "succulent",
            ], relations=["acquired"], key_events=[
                "acquisition_peace_lily", "acquisition_succulent",
            ], current_states=[], time_range="2026-01",
            frame_ids=[], turn_ids=[pair.node_id],
            routing_text="plants acquired; peace lily; succulent",
        ),
        RoutingCard(
            card_id="q:s3:card", question_id="q", session_id="s3",
            speaker_keys=["alice"], canonical_entities=["cactus"],
            relations=["acquired"], key_events=["acquisition_cactus"],
            current_states=[], time_range="2025-11",
            frame_ids=[], turn_ids=[old.node_id],
            routing_text="plant acquired; cactus",
        ),
    ]
    ir = build_query_ir("How many plants did I acquire in the last month?")
    hint = _event_collection_members_hint(index, ir, "2026-01-31")
    assert hint is not None
    assert hint["value"] == 3
    assert {row["identity"] for row in hint["members"]} == {
        "snake plant", "peace lily", "succulent",
    }
    assert hint["certified"] is True


def test_action_semantics_keeps_all_overlapping_generic_families() -> None:
    from graphmem_demo.v3.action_semantics import action_families

    assert {"complete", "project_work"}.issubset(
        action_families("I completed the project")
    )



def test_v41_temporal_operator_fails_closed_on_ambiguous_lifecycle_dates() -> None:
    from graphmem_demo.v36.operators import temporal_order_source_hint
    from graphmem_demo.v36.schema import TurnNodeV36

    _case_value, index, _embedder = _parsed()
    def turn(node_id: str, session: str, text: str) -> TurnNodeV36:
        return TurnNodeV36(
            node_id=node_id, question_id="q", session_id=session,
            session_date="2026-02-28", turn_index=0, speaker="Alice",
            speaker_key="alice", listener="Bob", transport_role="user",
            text=text, retrieval_text=text,
        )
    phone = turn(
        "q:s1:turn:0", "s1",
        "I pre-ordered the phone on January 28. It was expected on February 11 "
        "and actually arrived on February 25.",
    )
    laptop = turn(
        "q:s2:turn:0", "s2",
        "I got the laptop on February 20.",
    )
    index.turns = [phone, laptop]
    ir = build_query_ir("Which did I get earlier, the phone or the laptop?")
    assert temporal_order_source_hint(
        ir, index, [phone.node_id, laptop.node_id],
    ) is None


def test_v41_certified_temporal_result_is_last_and_keeps_lossless_sources() -> None:
    case, index, embedder = _parsed(
        "Which did I get earlier, the phone or the laptop?"
    )
    view = build_capability_view(index)
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    vectors = embedder.embed(
        query_views(ir, plan), question_id=case.question_id,
        variant="hierarchical_hybrid_graph_v4_1_query",
    )
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=view, sidecar=build_sidecar(index),
        query_ir=ir, query_vectors=vectors, token_budget=9200,
    )
    result.retrieval_trace["v41_evidence_certificate"] = {
        "entity_match": True, "relation_match": True,
        "scope_match": True, "provenance_complete": True,
    }
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "temporal_order_from_lossless_sources",
        "comparison": "earlier", "selected_target": "phone",
        "selected_time": "2026-01-20T00:00:00",
        "event_a_source_turn_id": "q:s1:turn:0",
        "event_b_source_turn_id": "q:s2:turn:0",
        "event_a_time": "2026-01-20T00:00:00",
        "event_b_time": "2026-02-20T00:00:00",
        "certified": True, "binding_complete": True,
    }]
    result.retrieval_trace["v41_temporal_operator_evidence"] = [{
        "source_turn_id": "q:s1:turn:0", "session_date": "2026-02-28",
        "text": "I got the phone last month.",
        "provenance_complete": True,
    }, {
        "source_turn_id": "q:s2:turn:0", "session_date": "2026-02-28",
        "text": "I got the laptop last week.",
        "provenance_complete": True,
    }]
    content = answer_messages(case, result)[-1]["content"]
    marker = "CERTIFIED_TEMPORAL_RESULT (FINAL MANDATORY BINDING):"
    assert marker in content
    assert content.rfind(marker) > content.rfind("DIRECT_DIALOGUE_EVIDENCE")
    assert "I got the phone last month." in content[content.rfind(marker):]
    assert content.rstrip().endswith("larger memory block.")

    lookup_case = replace(
        case, question="When did I get the phone after the trip?",
    )
    result.retrieval_trace["v41_query_augmentation"][
        "answer_algebra"
    ] = "temporal_lookup"
    lookup_content = answer_messages(lookup_case, result)[-1]["content"]
    assert marker not in lookup_content




def test_v41_domain_expansions_cover_reusable_scene_bridges() -> None:
    cases = {
        "My phone battery is weak": {"charge", "power"},
        "Find recent publications and conferences": {
            "paper", "article", "symposium",
        },
        "Rearrange my bedroom furniture": {
            "dresser", "layout", "placement",
        },
    }
    for question, expected in cases.items():
        plan = build_query_plan(build_query_ir(question))
        assert expected.issubset(set(plan.expanded_terms))


def test_v41_best_query_clause_prioritizes_requested_numeric_slot() -> None:
    from graphmem_demo.v41.retrieval import _best_query_clause

    ir = build_query_ir(
        "How much was allocated for influencer marketing in the plan?"
    )
    text = (
        "Detailed influencer marketing campaign plan for the wellness retreat. "
        "Tactics include outreach and social posts. "
        "Budget: Influencer marketing: $2,000. "
        "Timeline: May 1 through May 31."
    )
    assert "$2,000" in _best_query_clause(text, ir)


def test_v41_best_query_clause_prefers_complete_fact_over_topical_question() -> None:
    from graphmem_demo.v41.retrieval import _best_query_clause

    ir = build_query_ir("How many model kits have I worked on or bought?")
    text = (
        "Can you recommend weathering techniques for model kits? "
        "I started working on a diorama featuring a 1/16 scale German Tiger I tank."
    )
    assert "German Tiger I tank" in _best_query_clause(text, ir)
    anaphora = (
        "I am working on a new 1/72 scale B-29 bomber model kit. "
        "I have never tried photo-etching before. "
        "I just got this kit and a 1/24 scale Camaro."
    )
    excerpt = _best_query_clause(anaphora, ir)
    assert "B-29 bomber" in excerpt
    assert "Camaro" in excerpt



def test_v41_planner_evidence_prefers_typed_packed_sources() -> None:
    from graphmem_demo.v41.retrieval import _planner_evidence_candidates
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    good_a = "q:albums:turn:0"
    good_b = "q:albums:turn:2"
    noise = "q:travel:turn:0"
    documents = {
        good_a: SidecarDocumentV41(
            node_id=good_a, node_type="turn", session_ids=["albums"],
            source_turn_ids=[good_a],
            text="I downloaded the album Northern Lights yesterday.",
        ),
        good_b: SidecarDocumentV41(
            node_id=good_b, node_type="turn", session_ids=["albums"],
            source_turn_ids=[good_b],
            text="I bought the EP Midnight Sky at the festival.",
        ),
        noise: SidecarDocumentV41(
            node_id=noise, node_type="turn", session_ids=["travel"],
            source_turn_ids=[noise],
            text="I would love to visit North Ridge and buy a train ticket.",
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", documents=documents,
        inverted={}, adjacency={},
    )
    ir = build_query_ir(
        "How many albums or EPs have I purchased or downloaded?"
    )
    rows = _planner_evidence_candidates(
        [noise, good_a, good_b], ir, sidecar,
        preferred_source_ids=[good_a, good_b, "fabricated:turn:9"],
    )
    assert [row["source_turn_id"] for row in rows] == [good_a, good_b]
    assert all("North Ridge" not in row["text"] for row in rows)


def test_v41_owner_binding_rejects_listener_as_first_person_owner() -> None:
    from graphmem_demo.v41.retrieval import _source_owner_compatible
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    wrong = "q:s1:turn:0"
    right = "q:s1:turn:1"
    reported = "q:s1:turn:2"
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", inverted={}, adjacency={},
        documents={
            wrong: SidecarDocumentV41(
                node_id=wrong, node_type="turn", session_ids=["s1"],
                source_turn_ids=[wrong],
                text="speaker Melanie | listener Caroline | I married my partner.",
            ),
            right: SidecarDocumentV41(
                node_id=right, node_type="turn", session_ids=["s1"],
                source_turn_ids=[right],
                text="speaker Caroline | listener Melanie | I am a single parent.",
            ),
            reported: SidecarDocumentV41(
                node_id=reported, node_type="turn", session_ids=["s1"],
                source_turn_ids=[reported],
                text="speaker Melanie | listener Caroline | Caroline is single.",
            ),
        },
    )
    ir = build_query_ir("What is Caroline's relationship status?")
    assert not _source_owner_compatible(wrong, ir, sidecar)
    assert _source_owner_compatible(right, ir, sidecar)
    assert _source_owner_compatible(reported, ir, sidecar)


def test_v41_reply_bound_channel_binds_reply_to_previous_prompt() -> None:
    from graphmem_demo.v41.domains import augment_query
    from graphmem_demo.v41.retrieval import _reply_bound_turn_rank
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    prompt = "q:s1:turn:0"
    reply = "q:s1:turn:1"
    noise = "q:s2:turn:0"
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", inverted={}, adjacency={},
        documents={
            prompt: SidecarDocumentV41(
                node_id=prompt, node_type="turn", session_ids=["s1"],
                source_turn_ids=[prompt], fields={"owner": ["joanna"]},
                text="speaker Joanna | listener Nate | What flavor did you make?",
            ),
            reply: SidecarDocumentV41(
                node_id=reply, node_type="turn", session_ids=["s1"],
                source_turn_ids=[reply], fields={"owner": ["nate"]},
                text="speaker Nate | listener Joanna | Chocolate and vanilla swirl.",
            ),
            noise: SidecarDocumentV41(
                node_id=noise, node_type="turn", session_ids=["s2"],
                source_turn_ids=[noise], fields={"owner": ["nate"]},
                text="speaker Nate | listener Joanna | I once tried coconut ice cream.",
            ),
        },
    )
    ir = build_query_ir("What flavor of ice cream did Nate make?")
    ranked = _reply_bound_turn_rank(ir, augment_query(ir), sidecar, 3)
    assert ranked[0][0] == reply


def test_v41_planner_evidence_accepts_bounded_unpacked_source() -> None:
    from graphmem_demo.v41.retrieval import _planner_evidence_candidates
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    packed = "q:packed:turn:0"
    scene = "q:scene:turn:2"
    documents = {
        packed: SidecarDocumentV41(
            node_id=packed, node_type="turn", session_ids=["packed"],
            source_turn_ids=[packed], text="We discussed a dance class."
        ),
        scene: SidecarDocumentV41(
            node_id=scene, node_type="turn", session_ids=["scene"],
            source_turn_ids=[scene], text="They compared it to dancing together."
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="test", policy_version="test", documents=documents,
        inverted={}, adjacency={},
    )
    ir = build_query_ir("What did they compare their journey to?")
    rows = _planner_evidence_candidates(
        [packed], ir, sidecar, preferred_source_ids=[scene], limit=4,
    )
    assert rows[0]["source_turn_id"] == scene
    assert "dancing together" in rows[0]["text"]


def test_v41_answer_messages_accepts_noncollection_answer_bearing_trace() -> None:
    case, index, embedder = _parsed("Which camera did Bob recommend?")
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=build_sidecar(index), query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ),
        token_budget=9200,
    )
    result.retrieval_trace["v41_answer_bearing_evidence"] = [{
        "source_turn_id": "q:s1:turn:0",
        "text": "Bob recommended the Atlas camera.",
        "provenance_complete": True,
    }]
    content = answer_messages(case, result)[-1]["content"]
    assert "Bob recommended the Atlas camera." in content


def test_v41_global_lossless_shortlist_is_promoted_with_provenance() -> None:
    from graphmem_demo.v41.retrieval import _global_lossless_focused_evidence

    trace = {"generic_operator_hints": [{
        "operation": "global_lossless_source_candidates",
        "candidates": [{
            "source_turn_id": "q:s1:turn:2",
            "source_date": "2026-01-01",
            "evidence": "I bought the smoker ten days ago.",
            "routing_context": "kitchen appliance purchase",
            "lexical_score": 17,
        }],
        "certified": False,
    }]}
    rows = _global_lossless_focused_evidence(trace, algebra="direct_fact")
    assert rows == [{
        "source_turn_id": "q:s1:turn:2",
        "source_date": "2026-01-01",
        "selection_reason": "global_lossless_query_scene",
        "selection_rank": 1,
        "lexical_score": 17,
        "text": "I bought the smoker ten days ago.",
        "provenance_complete": True,
    }]


def test_v41_source_scanned_collection_members_are_bound_at_prompt_end() -> None:
    case, index, embedder = _parsed(
        "How many albums have I purchased?"
    )
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=build_sidecar(index), query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ),
        token_budget=9200,
    )
    result.retrieval_trace["v41_planner_verified_collection_members"] = [{
        "value": "North Star camera",
        "source_turn_id": "q:s1:turn:0",
        "provenance_complete": True,
    }]
    content = answer_messages(case, result)[-1]["content"]
    marker = "CERTIFIED_COLLECTION_MEMBERS (FINAL MANDATORY BINDING):"
    assert marker in content
    assert content.rfind(marker) > content.rfind("COLLECTION_FINAL_CHECK")
    assert "North Star camera" in content[content.rfind(marker):]
    assert "certified_exact_distinct_count" in content[content.rfind(marker):]
    constraint_start = content.rfind("ANSWER_CONSTRAINT:")
    assert '"operation": "certified_exact_distinct_count"' in content[
        constraint_start:content.rfind(marker)
    ]


def test_v41_global_lossless_match_invalidates_local_absence() -> None:
    from graphmem_demo.v41.retrieval import (
        _invalidate_contradicted_absence,
    )

    case, index, embedder = _parsed("Which camera did Bob recommend?")
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    sidecar = build_sidecar(index)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=sidecar, query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "exact_entity_absence",
        "value": "insufficient",
        "required_marker": "camera",
        "binding_complete": True,
        "certified": True,
    }]
    invalidated = _invalidate_contradicted_absence(
        result, index, ir, sidecar, 9200,
    )
    assert invalidated
    assert not any(
        row.get("operation") == "exact_entity_absence"
        for row in result.retrieval_trace["generic_operator_hints"]
    )
    assert "v41_global_exact_recovery_decisions" in result.retrieval_trace


def test_v41_noncontiguous_compound_terms_do_not_invalidate_absence() -> None:
    from graphmem_demo.v41.retrieval import (
        _invalidate_contradicted_absence,
    )

    case, index, embedder = _parsed(
        "How many engineers do I lead as a Software Engineer Manager?"
    )
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:team", session_id="s-team",
            transport_role="user",
            text="I lead four engineers, plus my manager Rachel.",
        )
    ]
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    sidecar = build_sidecar(index)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=sidecar, query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "exact_entity_absence",
        "value": "insufficient",
        "required_phrase": "engineer manager",
        "binding_complete": True,
        "certified": True,
    }]
    assert not _invalidate_contradicted_absence(
        result, index, ir, sidecar, 9200,
    )
    assert result.retrieval_trace["generic_operator_hints"][0][
        "operation"
    ] == "exact_entity_absence"


def test_v41_certified_absence_is_the_final_mandatory_binding() -> None:
    case, index, embedder = _parsed("How often do I see Dr. Johnson?")
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=build_sidecar(index), query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "exact_entity_absence",
        "value": "insufficient",
        "required_marker": "johnson",
        "binding_kind": "named_entity",
        "binding_complete": True,
        "certified": True,
    }]
    content = answer_messages(case, result)[-1]["content"]
    marker = "CERTIFIED_EXACT_ENTITY_ABSENCE (FINAL MANDATORY BINDING):"
    assert marker in content
    assert content.rfind(marker) > content.rfind("FOCUSED_SOURCE_EVIDENCE")
    assert '"required_marker": "johnson"' in content[content.rfind(marker):]


def test_v41_scene_coverage_certifies_identity_sets_not_cumulative_values() -> None:
    from graphmem_demo.v41.retrieval import (
        _planner_collection_exact_binding_safe,
    )

    members = [
        {"value": "Sarah's place", "source_turn_id": "q:s1:turn:0"},
        {"value": "Alex's place", "source_turn_id": "q:s2:turn:4"},
        {"value": "Mike's place", "source_turn_id": "q:s2:turn:4"},
    ]
    evidence = [
        {"source_turn_id": "q:s1:turn:0",
         "selection_features": ["attendance_bound"]},
        {"source_turn_id": "q:s1:turn:2",
         "selection_features": ["attendance_bound"]},
        {"source_turn_id": "q:s2:turn:4",
         "selection_features": ["attendance_bound"]},
    ]
    assert _planner_collection_exact_binding_safe(
        "How many dinner parties did I attend?", members, evidence,
    )
    assert not _planner_collection_exact_binding_safe(
        "How many bikes do I own?",
        members[:2],
        [*evidence, {
            "source_turn_id": "q:s3:turn:0",
            "selection_features": ["possession_bound"],
        }],
    )
    assert not _planner_collection_exact_binding_safe(
        "How many sessions did I attend?",
        [
            {"value": "three sessions", "source_turn_id": "q:s1:turn:0"},
            {"value": "five sessions", "source_turn_id": "q:s2:turn:0"},
        ],
        evidence,
    )


def test_v41_open_world_planner_members_are_not_forced_as_exact_count() -> None:
    case, index, embedder = _parsed(
        "How many magazine subscriptions do I currently have?"
    )
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=build_sidecar(index), query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    result.retrieval_trace["v41_planner_verified_collection_members"] = [{
        "value": "The New Yorker",
        "source_turn_id": "q:s1:turn:0",
        "provenance_complete": True,
    }]
    content = answer_messages(case, result)[-1]["content"]
    assert "CERTIFIED_COLLECTION_MEMBERS" not in content
    assert '"operation": "certified_exact_distinct_count"' not in content


def test_v41_collection_source_closure_rejects_meal_kit_and_plans() -> None:
    from graphmem_demo.v36.schema import RoutingCard, TurnNodeV36
    from graphmem_demo.v41.domains import augment_query
    from graphmem_demo.v41.retrieval import _collection_source_candidates

    _case_value, index, _embedder = _parsed()
    def turn(node_id: str, session: str, text: str) -> TurnNodeV36:
        return TurnNodeV36(
            node_id=node_id, question_id="q", session_id=session,
            session_date="2026-01-30", turn_index=0, speaker="Alice",
            speaker_key="alice", listener="Bob", transport_role="user",
            text=text, retrieval_text=text,
        )
    completed = turn(
        "q:model:turn:0", "model",
        "I recently finished a 1/48 scale aircraft model kit.",
    )
    active = turn(
        "q:tank:turn:0", "tank",
        "I started working on a diorama featuring a scale Tiger tank.",
    )
    meal = turn(
        "q:meal:turn:0", "meal",
        "I bought my first meal kit delivery last week.",
    )
    planned = turn(
        "q:plan:turn:0", "plan",
        "I am planning to work on a new model kit next month.",
    )
    index.turns = [completed, active, meal, planned]
    index.routing_cards = [RoutingCard(
        card_id=f"q:{item.session_id}:card", question_id="q",
        session_id=item.session_id, speaker_keys=["alice"],
        canonical_entities=[], relations=[], key_events=[], current_states=[],
        time_range="2026-01", frame_ids=[], turn_ids=[item.node_id],
        routing_text=item.text,
    ) for item in index.turns]
    ir = build_query_ir("How many model kits have I worked on or bought?")
    nodes, evidence = _collection_source_candidates(
        index, ir, augment_query(ir), "2026-01-31",
    )
    assert completed.node_id in nodes
    assert active.node_id in nodes
    assert meal.node_id not in nodes
    assert planned.node_id not in nodes
    assert all(row["provenance_complete"] is True for row in evidence)


def test_v41_collection_closure_accepts_only_attributed_reply_confirmation() -> None:
    from graphmem_demo.v36.schema import RoutingCard, TurnNodeV36
    from graphmem_demo.v41.domains import augment_query
    from graphmem_demo.v41.retrieval import _collection_source_candidates

    user = TurnNodeV36(
        node_id="q:delivery:turn:0", question_id="q", session_id="delivery",
        session_date="2026-01-30", turn_index=0, speaker="Alice",
        speaker_key="alice", listener="Assistant", transport_role="user",
        text="I rely on food delivery services every weekend.",
        retrieval_text="I rely on food delivery services every weekend.",
    )
    reply = TurnNodeV36(
        node_id="q:delivery:turn:1", question_id="q", session_id="delivery",
        session_date="2026-01-30", turn_index=1, speaker="Assistant",
        speaker_key="assistant", listener="Alice", transport_role="assistant",
        text=(
            "I'm glad to hear that Uber Eats has been useful for your weekends! "
            "You could also try Door Dash."
        ),
        retrieval_text="Uber Eats has been useful for your weekends.",
    )
    _case_value, index, _embedder = _parsed()
    index.turns = [user, reply]
    index.routing_cards = [RoutingCard(
        card_id="q:delivery:card", question_id="q", session_id="delivery",
        speaker_keys=["alice"], canonical_entities=["food delivery service"],
        relations=["used"], key_events=["used Uber Eats"], current_states=[],
        time_range="2026-01", frame_ids=[],
        turn_ids=[user.node_id, reply.node_id],
        routing_text="Alice used Uber Eats for food delivery.",
    )]
    ir = build_query_ir(
        "How many different food delivery services have I used?"
    )
    nodes, evidence = _collection_source_candidates(
        index, ir, augment_query(ir), "2026-01-31",
    )
    assert user.node_id in nodes
    assert reply.node_id in nodes
    reply_row = next(
        row for row in evidence if row["source_turn_id"] == reply.node_id
    )
    assert "Uber Eats" in reply_row["text"]
    assert "Door Dash" not in reply_row["text"]
    assert "dialogue_confirmation" in reply_row["selection_features"]



def test_v41_collection_source_closure_binds_current_owner_and_care() -> None:
    from graphmem_demo.v36.schema import RoutingCard, TurnNodeV36
    from graphmem_demo.v41.domains import augment_query
    from graphmem_demo.v41.retrieval import _collection_source_candidates

    _case_value, index, _embedder = _parsed()
    def turn(node_id: str, session: str, text: str) -> TurnNodeV36:
        return TurnNodeV36(
            node_id=node_id, question_id="q", session_id=session,
            session_date="2026-01-30", turn_index=0, speaker="Alice",
            speaker_key="alice", listener="Bob", transport_role="user",
            text=text, retrieval_text=text,
        )
    piano = turn("q:piano:turn:0", "piano", "I've had my Korg piano for three years.")
    niece = turn("q:niece:turn:0", "niece", "My niece just got her first violin.")
    plan = turn("q:uke:turn:0", "uke", "I'll get my new ukulele next month.")
    index.turns = [piano, niece, plan]
    index.routing_cards = [RoutingCard(
        card_id=f"q:{item.session_id}:card", question_id="q",
        session_id=item.session_id, speaker_keys=["alice"],
        canonical_entities=[], relations=[], key_events=[], current_states=[],
        time_range="2026-01", frame_ids=[], turn_ids=[item.node_id],
        routing_text=item.text,
    ) for item in index.turns]
    ir = build_query_ir("How many musical instruments do I currently own?")
    nodes, _evidence = _collection_source_candidates(
        index, ir, augment_query(ir), "2026-01-31",
    )
    assert nodes == [piano.node_id]

    personal = turn(
        "q:care:turn:0", "care",
        "I was diagnosed by an ENT specialist and prescribed a nasal spray.",
    )
    software = turn(
        "q:saas:turn:0", "saas",
        "Can a clinic have unlimited doctors access the back-end system?",
    )
    index.turns = [personal, software]
    index.routing_cards = []
    ir = build_query_ir("How many different doctors did I visit?")
    nodes, _evidence = _collection_source_candidates(
        index, ir, augment_query(ir), "2026-01-31",
    )
    assert nodes == [personal.node_id]



def test_v41_relative_anchor_answer_candidate_is_final_constraint() -> None:
    case, index, embedder = _parsed(
        "Which bike did I fix or service last weekend?"
    )
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=build_sidecar(index), query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "relative_anchor_source_lookup",
        "answer_candidate": "road bike",
        "source_turn_ids": ["q:s1:turn:0"],
        "binding_complete": True, "certified": True,
    }]
    content = answer_messages(case, result)[-1]["content"]
    marker = "CERTIFIED_OPERATOR_RESULT (FINAL MANDATORY BINDING):"
    assert marker in content
    assert '"answer_candidate": "road bike"' in content[content.rfind(marker):]
    assert 'FINAL_OUTPUT_MUST_CONTAIN: ["road bike"]' in content
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "threshold_progress_remaining",
        "value": 100, "unit": "points",
        "source_turn_ids": ["q:s1:turn:0"],
        "binding_complete": True, "certified": True,
    }]
    scalar_content = answer_messages(case, result)[-1]["content"]
    assert "CERTIFIED_FINAL_ANSWER_SLOT" in scalar_content
    assert '"value": 100' in scalar_content
    assert "Memory evidence:" not in scalar_content



def test_v41_source_date_and_explicit_unit_duration_are_final_slots() -> None:
    case, index, embedder = _parsed("When did I launch the site?")
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=build_sidecar(index), query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    result.retrieval_trace["v41_evidence_certificate"] = {
        "entity_match": True, "relation_match": True,
        "scope_match": True, "provenance_complete": True,
    }
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "source_bound_explicit_date",
        "value": "2023-07-02", "source_turn_ids": ["q:s1:turn:0"],
        "binding_complete": True, "certified": True,
    }]
    date_content = answer_messages(case, result)[-1]["content"]
    assert "CERTIFIED_FINAL_ANSWER_SLOT" in date_content
    assert "2023-07-02" in date_content
    result.retrieval_trace["query_ir"] = {
        "temporal_constraints": ["after"],
        "comparison_targets": ["walk", "trip"],
    }
    scoped_content = answer_messages(case, result)[-1]["content"]
    assert "CERTIFIED_FINAL_ANSWER_SLOT" not in scoped_content
    result.retrieval_trace["query_ir"] = {}

    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "duration_total", "value": 19, "unit": "days",
        "frame_ids": ["q:s1:frame:0"],
        "binding_complete": True, "certified": True,
    }]
    explicit_case = replace(case, question="How many days did the trip take?")
    explicit_content = answer_messages(explicit_case, result)[-1]["content"]
    assert "CERTIFIED_FINAL_ANSWER_SLOT" in explicit_content
    assert "19" in explicit_content

    ambiguous_case = replace(case, question="How long did the trip take?")
    ambiguous_content = answer_messages(ambiguous_case, result)[-1]["content"]
    assert "CERTIFIED_FINAL_ANSWER_SLOT" not in ambiguous_content


def test_v41_local_temporal_pair_survives_unrelated_global_role_gap() -> None:
    case, index, embedder = _parsed(
        "How many days passed since I launched the site when I signed the contract?"
    )
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_1_query",
        index=index, capability_view=build_capability_view(index),
        sidecar=build_sidecar(index), query_ir=ir,
        query_vectors=embedder.embed(
            query_views(ir, plan), question_id=case.question_id,
            variant="hierarchical_hybrid_graph_v4_1_query",
        ), token_budget=9200,
    )
    result.retrieval_trace["v41_evidence_certificate"] = {
        "entity_match": True,
        "relation_match": True,
        "scope_match": True,
        "provenance_complete": True,
        "missing_roles": ["event_a", "event_b"],
        "complete": False,
    }
    result.retrieval_trace["generic_operator_hints"] = [{
        "operation": "time_difference_from_lossless_sources",
        "value": 19,
        "unit": "days",
        "event_a_source_turn_id": "q:s1:turn:0",
        "event_b_source_turn_id": "q:s1:turn:1",
        "event_a_time": "2023-02-10T00:00:00",
        "event_b_time": "2023-03-01T00:00:00",
        "binding_complete": True,
        "certified": True,
    }]
    content = answer_messages(case, result)[-1]["content"]
    assert "CERTIFIED_FINAL_ANSWER_SLOT" in content
    assert '"value": 19' in content
    assert '"unit": "days"' in content


def test_v41_refreshes_threshold_operator_after_source_expansion() -> None:
    from types import SimpleNamespace
    from graphmem_demo.v41.retrieval import (
        _refresh_post_expansion_operator_hints,
    )

    _case_value, index, _embedder = _parsed()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:current", session_date="2023/05/21",
            transport_role="user",
            text=(
                "I'm looking for skincare products. I earned 50 points, "
                "bringing my total to 200 points so far at Sephora."
            ),
        ),
        replace(
            template, node_id="q:target", session_date="2023/05/29",
            transport_role="user",
            text=(
                "To redeem a free skincare product at Sephora, "
                "I need a total of 300 points."
            ),
        ),
    ]
    result = SimpleNamespace(retrieval_trace={
        "generic_operator_hints": [{
            "operation": "latest_scalar_state_from_lossless_sources",
            "value": 300,
        }],
    })
    refreshed = _refresh_post_expansion_operator_hints(
        result,
        index,
        build_query_ir(
            "How many points do I need to earn to redeem a free "
            "skincare product at Sephora?"
        ),
        ["q:current", "q:target"],
    )
    assert refreshed[0]["operation"] == "threshold_progress_remaining"
    assert refreshed[0]["value"] == 100
    assert all(
        row.get("operation")
        != "latest_scalar_state_from_lossless_sources"
        for row in result.retrieval_trace["generic_operator_hints"]
    )



def test_v41_owner_normalization_and_auxiliary_have_disambiguation() -> None:
    relationship = build_query_ir("What is Caroline's relationship status?")
    assert relationship.target_owner == "caroline"
    media = build_query_ir("What movies have both Joanna and Nate seen?")
    plan = build_query_plan(media)
    assert media.target_owner == ""
    assert {"watch", "watched", "saw", "viewed"}.intersection(
        plan.expanded_terms
    )
    assert not {"own", "owned", "possess", "received", "got"}.intersection(
        plan.expanded_terms
    )


def test_v41_inferential_profile_is_a_general_planner_branch() -> None:
    for question in (
        "What career might Dana pursue in the future?",
        "Would Priya likely enjoy a classical concert?",
        "Which exercise could Lee benefit from?",
    ):
        plan = build_query_plan(build_query_ir(question))
        assert plan.answer_algebra == "inferential_profile"
        assert plan.planner_required is True
        assert {"profile_fact", "support", "source"}.issubset(
            plan.required_roles
        )


def test_v41_before_after_date_binds_both_events() -> None:
    ir = build_query_ir(
        "When did Melanie go on a hike after the roadtrip?"
    )
    assert ir.requested_value_type == "date"
    assert ir.comparison_targets == [
        "melanie go hike", "roadtrip",
    ]
    assert {"event", "time", "identity", "source"}.issubset(
        ir.required_roles
    )


def test_v41_collection_planner_can_return_retrieval_aliases() -> None:
    from graphmem_demo.models import QuestionCase
    from graphmem_demo.v41.retrieval import planner_messages

    case = QuestionCase(
        question_id="q:media", question_type="list",
        question="What movies have both Joanna and Nate seen?",
        question_date=None, answer="", answer_session_ids=[],
        haystack_sessions=[], haystack_session_ids=[], haystack_dates=[],
    )
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    messages = planner_messages(case, ir, plan, {}, [])
    system = messages[0]["content"]
    assert "alternative_entities" in system
    assert "event_aliases" in system
    assert "relations" in system
    assert "member_candidates" in system


def test_v41_collection_planner_requires_category_child_terms_on_role_gap() -> None:
    from graphmem_demo.clients import rough_token_count
    from graphmem_demo.models import QuestionCase
    from graphmem_demo.v41.retrieval import planner_messages

    case = QuestionCase(
        question_id="q:category", question_type="count",
        question="How many kinds of musical instruments have I repaired?",
        question_date=None, answer="", answer_session_ids=[],
        haystack_sessions=[], haystack_session_ids=[], haystack_dates=[],
    )
    ir = build_query_ir(case.question)
    plan = build_query_plan(ir)
    messages = planner_messages(
        case, ir, plan,
        {"present_roles": ["source"], "missing_roles": ["scope", "members"]},
        [],
    )
    system = messages[0]["content"]
    payload = messages[1]["content"]
    assert "up to eight common child-type names" in system
    assert "even without evidence candidates" in system
    assert "\"missing_roles\": [\"scope\", \"members\"]" in payload
    assert rough_token_count("\n".join(row["content"] for row in messages)) <= 700




def test_v41_owner_lifecycle_dense_channel_excludes_plans_and_wrong_roles() -> None:
    from graphmem_demo.v36.schema import TurnNodeV36, V36Index
    from graphmem_demo.v41.retrieval import _owner_lifecycle_dense_turn_rank

    def turn(
        node_id: str, session: str, role: str, text: str,
        embedding: list[float],
    ) -> TurnNodeV36:
        return TurnNodeV36(
            node_id=node_id, question_id="q", session_id=session,
            session_date="2026-01-15", turn_index=0, speaker="Dana",
            speaker_key="dana", listener="Riley", transport_role=role,
            text=text, retrieval_text=text, embedding=embedding,
        )

    kept_a = turn(
        "q:s1:turn:0", "s1", "user",
        "I repaired my wooden flute yesterday.", [1.0, 0.0],
    )
    kept_b = turn(
        "q:s1:turn:1", "s1", "user",
        "I repaired my brass horn last week.", [0.9, 0.1],
    )
    index = V36Index(turns=[
        kept_a,
        kept_b,
        turn("q:s1:turn:2", "s1", "user",
             "I plan to repair my drum next week.", [1.0, 0.0]),
        turn("q:s1:turn:3", "s1", "user",
             "I never repaired the borrowed violin.", [1.0, 0.0]),
        turn("q:s1:turn:4", "s1", "assistant",
             "I repaired a cello yesterday.", [1.0, 0.0]),
        turn("q:s2:turn:0", "s2", "user",
             "I repaired a saxophone yesterday.", [1.0, 0.0]),
    ])
    ir = build_query_ir("How many musical instruments have I repaired?")
    plan = build_query_plan(ir)
    ranked = _owner_lifecycle_dense_turn_rank(
        index, ir, plan, [[1.0, 0.0]], {"s1"}, limit=8,
    )
    assert [node_id for node_id, _score in ranked] == [
        kept_a.node_id, kept_b.node_id,
    ]

def test_v41_geographic_state_is_not_a_lifecycle_state() -> None:
    ir = build_query_ir("What state did Nate visit during his trip?")
    plan = build_query_plan(ir)
    assert ir.requested_value_type == "location"
    assert {"event", "location", "source"}.issubset(ir.required_roles)
    assert plan.answer_algebra != "state_update"


def test_v41_general_profile_inference_question_forms() -> None:
    for question in (
        "Does Dana live closer to the mountains or the beach?",
        "What other exercises can help Priya recover?",
        "How old is Lee likely to be?",
    ):
        plan = build_query_plan(build_query_ir(question))
        assert plan.answer_algebra == "inferential_profile"
        assert plan.planner_required is True


def test_v41_owner_supports_modal_and_relative_clause_forms() -> None:
    assert build_query_ir(
        "What kind of yoga might John benefit from?"
    ).target_owner == "john"
    assert build_query_ir(
        "What is a Star Wars book that Tim might enjoy?"
    ).target_owner == "tim"


def test_v41_inference_candidates_do_not_recertify_question_entities() -> None:
    from graphmem_demo.v41.retrieval import _verified_inference_candidates
    from graphmem_demo.v41.schema import (
        PlannerResultV41, QuerySidecarV41, SidecarDocumentV41,
    )
    ir = build_query_ir("What kind of yoga might John benefit from?")
    planner = PlannerResultV41(
        alternative_entities=["John", "Hatha Yoga"], valid=True,
    )
    sidecar = QuerySidecarV41(
        index_hash="x", policy_version="test",
        documents={
            "turn:1": SidecarDocumentV41(
                node_id="turn:1", node_type="turn", session_ids=["s"],
                source_turn_ids=["turn:1"],
                text="speaker John | Hatha Yoga may build core strength.",
            ),
        },
        inverted={"speaker": {"john": ["turn:1"], "tim": ["turn:1"]}},
        adjacency={},
    )
    planner.alternative_entities.insert(1, "Tim")
    rows = _verified_inference_candidates(
        planner, ["turn:1"], ir, sidecar,
    )
    assert [row["candidate"] for row in rows] == ["Hatha Yoga"]


def test_v41_owner_supports_for_name_to_form() -> None:
    assert build_query_ir(
        "What hobby would be good for Tim to pick up?"
    ).target_owner == "tim"


def test_v41_location_source_rank_prefers_requested_owner() -> None:
    from graphmem_demo.v41.retrieval import _location_source_rank
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41
    ir = build_query_ir("What state did Nate visit?")
    docs = {
        "nate": SidecarDocumentV41(
            node_id="nate", node_type="turn", session_ids=["s1"],
            source_turn_ids=["nate"],
            text="speaker Nate | I took my turtles to the beach in Tampa.",
            fields={"owner": ["nate"]},
        ),
        "joanna": SidecarDocumentV41(
            node_id="joanna", node_type="turn", session_ids=["s2"],
            source_turn_ids=["joanna"],
            text="speaker Joanna | I traveled to Boston.",
            fields={"owner": ["joanna"]},
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="x", policy_version="test", documents=docs,
        inverted={}, adjacency={},
    )
    rows = _location_source_rank(ir, sidecar)
    assert [row[0] for row in rows[:2]] == ["nate", "joanna"]


def test_v41_planner_parses_general_inference_candidates() -> None:
    from graphmem_demo.v41.retrieval import parse_planner_result
    plan = parse_planner_result(
        '{"alternative_entities":[],"event_aliases":[],"relations":[],'
        '"temporal_constraints":[],"missing_roles":[],"selected_source_ids":[],'
        '"inference_candidates":["Concrete title","Specific activity"]}'
    )
    assert plan.valid is True
    assert plan.inference_candidates == ["Concrete title", "Specific activity"]


def test_v41_possessive_owner_is_not_a_relation_concept() -> None:
    from graphmem_demo.v41.retrieval import _query_base_terms

    ir = build_query_ir("What is Caroline\u0027s relationship status?")
    assert ir.target_owner == "caroline"
    assert "caroline" not in _query_base_terms(ir)
    assert {"relationship", "status"}.issubset(_query_base_terms(ir))


def test_v41_relationship_status_expands_state_scene_vocabulary() -> None:
    plan = build_query_plan(
        build_query_ir("What is Caroline\u0027s relationship status?")
    )
    assert {"single parent", "breakup", "married"}.issubset(
        set(plan.expanded_terms)
    )


def test_v41_planner_slot_candidates_are_source_bound() -> None:
    from graphmem_demo.v41.retrieval import (
        _verified_planner_slot_candidates,
    )
    from graphmem_demo.v41.schema import (
        QuerySidecarV41, SidecarDocumentV41,
    )

    parsed = parse_planner_result(
        "{\"selected_source_ids\":[\"q:turn:1\"],"
        "\"slot_candidates\":[[\"tasty soup with sage\",\"q:turn:1\"],"
        "[\"invented stew\",\"q:turn:2\"]]}"
    )
    assert parsed.slot_candidates[0]["value"] == "tasty soup with sage"
    sidecar = QuerySidecarV41(
        index_hash="x", policy_version="test",
        documents={
            "q:turn:1": SidecarDocumentV41(
                node_id="q:turn:1", node_type="turn", session_ids=["s"],
                source_turn_ids=["q:turn:1"],
                text="I recently made a tasty soup with sage.",
            ),
            "q:turn:2": SidecarDocumentV41(
                node_id="q:turn:2", node_type="turn", session_ids=["s"],
                source_turn_ids=["q:turn:2"], text="No food is named here.",
            ),
        },
        inverted={}, adjacency={},
    )
    ir = build_query_ir("What kind of soup did John make recently?")
    assert _verified_planner_slot_candidates(
        parsed, ["q:turn:1"], ir, sidecar,
    ) == [{
        "value": "tasty soup with sage",
        "source_turn_id": "q:turn:1",
        "source_text": "I recently made a tasty soup with sage.",
        "provenance_complete": True,
    }]


def test_v41_comparison_query_uses_analogy_scene_terms() -> None:
    plan = build_query_plan(build_query_ir(
        "What did Dana compare the project journey to?"
    ))
    assert {"like", "analogy", "metaphor"}.issubset(
        set(plan.expanded_terms)
    )


def test_v41_scoped_relative_date_binds_local_event_scene() -> None:
    from graphmem_demo.v36.schema import TurnNodeV36, V36Index
    from graphmem_demo.v41.retrieval import (
        _scoped_relative_date_hint, _source_date,
    )

    assert _source_date("6:55 pm on 20 October, 2023").strftime("%Y-%m-%d") == "2023-10-20"
    prefix = "memory:session_1:turn:"
    turns = [
        TurnNodeV36(
            node_id=prefix + "0", question_id="q", session_id="s1",
            session_date="9:00 am on 5 March, 2025", turn_index=0,
            speaker="Alex", speaker_key="alex", listener="Blair",
            transport_role="user", text="The audit is finally complete.",
            retrieval_text="The audit is finally complete.",
        ),
        TurnNodeV36(
            node_id=prefix + "1", question_id="q", session_id="s1",
            session_date="9:00 am on 5 March, 2025", turn_index=1,
            speaker="Blair", speaker_key="blair", listener="Alex",
            transport_role="assistant", text="Did you deploy after the audit?",
            retrieval_text="Did you deploy after the audit?",
        ),
        TurnNodeV36(
            node_id=prefix + "2", question_id="q", session_id="s1",
            session_date="9:00 am on 5 March, 2025", turn_index=2,
            speaker="Alex", speaker_key="alex", listener="Blair",
            transport_role="user", text="Yes, I did it yesterday.",
            retrieval_text="Yes, I did it yesterday.",
        ),
    ]
    ir = build_query_ir("When did Alex deploy after the audit?")
    hint = _scoped_relative_date_hint(
        V36Index(turns=turns), [turn.node_id for turn in turns], ir,
    )
    assert hint is not None
    assert hint["value"] == "2025-03-04"
    assert hint["source_turn_ids"] == [turn.node_id for turn in turns]
    assert hint["operator_certificate"] == {
        "entity_match": True, "relation_match": True,
        "scope_match": True, "provenance_complete": True,
    }



def test_v41_media_reaction_scene_adds_immediate_reply() -> None:
    from graphmem_demo.v41.retrieval import _scene_window_nodes
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    prefix = "memory:session_1:turn:"
    documents = {
        prefix + "4": SidecarDocumentV41(
            node_id=prefix + "4", node_type="turn", session_ids=["s"],
            source_turn_ids=[prefix + "4"],
            text="I shared a photo of dancers on a stage. [Media shared; caption: dancers]",
        ),
        prefix + "5": SidecarDocumentV41(
            node_id=prefix + "5", node_type="turn", session_ids=["s"],
            source_turn_ids=[prefix + "5"], text="They look graceful.",
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="x", policy_version="test", documents=documents,
        inverted={}, adjacency={},
    )
    ir = build_query_ir("What did Dana say about the dancers in the photo?")
    assert prefix + "5" in _scene_window_nodes(
        [prefix + "4"], ir, sidecar,
    )


def test_v41_planner_parser_recovers_rich_alias_objects() -> None:
    parsed = parse_planner_result(
        "{\"event_aliases\":[{\"entity\":\"relationship status\","
        "\"aliases\":[\"single\",\"married\"]}],"
        "\"relations\":[{\"relation\":\"current relationship\"}]}"
    )
    assert parsed.valid is True
    assert {"relationship status", "single", "married"}.issubset(
        set(parsed.event_aliases)
    )
    assert parsed.relations == ["current relationship"]


def test_v41_forward_slot_closure_keeps_elliptical_reply() -> None:
    from graphmem_demo.v41.retrieval import _scene_window_nodes
    from graphmem_demo.v41.schema import QuerySidecarV41, SidecarDocumentV41

    prefix = "memory:session_1:turn:"
    documents = {
        prefix + "5": SidecarDocumentV41(
            node_id=prefix + "5", node_type="turn", session_ids=["s"],
            source_turn_ids=[prefix + "5"],
            text="You need to practice first, then we can play together.",
        ),
        prefix + "6": SidecarDocumentV41(
            node_id=prefix + "6", node_type="turn", session_ids=["s"],
            source_turn_ids=[prefix + "6"], text="I hope it is easy to control.",
        ),
        prefix + "7": SidecarDocumentV41(
            node_id=prefix + "7", node_type="turn", session_ids=["s"],
            source_turn_ids=[prefix + "7"],
            text="All you need is a gamepad and a sense of timing.",
        ),
    }
    sidecar = QuerySidecarV41(
        index_hash="x", policy_version="test", documents=documents,
        inverted={}, adjacency={},
    )
    ir = build_query_ir(
        "What did Dana suggest Lee practice before playing together?"
    )
    rows = _scene_window_nodes([prefix + "5"], ir, sidecar)
    assert prefix + "6" in rows
    assert prefix + "7" in rows
