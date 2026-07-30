from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from graphmem_demo.clients import (
    MockCompressor, MockDeepSeekClient, MockEmbeddingClient,
)
from graphmem_demo.models import QuestionCase
from graphmem_demo.pipeline import (
    BuildMetrics, DemoConfig, InflightLimiter, MemoryBuild,
    _build_v36_memory, _load_memory_cache, _memory_cache_fingerprint,
    _v36_index_diagnostics_payload, _write_memory_cache, run_case,
)
from graphmem_demo.v36.build import (
    _attach_routing_relations, add_routing_semantic_edges,
    build_index,
    build_turn_nodes,
    parse_session_extraction,
    session_extraction_messages,
    validate_index,
)
from graphmem_demo.v36.retrieval import (
    _adaptive_cards, _certificate, _content_tokens, _focused_turn_text, _pack,
    _structured_rank, build_query_ir, retrieve,
)
from graphmem_demo.v36.schema import EvidenceGroup, RoleFrameNode, RoutingCard


def _case(question: str = "What did Alice buy?") -> QuestionCase:
    return QuestionCase(
        question_id="q",
        question_type="unknown",
        question=question,
        answer="camera",
        question_date="2026-01-03",
        haystack_sessions=[[
            {
                "role": "user", "speaker": "Alice", "listener": "Bob",
                "content": "Which camera should I buy?",
            },
            {
                "role": "assistant", "speaker": "Bob", "listener": "Alice",
                "content": "Buy the Lumina X2 camera.",
            },
        ]],
        haystack_session_ids=["s"],
        haystack_dates=["2026-01-02"],
        answer_session_ids=["s"],
        memory_cache_key="v36-test",
    )


def _parsed(question: str = "Which camera did Bob recommend?"):
    case = _case(question)
    turns = build_turn_nodes(case)
    payload = {
        "frames": [
            [
                "fact", "Alice", "camera", "asks recommendation", "camera",
                "purchase", "positive", "asserted", "proposed", "none",
                None, "", None, None, None, None, "unknown", None, "",
                ["request"], [turns[0].node_id], 0.95,
            ],
            [
                "dialogue_answer", "Bob", "camera", "recommended", "Lumina X2",
                "purchase", "positive", "asserted", "proposed", "none",
                None, "", None, None, None, None, "unknown", None, "",
                ["recommendation"], [turns[1].node_id], 0.98,
            ],
        ],
        "routing_card": {
            "entities": ["Alice", "Bob", "Lumina X2"],
            "relations": ["asks recommendation", "recommended"],
            "events": ["camera recommendation"],
            "current_states": [],
            "time_range": "2026-01-02",
        },
        "coverage": [
            [turns[0].node_id, "dialogue_context", [0]],
            [turns[1].node_id, "memory_frame", [1]],
        ],
    }
    frames, card, coverage, error = parse_session_extraction(
        json.dumps(payload),
        question_id=case.question_id,
        session_id="s",
        session_date="2026-01-02",
        turns=turns,
    )
    assert error is None
    index = build_index(
        question_id="q", turns=turns, frames=frames,
        routing_cards=[card], coverage=coverage,
    )
    embedder = MockEmbeddingClient()
    for nodes, attr in (
        (index.turns, "retrieval_text"),
        (index.frames, "retrieval_text"),
        (index.routing_cards, "routing_text"),
        (index.evidence_groups, "retrieval_text"),
    ):
        vectors = embedder.embed(
            [getattr(node, attr) for node in nodes],
            question_id="q", variant="hierarchical_role_graph_v3_6",
        )
        for node, vector in zip(nodes, vectors):
            node.embedding = vector
    add_routing_semantic_edges(index)
    return case, index, embedder


def test_extraction_protocol_uses_named_frame_objects() -> None:
    turns = build_turn_nodes(_case())
    messages = session_extraction_messages("s", "2026-01-02", turns)
    payload = json.loads(messages[1]["content"])
    assert isinstance(payload["F"], dict)
    assert {"kind", "entity", "predicate", "sources"} <= set(payload["F"])
    assert "never emit positional arrays" in messages[0]["content"]
    assert "coverage:[[Tn,class]]" in messages[0]["content"]
    assert "frame indexes" in messages[0]["content"]
    assert "zero-based frame indexes" not in messages[0]["content"]



def test_extraction_view_is_bounded_but_turn_nodes_remain_lossless() -> None:
    long_middle = "durable middle value 42. " * 180
    case = _case()
    case.haystack_sessions[0][0]["content"] = "opening fact. " + long_middle + "closing fact."
    turns = build_turn_nodes(case)
    original = turns[0].text
    payload = json.loads(session_extraction_messages("s", "2026-01-02", turns)[1]["content"])
    assert len(payload["T"][0][3]) <= 1800
    assert payload["T"][0][3] != original
    assert turns[0].text == original
    assert 6 <= payload["B"] <= 40


def test_v36_diagnostics_split_failure_categories_and_recovery() -> None:
    payload = _v36_index_diagnostics_payload([
        {
            "stage": "v36_session_extraction", "session_id": "s0",
            "parse_error": "invalid_json", "local_lossless_frame_count": 0,
            "lossless_only_count": 2,
        },
        {
            "stage": "v36_session_extraction", "session_id": "s1",
            "parse_error": None, "local_lossless_frame_count": 3,
            "lossless_only_count": 1,
        },
    ])
    summary = payload["summary"]
    assert summary["parse_validation_failure_count"] == 1
    assert summary["invalid_json_count"] == 1
    assert summary["empty_frames_count"] == 0
    assert summary["coverage_gap_count"] == 0
    assert summary["local_lossless_frame_count"] == 3
    assert summary["lossless_only_turn_count"] == 3


def test_role_frames_replace_duplicate_catalog_nodes() -> None:
    _case_value, index, _embedder = _parsed()
    assert validate_index(index) == []
    assert len(index.frames) == 2
    assert not hasattr(index, "operands")
    assert not hasattr(index, "events")
    assert not hasattr(index, "event_frames")
    assert not any(
        edge.relation in {
            "participant", "temporal_scope", "operand_projection",
            "event_frame_member",
        }
        for edge in index.edges
    )
    assert not [
        group for group in index.evidence_groups
        if group.group_kind == "single_fact"
    ]


def test_dialogue_pair_is_complete_and_grounded() -> None:
    _case_value, index, _embedder = _parsed()
    groups = [
        group for group in index.evidence_groups
        if group.group_kind == "dialogue_pair"
    ]
    assert len(groups) == 1
    assert set(groups[0].source_turn_ids) == {
        turn.node_id for turn in index.turns
    }
    assert groups[0].completeness_mask["prompt_turn"]
    assert groups[0].completeness_mask["reply_turn"]
    assert groups[0].completeness_mask["reply_content"]


def test_atomic_packing_keeps_both_dialogue_sources() -> None:
    case, index, embedder = _parsed()
    vectors = embedder.embed(
        [case.question], question_id="q",
        variant="hierarchical_role_graph_v3_6",
    )
    result = retrieve(
        case=case, variant="hierarchical_role_graph_v3_6",
        index=index, query_vectors=vectors, token_budget=2000,
    )
    assert set(result.evidence_leaf_ids) == {
        turn.node_id for turn in index.turns
    }
    assert any(
        row.get("group_kind") == "dialogue_pair"
        for row in result.evidence_ledger
    )
    assert any(
        "structured" in channels
        for channels in result.retrieval_trace["fine_channels"].values()
    )


def test_unpacked_group_does_not_suppress_a_ranked_member_frame() -> None:
    _case_value, index, _embedder = _parsed()
    group = next(
        item for item in index.evidence_groups
        if item.group_kind == "dialogue_pair"
    )
    target = next(
        frame for frame in index.frames
        if frame.frame_kind == "dialogue_answer"
    )
    _context, selected_groups, selected_frames, _sources, _ledger = _pack(
        cards=[], ranked_groups=[(group, 1.0)],
        ranked_frames=[(target, 1.0)],
        turn_by_id={turn.node_id: turn for turn in index.turns},
        token_budget=500,
    )
    assert selected_groups == []
    assert [frame.frame_id for frame in selected_frames] == [target.frame_id]


def test_query_ir_is_topic_invariant() -> None:
    first = build_query_ir("How many cameras did Alice buy?")
    second = build_query_ir("How many orchids did Priya catalog?")
    assert first.requested_value_type == second.requested_value_type == "count"
    assert first.required_roles == second.required_roles


def test_recommendation_context_does_not_become_state_query() -> None:
    ir = build_query_ir(
        "What accessories would you recommend for my current setup?"
    )
    assert ir.requested_value_type == "recommendation"
    assert {"current_state", "context", "source"} <= set(ir.required_roles)
    assert "preference" not in ir.required_roles
    assert "reply_content" not in ir.required_roles


def test_advice_and_selection_language_is_a_generic_recommendation() -> None:
    ir = build_query_ir(
        "I'm excited about the shop visit. Any tips on what to look for "
        "when choosing a new instrument?"
    )
    assert ir.requested_value_type == "recommendation"
    assert {"current_state", "context", "source"} <= set(ir.required_roles)
    assert "i'm" not in ir.target_entities
    assert "tips" not in ir.target_entities


def test_temporal_order_binds_both_explicit_alternatives() -> None:
    ir = build_query_ir(
        "Who did I meet first, the vendor selling preserves at the market "
        "or the visitor from abroad?"
    )
    assert ir.requested_value_type == "temporal_order"
    assert ir.comparison_targets == [
        "vendor selling preserves market", "visitor abroad",
    ]
    assert set(ir.required_roles) == {
        "event_a", "event_b", "time_a", "time_b", "source",
    }


def test_focused_table_row_keeps_column_header() -> None:
    case = _case("What was the rotation for Admon on Sunday?")
    case.haystack_sessions[0][1]["content"] = (
        "| day | 8 am - 4 pm | 4 pm - 12 am |\n"
        "| --- | --- | --- |\n"
        "| Sunday | Admon | Sara |\n"
        "| Monday | Ehab | Admon |"
    )
    turn = build_turn_nodes(case)[1]
    focused = _focused_turn_text(build_query_ir(case.question), turn)
    assert "8 am - 4 pm" in focused
    assert "| Sunday | Admon | Sara |" in focused


def test_counterfactual_query_requests_condition_and_effect_roles() -> None:
    ir = build_query_ir(
        "Would Dana still upgrade the lab if the grant hadn't arrived?"
    )
    assert ir.requested_value_type == "boolean"
    assert ir.target_owner == "dana"
    assert ir.required_roles == ["condition", "effect", "source"]
    assert ir.state_constraints == []


def test_open_preference_query_requests_member_collection_roles() -> None:
    ir = build_query_ir("What do Dana's colleagues enjoy?")
    assert ir.requested_value_type == "preference"
    assert {"owner", "members", "preference", "polarity", "context", "source"} <= set(
        ir.required_roles
    )
    assert ir.collection_constraints == ["distinct", "complete_scope"]


def test_where_query_requests_location_collection_roles() -> None:
    ir = build_query_ir("Where has Melanie camped?")
    assert ir.requested_value_type == "location"
    assert {"scope", "members", "event", "source"} <= set(ir.required_roles)
    scalar = build_query_ir("Where did Dana relocate last year?")
    assert scalar.requested_value_type == "location"
    assert {"event", "location", "source"} <= set(scalar.required_roles)
    assert "members" not in scalar.required_roles
def test_lexical_tokens_preserve_surface_evidence() -> None:
    assert _content_tokens("camped camping camps") == ["camped", "camping", "camps"]
    assert _content_tokens("led leading") == ["led", "leading"]


def test_adaptive_cards_preserves_fused_order_for_counts() -> None:
    ranked = [(f"card:{index}", 1.0 - index / 100.0) for index in range(10)]
    channels = {
        "card:9": {"bm25": 1},
        "card:8": {"exact": 1},
    }
    selected = _adaptive_cards(
        ranked, build_query_ir("How many projects are active?"), channels
    )
    assert selected == [f"card:{index}" for index in range(8)]


def test_adaptive_cards_rescue_replaces_weakest_not_multichannel_tail() -> None:
    ranked = [(f"card:{index}", 1.0 - index / 100.0) for index in range(10)]
    channels = {
        "card:7": {
            "dense": 2, "bm25": 3, "lossless_dense": 3, "lossless_bm25": 4,
        },
        "card:9": {"lossless_exact": 1, "lossless_bm25": 5},
    }
    selected = _adaptive_cards(
        ranked, build_query_ir("How many routes have I used?"), channels
    )
    assert "card:7" in selected
    assert "card:9" in selected
    assert "card:6" not in selected


def test_adaptive_cards_keeps_independently_supported_dense_winner() -> None:
    ranked = [(f"card:{index}", 1.0 - index / 100.0) for index in range(10)]
    channels = {
        "card:9": {"dense": 1, "lossless_dense": 5},
    }
    selected = _adaptive_cards(
        ranked, build_query_ir("Could you give me advice on choosing a tool?"),
        channels,
    )
    assert selected == [
        "card:0", "card:1", "card:2", "card:3",
        "card:4", "card:5", "card:6", "card:9",
    ]


def test_routing_relation_view_is_local_and_entity_bound() -> None:
    service = RoleFrameNode(
        frame_id="q:f:service", question_id="q", session_ids=["s"],
        frame_kind="state", owner_key="dana", entity_key="nebulanet",
        predicate_key="lifesaver", object_key="weekends",
        source_turn_ids=["q:t:0"], retrieval_text="Dana | NebulaNet | lifesaver",
    )
    device = RoleFrameNode(
        frame_id="q:f:device", question_id="q", session_ids=["s"],
        frame_kind="fact", owner_key="dana", entity_key="solder joint",
        predicate_key="repaired", object_key="controller",
        source_turn_ids=["q:t:1"], retrieval_text="Dana | solder joint | repaired",
    )
    card = RoutingCard(
        card_id="q:s:card", question_id="q", session_id="s",
        speaker_keys=["dana"], canonical_entities=["NebulaNet", "solder joint"],
        relations=[
            "Dana relies on NebulaNet on weekends",
            "Dana repaired the controller solder joint",
        ],
        key_events=[], current_states=[], time_range="unknown",
        frame_ids=[service.frame_id, device.frame_id],
        turn_ids=["q:t:0", "q:t:1"], routing_text="local card",
    )
    _attach_routing_relations([service, device], card)
    assert "relies on NebulaNet" in service.retrieval_text
    assert "solder joint" not in service.retrieval_text
    assert "controller solder joint" in device.retrieval_text
    assert "NebulaNet" not in device.retrieval_text


def test_online_retrieval_cannot_read_gold_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src" / "graphmem_demo" / "v36" / "retrieval.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prohibited = {"answer", "answer_session_ids", "question_type"}
    accesses = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "case"
        and node.attr in prohibited
    }
    assert accesses == set()

    pipeline_path = path.parents[1] / "pipeline.py"
    pipeline_tree = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    online = next(
        node for node in ast.walk(pipeline_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_v36_case_with_memory"
    )
    online_accesses = {
        node.attr for node in ast.walk(online)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "case"
        and node.attr in prohibited
    }
    assert online_accesses == set()


def test_build_cache_fingerprint_excludes_retrieval_version() -> None:
    source = inspect.getsource(_memory_cache_fingerprint)
    assert "v36_retrieval_version" not in source


def test_v36_mock_pipeline_is_single_version_and_budgeted(tmp_path: Path) -> None:
    case = _case()
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "out",
        variants=("hierarchical_role_graph_v3_6",),
        question_type="all", max_questions=1, mock_services=True,
        qa_max_tokens=512, build_budget_tokens=300_000,
        answer_budget_tokens=10_000,
    )
    run = run_case(
        config, case, "hierarchical_role_graph_v3_6",
        MockDeepSeekClient(), MockEmbeddingClient(), MockCompressor(1.0),
        InflightLimiter(4),
    )
    assert run.v36_index is not None
    assert run.v3_index is None
    assert run.stats.build_budget_pass and run.stats.answer_budget_pass
    assert run.retrieval.schema_version == "graphmem_v3_6"
    assert run.retrieval.retrieval_trace["answer_target_budget_pass"]
    assert all(record.reasoning_tokens == 0 for record in run.llm_records)


def test_explicit_non_durable_session_does_not_create_noise_frames() -> None:
    case = _case()
    turns = build_turn_nodes(case)
    payload = {
        "frames": [],
        "routing_card": {"entities": [], "relations": [], "events": [], "current_states": [], "time_range": "2026-01-02"},
        "coverage": [[turn.node_id, "boilerplate", []] for turn in turns],
    }
    frames, card, coverage, status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    assert frames == []
    assert card.frame_ids == []
    assert status == "empty_durable_memory"
    assert all(item.coverage_class == "boilerplate" for item in coverage)


def test_compact_object_frame_aliases_are_parsed_losslessly() -> None:
    case = _case()
    turns = build_turn_nodes(case)
    payload = {
        "frames": [{
            "frame_kind": "quantity", "owner": "Alice", "entity": "elapsed activity",
            "relation": "took", "value": "two weeks", "context_key": "activity",
            "polarity": "positive", "modality": "asserted", "status": "completed",
            "op": "complete", "quantity": {"value": 2, "unit": "weeks", "multiplier": 1},
            "semantic_type_keys": ["duration"], "sources": ["T0"], "confidence": 0.9,
        }],
        "routing_card": {},
        "coverage": [["T0", "memory_frame", [0]], ["T1", "dialogue_context", []]],
    }
    frames, _card, _coverage, error = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    assert error is None
    assert frames[0].frame_kind == "quantity"
    assert frames[0].quantity.value == 2
    assert frames[0].quantity.unit == "weeks"
    assert frames[0].source_turn_ids == [turns[0].node_id]


def test_truncated_frames_array_salvages_complete_items() -> None:
    case = _case()
    turns = build_turn_nodes(case)
    frame = ["quantity", "Alice", "activity", "took", "activity", "", "affirmed", "asserted", "completed", "complete", 2, "weeks", None, None, None, None, "exact", "T0", "duration", ["duration"], "T0", 0.9]
    text = "{\"frames\":" + json.dumps([frame])[:-1] + ", [\"fact\""
    frames, _card, coverage, status = parse_session_extraction(
        text, question_id="q", session_id="s", session_date="2026-01-02", turns=turns,
    )
    assert status == "partial_json_salvaged"
    assert len(frames) == 1
    assert frames[0].quantity.value == 2
    assert frames[0].polarity == "positive"
    assert coverage[0].frame_ids == [frames[0].frame_id]


def test_explicit_quantity_fallback_deduplicates_unit_plurality() -> None:
    case = _case()
    case.haystack_sessions[0][0]["content"] = "The activity took two weeks."
    turns = build_turn_nodes(case)
    payload = {
        "frames": [["quantity", "Alice", "activity", "took", "two weeks", "", "positive", "asserted", "completed", "complete", 2, "week", None, None, None, None, "exact", "T0", "duration", ["duration"], "T0", 0.9]],
        "routing_card": {},
        "coverage": [["T0", "memory_frame", [0]], ["T1", "boilerplate", []]],
    }
    frames, _card, _coverage, _status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    matching = [frame for frame in frames if frame.quantity.value == 2]
    assert len(matching) == 1


def test_v36_source_contains_no_benchmark_or_topic_branches() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "graphmem_demo" / "v36"
    text = "\n".join(path.read_text(encoding="utf-8").casefold() for path in root.glob("*.py"))
    assert "longmemeval" not in text
    assert "locomo" not in text
    for topic in ("orchid", "vintage camera", "marvel cinematic", "lgbtq support"):
        assert topic not in text


def test_lossless_quantity_validator_recovers_worded_duration() -> None:
    case = _case()
    case.haystack_sessions[0][0]["content"] = "The activity took a week and a half, then two days."
    turns = build_turn_nodes(case)
    payload = {
        "frames": [], "routing_card": {},
        "coverage": [["T0", "dialogue_context", []], ["T1", "boilerplate", []]],
    }
    frames, _card, coverage, _status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    quantities = {(frame.quantity.value, frame.quantity.unit) for frame in frames}
    assert (1.5, "week") in quantities
    assert (2.0, "days") in quantities
    assert coverage[0].coverage_class == "memory_frame"
    assert len(coverage[0].frame_ids) == 2


def test_wrapped_positional_frame_and_turn_aliases_are_normalized() -> None:
    case = _case()
    turns = build_turn_nodes(case)
    actual = [
        "dialogue_answer", "Bob", "camera", "recommended", "Lumina X2",
        "purchase", "positive", "asserted", "completed", "complete",
        None, "", None, "T1", None, None, "unknown", "T1", "T1",
        ["recommendation"], ["T1"], 0.98,
    ]
    payload = {
        "frames": [["fact", *actual]], "routing_card": {},
        "coverage": [["T0", "dialogue_context", []], ["T1", "memory_frame", [0]]],
    }
    frames, _card, _coverage, status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    assert status is None
    assert len(frames) == 1
    assert frames[0].frame_kind == "dialogue_answer"
    assert frames[0].owner_key == "bob"
    assert frames[0].temporal.event_time is None
    assert frames[0].temporal.anchor_source is None
    assert frames[0].event_identity_key == ""


def test_dialogue_pair_excludes_frames_sourced_outside_the_pair() -> None:
    case = _case()
    case.haystack_sessions[0].append({
        "role": "user", "speaker": "Alice", "listener": "Bob",
        "content": "I will compare that recommendation later.",
    })
    turns = build_turn_nodes(case)
    payload = {
        "frames": [{
            "kind": "dialogue_answer", "owner": "Bob",
            "entity": "camera", "predicate": "recommended",
            "object": "Lumina X2", "sources": ["T1", "T2"],
            "confidence": 0.9,
        }],
        "routing_card": {},
        "coverage": [
            ["T0", "dialogue_context", []],
            ["T1", "memory_frame", [0]],
            ["T2", "memory_frame", [0]],
        ],
    }
    frames, card, coverage, _status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    index = build_index(
        question_id="q", turns=turns, frames=frames,
        routing_cards=[card], coverage=coverage,
    )
    pair = next(
        group for group in index.evidence_groups
        if group.group_kind == "dialogue_pair"
    )
    assert pair.member_frame_ids == []
    assert pair.source_turn_ids == [turns[0].node_id, turns[1].node_id]


def test_plain_alternating_statements_do_not_create_dialogue_pair() -> None:
    case = _case()
    case.haystack_sessions[0][0]["content"] = "I bought a camera yesterday."
    case.haystack_sessions[0][1]["content"] = "That sounds exciting."
    turns = build_turn_nodes(case)
    payload = {
        "frames": [["event", "Alice", "camera", "bought", "camera", "", "positive", "asserted", "completed", "complete", None, "", None, "yesterday", None, None, "relative", "2026-01-02", "camera purchase", ["purchase"], ["T0"], 0.9]],
        "routing_card": {},
        "coverage": [["T0", "memory_frame", [0]], ["T1", "dialogue_context", []]],
    }
    frames, card, coverage, _status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    index = build_index(question_id="q", turns=turns, frames=frames, routing_cards=[card], coverage=coverage)
    assert not [group for group in index.evidence_groups if group.group_kind == "dialogue_pair"]
    assert not [edge for edge in index.edges if edge.relation == "dialogue_pair"]
    assert len([edge for edge in index.edges if edge.relation == "next_turn"]) == 1


def test_query_ir_uses_generic_grammar_not_bare_what_rule() -> None:
    span = build_query_ir("What did Alice say yesterday?")
    listing = build_query_ir("What fields would Caroline pursue?")
    assert span.requested_value_type == "span"
    assert listing.requested_value_type == "list"
    assert listing.target_owner == "caroline"


def test_distinct_measures_survive_semantic_deduplication() -> None:
    case = _case()
    case.haystack_sessions[0][0]["content"] = "The activity covered 22 movies in two weeks."
    turns = build_turn_nodes(case)
    base = ["quantity", "Alice", "activity", "measure", "activity", "", "positive", "asserted", "completed", "complete"]
    payload = {
        "frames": [
            [*base, 22, "movies", None, None, None, None, "exact", None, "", ["count"], ["T0"], 0.9],
            [*base, 2, "weeks", None, None, None, None, "exact", None, "", ["duration"], ["T0"], 0.9],
        ],
        "routing_card": {},
        "coverage": [["T0", "memory_frame", [0, 1]], ["T1", "boilerplate", []]],
    }
    frames, card, coverage, _status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    assert {(frame.quantity.value, frame.quantity.unit) for frame in frames} >= {(22.0, "movies"), (2.0, "weeks")}
    index = build_index(question_id="q", turns=turns, frames=frames, routing_cards=[card], coverage=coverage)
    assert validate_index(index) == []


def test_coverage_indices_remap_to_the_kept_duplicate_frame() -> None:
    case = _case()
    turns = build_turn_nodes(case)
    duplicate = [
        "fact", "Alice", "camera", "owns", "Lumina X2", "purchase",
        "positive", "asserted", "completed", "none", None, "", None,
        None, None, None, "unknown", None, "", ["device"], ["T0"], 0.9,
    ]
    payload = {
        "frames": [duplicate, duplicate],
        "routing_card": {},
        "coverage": [
            ["T0", "memory_frame", [1]],
            ["T1", "dialogue_context", []],
        ],
    }
    frames, card, coverage, status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    assert status is None
    assert len(frames) == 1
    assert coverage[0].frame_ids == [frames[0].frame_id]
    index = build_index(
        question_id="q", turns=turns, frames=frames,
        routing_cards=[card], coverage=coverage,
    )
    assert validate_index(index) == []


def test_coverage_ignores_legacy_indices_and_recovers_durable_turn() -> None:
    case = _case()
    turns = build_turn_nodes(case)
    frame = [
        "fact", "Alice", "camera", "owns", "Lumina X2", "purchase",
        "positive", "asserted", "completed", "none", None, "", None,
        None, None, None, "unknown", None, "", ["device"], ["T0"], 0.9,
    ]
    payload = {
        "frames": [frame],
        "routing_card": {},
        # Legacy indexes are deliberately wrong; sources are authoritative.
        "coverage": [
            ["T0", "memory_frame", ["99"]],
            ["T1", "memory_frame", ["99"]],
        ],
    }
    frames, card, coverage, status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    assert status is None
    assert len(frames) == 2
    assert coverage[0].frame_ids == [frames[0].frame_id]
    assert coverage[1].coverage_class == "memory_frame"
    assert coverage[1].frame_ids == [frames[1].frame_id]
    assert frames[1].semantic_type_keys == ["lossless fallback"]
    assert frames[1].source_turn_ids == [turns[1].node_id]
    assert frames[1].state_op == "none"
    index = build_index(
        question_id="q", turns=turns, frames=frames,
        routing_cards=[card], coverage=coverage,
    )
    assert validate_index(index) == []


def test_empty_rich_extraction_recovers_declared_durable_turns() -> None:
    turns = build_turn_nodes(_case())
    payload = {
        "frames": [],
        "routing_card": {},
        "coverage": [["T0", "dialogue_context"], ["T1", "memory_frame"]],
    }
    frames, card, coverage, status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-02", turns=turns,
    )
    assert status is None
    assert len(frames) == 1
    assert frames[0].source_turn_ids == [turns[1].node_id]
    assert coverage[1].frame_ids == [frames[0].frame_id]
    index = build_index(
        question_id="q", turns=turns, frames=frames,
        routing_cards=[card], coverage=coverage,
    )
    assert validate_index(index) == []


def test_duplicate_source_session_ids_receive_stable_occurrence_ids(
    tmp_path: Path,
) -> None:
    case = _case()
    duplicate = list(case.haystack_sessions[0])
    case.haystack_sessions = [duplicate, duplicate]
    case.haystack_session_ids = ["same", "same"]
    case.haystack_dates = ["2026-01-01", "2026-01-02"]
    turns = build_turn_nodes(case)
    assert {turn.session_id for turn in turns} == {"same", "same__occ2"}
    assert {turn.session_id: turn.session_date for turn in turns} == {
        "same": "2026-01-01", "same__occ2": "2026-01-02",
    }
    assert len({turn.node_id for turn in turns}) == len(turns)
    case.answer_session_ids = ["same"]
    config = DemoConfig(
        data_path=tmp_path / "unused.json", output_dir=tmp_path / "out",
        variants=("hierarchical_role_graph_v3_6",),
        question_type="all", max_questions=1, mock_services=True,
        qa_max_tokens=512, build_budget_tokens=300_000,
        answer_budget_tokens=10_000,
    )
    run = run_case(
        config, case, "hierarchical_role_graph_v3_6",
        MockDeepSeekClient(), MockEmbeddingClient(), MockCompressor(1.0),
        InflightLimiter(4),
    )
    assert run.v36_index is not None
    assert validate_index(run.v36_index) == []
    assert {turn.session_id for turn in run.v36_index.turns} == {
        "same", "same__occ2",
    }


def test_query_ir_supports_generic_numeric_aggregates() -> None:
    average = build_query_ir(
        "What is the average age of me, my parents, and my grandparents?"
    )
    total = build_query_ir("How much did I spend on gifts for my sibling?")
    difference = build_query_ir(
        "How much more did I spend on lodging in Island A compared to City B?"
    )
    assert (average.requested_value_type, average.aggregation_op) == (
        "aggregate", "average",
    )
    assert (total.requested_value_type, total.aggregation_op) == (
        "aggregate", "sum",
    )
    assert (difference.requested_value_type, difference.aggregation_op) == (
        "aggregate", "difference",
    )
    assert difference.operand_targets == ["island", "city"]


def test_query_ir_distinguishes_where_relative_clause_from_location() -> None:
    relative = build_query_ir(
        "In our prior conversation where you wrote two songs, what was the chorus?"
    )
    location = build_query_ir("Where did I leave the package?")
    assert relative.requested_value_type == "span"
    assert location.requested_value_type == "location"


def test_query_ir_supports_relative_entity_and_multi_event_order() -> None:
    relative = build_query_ir("Which book did I finish a week ago?")
    sequence = build_query_ir(
        "What is the order of the events: 'I signed up for rewards', "
        "'I used a coupon', and 'I redeemed cashback'?"
    )
    assert relative.requested_value_type == "entity"
    assert {"entity", "event", "time", "source"} <= set(relative.required_roles)
    assert sequence.requested_value_type == "temporal_order"
    assert sequence.comparison_targets == [
        "signed up rewards", "used coupon", "redeemed cashback",
    ]
    assert {"events", "times", "source"} <= set(sequence.required_roles)


def test_unrelated_collection_cannot_complete_query_roles() -> None:
    ir = build_query_ir("How many different transport services has Dana used?")
    relevant = RoleFrameNode(
        frame_id="q:transport", question_id="q", session_ids=["s1"],
        frame_kind="fact", owner_key="dana", entity_key="metro shuttle",
        predicate_key="used", object_key="weekend transport service",
        source_turn_ids=["q:s1:turn:0"],
        retrieval_text="dana used metro shuttle transport service",
    )
    unrelated = RoleFrameNode(
        frame_id="q:database", question_id="q", session_ids=["s2"],
        frame_kind="fact", owner_key="dana", entity_key="identity graph",
        predicate_key="database", object_key="neptune",
        source_turn_ids=["q:s2:turn:0"],
        retrieval_text="identity graph database neptune",
    )
    group = EvidenceGroup(
        group_id="q:g", question_id="q", group_kind="collection",
        member_frame_ids=[unrelated.frame_id],
        source_turn_ids=unrelated.source_turn_ids,
        required_roles=["scope", "members", "source"],
        completeness_mask={"scope": True, "members": True, "source": True},
        provenance_complete=True, confidence=0.95,
        retrieval_text="database choices identity graph neptune",
        session_ids=["s2"],
    )
    certificate = _certificate(
        ir, [relevant, unrelated], [group],
        routed_sessions={"s1", "s2"}, excluded=[], expansion_rounds=0,
    )
    assert not certificate.complete
    assert {"scope", "members"} <= set(certificate.missing_roles)



def test_v36_memory_cache_stores_embeddings_in_binary_companion(tmp_path: Path) -> None:
    case, index, _embedder = _parsed()
    memory = MemoryBuild(
        leaves=[], summaries=[], roots=[], edges=[], llm_records=[],
        metrics=BuildMetrics(), build_latency_sec=0.0, v36_index=index,
    )
    config = DemoConfig(
        data_path=tmp_path / "unused.json", output_dir=tmp_path / "out",
        variants=("hierarchical_role_graph_v3_6",), mock_services=True,
    )
    path = tmp_path / "memory.json"
    original = {
        node.node_id: list(node.embedding or [])
        for node in [
            *index.turns, *index.frames, *index.routing_cards,
            *index.evidence_groups,
        ]
    }
    _write_memory_cache(
        path, memory, case, "hierarchical_role_graph_v3_6", config,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 5
    assert payload["v36_vector_cache"] == "memory.json.vectors"
    assert '"embedding": [' not in path.read_text(encoding="utf-8")
    assert list((tmp_path / "memory.json.vectors").glob("*.npy"))
    # The live index is restored after the temporary structure-only asdict.
    assert all(
        list(node.embedding or []) == original[node.node_id]
        for node in [
            *index.turns, *index.frames, *index.routing_cards,
            *index.evidence_groups,
        ]
    )
    loaded = _load_memory_cache(path)
    assert loaded is not None and loaded.v36_index is not None
    loaded_nodes = [
        *loaded.v36_index.turns, *loaded.v36_index.frames,
        *loaded.v36_index.routing_cards, *loaded.v36_index.evidence_groups,
    ]
    assert {node.node_id for node in loaded_nodes} == set(original)
    for node in loaded_nodes:
        assert len(node.embedding or []) == len(original[node.node_id])
        assert all(
            abs(actual - expected) < 1e-6
            for actual, expected in zip(node.embedding or [], original[node.node_id])
        )
