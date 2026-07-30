from __future__ import annotations

import inspect
from dataclasses import replace

from graphmem_demo.v4 import (
    GRAPHMEM_V4_SCHEMA,
    answer_messages,
    build_capability_view,
    build_query_ir,
    query_views,
    retrieve,
    validate_capability_view,
)
from test_v36_role_graph import _parsed


def test_v4_projects_v2_capabilities_without_duplicate_nodes() -> None:
    _case, index, _embedder = _parsed()
    physical_ids_before = [frame.frame_id for frame in index.frames]
    view = build_capability_view(index)
    assert validate_capability_view(index, view) == []
    assert [frame.frame_id for frame in index.frames] == physical_ids_before
    assert view.diagnostics["duplicated_physical_nodes"] == 0
    assert view.frame_ids_by_capability["dialogue_answer"]
    assert view.source_coverage_complete is True
    assert view.schema_version == GRAPHMEM_V4_SCHEMA


def test_v4_topology_changes_sources_without_benchmark_switch() -> None:
    _case, index, _embedder = _parsed()
    assistant_view = build_capability_view(index)
    assert assistant_view.topology_mode == "peer_dialogue"
    assert {turn.node_id for turn in index.turns} == set(
        assistant_view.memory_source_turn_ids
    )

    mediated_turns = [
        replace(turn, listener="", speaker_key=turn.transport_role)
        for turn in index.turns
    ]
    mediated_index = replace(index, turns=mediated_turns)
    mediated_view = build_capability_view(mediated_index)
    assert mediated_view.topology_mode == "assistant_mediated"
    assert all(
        turn.transport_role == "user"
        for turn in mediated_turns
        if turn.node_id in mediated_view.memory_source_turn_ids
    )


def test_v4_retrieval_records_generic_capability_policy() -> None:
    case, index, embedder = _parsed("Which camera did Bob recommend?")
    view = build_capability_view(index)
    ir = build_query_ir(case.question)
    vectors = embedder.embed(
        query_views(ir), question_id=case.question_id,
        variant="hierarchical_hybrid_graph_v4_0",
    )
    result = retrieve(
        case=case, variant="hierarchical_hybrid_graph_v4_0",
        index=index, capability_view=view, query_vectors=vectors,
        token_budget=8000,
    )
    policy = result.retrieval_trace["v4_capability_policy"]
    assert policy["single_physical_role_graph"] is True
    assert policy["dialogue_navigation_enabled"] is True
    assert result.schema_version == GRAPHMEM_V4_SCHEMA
    prompt = "\n".join(
        message["content"] for message in answer_messages(case, result)
    )
    assert "verify every value against cited lossless source turns" in prompt


def test_v4_source_has_no_benchmark_or_topic_dispatch() -> None:
    import graphmem_demo.v4.build as build_module
    import graphmem_demo.v4.retrieval as retrieval_module

    source = (
        inspect.getsource(build_module) + inspect.getsource(retrieval_module)
    ).casefold()
    banned = (
        "longmemeval", "locomo", "question_type", "answer_session_ids",
        "benchmark_id", "topic_words",
    )
    assert all(token not in source for token in banned)
