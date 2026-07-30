from __future__ import annotations

from dataclasses import replace

from graphmem_demo.v36.dialogue_topology import infer_dialogue_topology
from graphmem_demo.v36.retrieval import build_query_ir
from graphmem_demo.v36.source_spans import build_source_span_closure
from test_v36_role_graph import _case
from graphmem_demo.v36.build import build_turn_nodes


def _turns():
    return build_turn_nodes(_case())


def test_peer_dialogue_treats_both_speakers_as_memory_owners() -> None:
    turns = _turns()
    turns = [
        replace(
            turns[0], node_id="q:alex", session_id="peer",
            speaker="Alex", speaker_key="alex", listener="Blair",
            transport_role="user", text="I bought a telescope last week.",
        ),
        replace(
            turns[1], node_id="q:blair", session_id="peer",
            speaker="Blair", speaker_key="blair", listener="Alex",
            transport_role="assistant", text="I bought a microscope last week.",
        ),
    ]
    assert infer_dialogue_topology(turns).peer_dialogue is True
    closure = build_source_span_closure(
        build_query_ir("What did Blair buy last week?"), turns, {"peer"},
    )
    assert "q:blair" in closure.selected_source_turn_ids


def test_assistant_mediated_keeps_non_dialogue_fact_user_owned() -> None:
    turns = _turns()
    turns = [
        replace(
            turns[0], node_id="q:user", session_id="chat",
            speaker_key="participant 1", listener="",
            transport_role="user", text="I bought a telescope last week.",
        ),
        replace(
            turns[1], node_id="q:assistant", session_id="chat",
            speaker_key="participant 2", listener="",
            transport_role="assistant",
            text="A microscope could also be useful.",
        ),
    ]
    assert infer_dialogue_topology(turns).peer_dialogue is False
    closure = build_source_span_closure(
        build_query_ir("What did I buy last week?"), turns, {"chat"},
    )
    assert "q:user" in closure.selected_source_turn_ids
    assert "q:assistant" not in closure.selected_source_turn_ids
