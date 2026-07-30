from __future__ import annotations

from graphmem_demo.v3.binding_hints import missing_count_target_hint
from graphmem_demo.v3.retrieval import _node_text, _tokens, build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(text: str) -> TurnNode:
    return TurnNode(
        node_id="turn",
        question_id="q",
        session_id="s",
        session_date="2026-01-01",
        turn_index=0,
        speaker="A",
        speaker_key="a",
        listener="B",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_missing_exact_count_target_rejects_sibling_category() -> None:
    frame = build_query_frame("How many signed notebooks did I collect?")
    hint = missing_count_target_hint(
        frame,
        [("turn", _turn("I collected 15 signed tablets."), 1.0, "test")],
        node_text=_node_text,
        tokenize=_tokens,
    )
    assert hint is not None
    assert hint["operation"] == "missing_count_target"
    assert hint["missing_heads"] == ["notebook"]


def test_present_count_target_does_not_force_abstention() -> None:
    frame = build_query_frame("How many signed notebooks did I collect?")
    hint = missing_count_target_hint(
        frame,
        [("turn", _turn("I collected 4 signed notebooks."), 1.0, "test")],
        node_text=_node_text,
        tokenize=_tokens,
    )
    assert hint is None


def test_generic_count_head_is_not_treated_as_missing_entity() -> None:
    frame = build_query_frame("How many items did I receive?")
    hint = missing_count_target_hint(
        frame,
        [("turn", _turn("A package arrived yesterday."), 1.0, "test")],
        node_text=_node_text,
        tokenize=_tokens,
    )
    assert hint is None


def test_any_present_alternative_makes_or_count_answerable() -> None:
    frame = build_query_frame("How many albums or EPs have I downloaded?")
    hint = missing_count_target_hint(
        frame,
        [("turn", _turn("I downloaded an album yesterday."), 1.0, "test")],
        node_text=_node_text,
        tokenize=_tokens,
    )
    assert hint is None
