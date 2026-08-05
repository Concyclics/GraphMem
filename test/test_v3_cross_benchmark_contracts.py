from __future__ import annotations

import ast
from pathlib import Path

from graphmem_demo.v3.catalog_arithmetic import arithmetic_hint
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.compact_packing import _focused_text
from graphmem_demo.v3.retrieval import (
    _evidence_time,
    _node_text,
    _tokens,
    build_query_frame,
)
from graphmem_demo.v3.schema import ClaimNode, TurnNode
from graphmem_demo.v3.state_temporal_operators import relative_time_hint


def test_v3_retrieval_cannot_read_benchmark_gold_or_type_fields() -> None:
    """Keep benchmark metadata outside the online retrieval decision boundary."""
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "graphmem_demo"
        / "v3"
        / "retrieval.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    prohibited = {"answer", "answer_session_ids", "question_type"}
    accesses = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "case"
        and node.attr in prohibited
    }
    assert accesses == set()


def _operand(
    operand_id: str,
    predicate: str,
    obj: str,
    *,
    observed_at: str,
    quantity: float | None = None,
    unit: str = "",
    context_key: str = "",
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="participant",
        predicate_key=predicate,
        object_key=obj,
        object_text=obj,
        observed_at=observed_at,
        quantity=quantity,
        unit=unit,
        context_key=context_key,
        source_turn_ids=[f"turn:{operand_id}"],
        retrieval_text=f"participant | {predicate} | {obj} | {context_key}",
    )


def test_query_contracts_are_grammatical_not_topic_specific() -> None:
    assert build_query_frame("How much did I spend on gifts?").requested_operation == "count"
    assert build_query_frame(
        "How many days before I bought my tablet did I visit the market?"
    ).requested_operation == "duration"
    assert build_query_frame(
        "In our conversation where you wrote a poem, what was its title?"
    ).requested_operation == "lookup"
    assert build_query_frame(
        "I want to rearrange my room. Any tips?"
    ).requested_operation == "recommendation"
    total = build_query_frame(
        "What is the total weight of the mineral samples I collected?"
    )
    assert total.requested_operation == "count"
    assert total.answer_form == "number"


def test_latest_scalar_snapshot_precedes_lexical_similarity() -> None:
    frame = build_query_frame("How many tops have I bought so far?")
    hint = arithmetic_hint(
        frame,
        [
            _operand(
                "old", "bought three tops", "already",
                observed_at="2023-08-11", quantity=3, unit="tops",
            ),
            _operand(
                "new", "has", "five tops",
                observed_at="2023-09-30", quantity=5, unit="tops",
            ),
        ],
        [],
        lambda query, text: len(set(query.content_terms) & set(_tokens(text))),
    )
    assert hint is not None
    assert hint["operation"] == "scalar_snapshot"
    assert hint["value"] == 5


def test_scalar_snapshot_applies_only_bound_later_deltas() -> None:
    frame = build_query_frame("How many pre-1920 American coins do I have?")
    hint = arithmetic_hint(
        frame,
        [
            _operand(
                "base", "quantity", "pre-1920 American coins",
                observed_at="2023-05-27", quantity=37, unit="coins",
            ),
            _operand(
                "valid", "added", "1915 quarter",
                context_key="pre-1920 American coins",
                observed_at="2023-05-29",
            ),
            _operand(
                "sibling", "bought", "1972 error coin",
                context_key="error coins",
                observed_at="2023-05-29",
            ),
        ],
        [],
        lambda query, text: len(set(query.content_terms) & set(_tokens(text))),
    )
    assert hint is not None
    assert hint["value"] == 38
    assert hint["delta_operand_ids"] == ["valid"]


def test_scalar_snapshot_rejects_sibling_category_missing_modifier() -> None:
    frame = build_query_frame("How many alpine cabins have I inspected?")
    hint = arithmetic_hint(
        frame,
        [
            _operand(
                "sibling", "inspected", "four coastal cabins",
                observed_at="2023-05-29", quantity=4, unit="cabins",
            ),
        ],
        [],
        lambda query, text: len(set(query.content_terms) & set(_tokens(text))),
    )
    assert hint is None


def test_relative_time_prefers_fine_fact_over_noisy_turn_on_same_day() -> None:
    frame = build_query_frame(
        "Where was the art event I attended two weeks ago held?"
    )
    noisy = TurnNode(
        node_id="noisy",
        question_id="q",
        session_id="s0",
        session_date="2023-01-15",
        turn_index=0,
        speaker="A",
        speaker_key="a",
        listener="B",
        transport_role="user",
        text="A long unrelated document mentioning art and events.",
        retrieval_text="A long unrelated document mentioning art and events.",
    )
    fact = ClaimNode(
        node_id="fact",
        question_id="q",
        session_id="s1",
        subject="participant",
        subject_key="participant",
        predicate="attended",
        predicate_key="attend",
        object="Ancient Civilizations at Metropolitan Museum",
        object_key="ancient civilizations metropolitan museum",
        event_time="2023-01-15",
        observed_at="2023-01-15",
        retrieval_text="participant attended art exhibit at Metropolitan Museum",
    )
    hint = relative_time_hint(
        frame,
        [
            ("turn", noisy, 2.0, "test"),
            ("claim", fact, 1.0, "test"),
        ],
        "2023-02-01",
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
    )
    assert hint is not None
    assert hint["supporting_node_ids"] == ["fact"]


def test_focused_text_keeps_structured_value_after_heading() -> None:
    frame = build_query_frame(
        "What was the progression for the chorus in the second result?"
    )
    text = (
        "Background sentence. " * 100
        + "\nChorus:\nC D E F G A B A G F E D C\n"
        + "A final explanatory sentence."
    )
    focused = _focused_text(text, frame, 1200)
    assert "Chorus:" in focused
    assert "C D E F G A B A G F E D C" in focused
