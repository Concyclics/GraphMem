from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pytest

from graphmem.domain import QueryOperator
from graphmem.retrieval.operators import (
    ArgMaxTime,
    ArgMinTime,
    CountDistinct,
    DateDifference,
    ExistsAll,
    GroupByOwner,
    IntersectionDistinct,
    LatestState,
    Lookup,
    Ordinal,
    UnionDistinct,
    describe,
    operand_ids,
    requires_exhaustive_scope,
    root_operator,
)
from graphmem.retrieval.query_ir import compile_query
from graphmem.retrieval.slots import parse_slots


@dataclass
class _View:
    """Minimal read view: owner aliases and a predicate vocabulary."""

    owner_alias_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    predicate_index: tuple[str, ...] = ()
    scope_fact_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    nodes: Mapping[str, object] = field(default_factory=dict)


def _view(*owners: str, predicates: tuple[str, ...] = ("visit", "volunteer", "read", "live"),
          scopes: tuple[str, ...] = ("travel", "reading")) -> _View:
    return _View({name: (f"entity:{name}",) for name in owners}, predicates,
                 {name: () for name in scopes})


def _ast(query: str, *owners: str):
    return compile_query(query, _view(*owners)).ast


def _shape(node) -> str:
    """Structure only, with operand ids replaced by their position."""
    order = {value: index for index, value in enumerate(operand_ids(node))}
    text = describe(node)
    for operand_id, index in order.items():
        text = text.replace(operand_id, f"#{index}")
    return text


# --- composition ---------------------------------------------------------------

def test_count_over_intersection_keeps_both_halves() -> None:
    """The case V5.5 could not express: it kept only COUNT_DISTINCT."""
    node = _ast("How many places have Alice and Bob both visited?", "alice", "bob")

    assert isinstance(node, CountDistinct)
    assert isinstance(node.child, IntersectionDistinct)
    assert _shape(node) == "CountDistinct(IntersectionDistinct(FactSet(#0),FactSet(#1)))"
    assert len(operand_ids(node)) == 2


def test_count_by_occurrence_uses_event_instances() -> None:
    node = _ast("How many times did Alice visit Paris?", "alice")

    assert isinstance(node, CountDistinct)
    assert node.distinct_by == "event_instance"


def test_each_becomes_group_by_owner() -> None:
    node = _ast("What are Alice and Bob each reading?", "alice", "bob")

    assert isinstance(node, GroupByOwner)
    assert len(operand_ids(node)) == 2


def test_both_with_auxiliary_lead_becomes_exists_all() -> None:
    """V5.5 could never reach EXISTS_ALL here: 'both' won the race first."""
    node = _ast("Do both Alice and Bob have pets?", "alice", "bob")

    assert isinstance(node, ExistsAll)
    assert len(operand_ids(node)) == 2


def test_bare_quantifier_without_auxiliary_is_an_intersection() -> None:
    node = _ast("Which cities have Alice and Bob both visited?", "alice", "bob")

    assert isinstance(node, IntersectionDistinct)


def test_ordinal_carries_its_index() -> None:
    """V5.5 parsed the ordinal word but never carried k, so 'second' == 'first'."""
    second = _ast("What was the second book Alice read?", "alice")
    third = _ast("What was the third book Alice read?", "alice")

    assert isinstance(second, Ordinal) and second.index == 2
    assert isinstance(third, Ordinal) and third.index == 3


def test_last_is_a_descending_ordinal_not_a_state_lookup() -> None:
    node = _ast("What was the last book Alice read?", "alice")

    assert isinstance(node, Ordinal)
    assert node.index == -1 and node.order == "descending"


def test_duration_between_two_events_is_a_date_difference() -> None:
    node = _ast("How long between Alice moving and Bob moving?", "alice", "bob")

    assert isinstance(node, DateDifference)
    assert len(operand_ids(node)) == 2


def test_how_long_does_not_become_a_count() -> None:
    """Both open with 'how'; only one of them is a count."""
    assert parse_slots("How long did Alice stay?").is_count is False
    assert parse_slots("How many books did Alice read?").is_count is True


def test_current_state_query_is_latest_state() -> None:
    node = _ast("Where does Alice currently live?", "alice")

    assert isinstance(node, LatestState)


def test_now_inside_know_does_not_trigger_latest_state() -> None:
    """The substring test in V5.5 matched 'now' inside 'know'."""
    assert parse_slots("Does Alice know Bob?").is_latest is False
    assert parse_slots("Where does Alice live now?").is_latest is True


def test_indirect_question_frame_is_not_an_existence_check() -> None:
    """'Do you know where X lives' asks for a place, not for whether I know."""
    node = _ast("Do you know where Alice lives?", "alice")

    assert not isinstance(node, ExistsAll)
    assert parse_slots("Do you know where Alice lives?").indirect is True


def test_before_and_after_pick_opposite_temporal_ends() -> None:
    assert isinstance(_ast("What did Alice do before the trip?", "alice"), ArgMinTime)
    assert isinstance(_ast("What did Alice do after the trip?", "alice"), ArgMaxTime)


def test_plural_head_makes_a_single_owner_question_a_set_question() -> None:
    node = _ast("Which cities did Alice visit?", "alice")

    assert isinstance(node, UnionDistinct)
    assert requires_exhaustive_scope(node)


def test_singular_lookup_stays_a_lookup() -> None:
    node = _ast("What is Alice's job?", "alice")

    assert isinstance(node, Lookup)
    assert not requires_exhaustive_scope(node)


def test_possessive_owner_alias_resolves() -> None:
    ir = compile_query("What did Alice's travel include?", _view("alice"))

    assert ir.ast is not None
    assert ir.ast_operands[0].owner_aliases == ("alice",)
    assert parse_slots("What did Alice's travel include?").possessive is True


def test_question_without_any_owner_still_compiles() -> None:
    node = _ast("What is the capital of France?")

    assert isinstance(node, Lookup)
    assert len(operand_ids(node)) == 1


def test_operator_divergence_lowers_compile_confidence_and_softens_filters() -> None:
    compiled = compile_query(
        "Did both Alice and Bob have pets?",
        _view("alice", "bob", predicates=()))
    promoted = compiled.promote_ast()

    softened = promoted.soften_with_legacy(compiled)

    assert compiled.ast_diverges
    assert compiled.compile_confidence < 0.80
    assert "legacy_ast_operator_divergence" in compiled.fallback_reasons
    assert softened.soft_fallback_applied
    assert len(softened.operands) == len(promoted.operands)


def test_negation_is_recorded_on_the_operands() -> None:
    ir = compile_query("What did Alice not visit?", _view("alice"))

    assert parse_slots("What did Alice not visit?").negation is True
    assert all(item.polarity == "negative" for item in ir.ast_operands)


# --- obligations and invariants -------------------------------------------------

def test_set_operators_carry_a_collection_obligation() -> None:
    ir = compile_query("How many places have Alice and Bob both visited?", _view("alice", "bob"))

    kinds = {item.kind for item in ir.ast_obligations}
    assert "collection" in kinds and "binding" in kinds and "provenance" in kinds


def test_date_difference_carries_a_time_endpoint_obligation() -> None:
    ir = compile_query("How long between Alice moving and Bob moving?", _view("alice", "bob"))

    assert "time_endpoint" in {item.kind for item in ir.ast_obligations}


def test_latest_state_carries_a_state_history_obligation() -> None:
    ir = compile_query("Where does Alice currently live?", _view("alice"))

    assert "state_history" in {item.kind for item in ir.ast_obligations}


def test_compilation_is_deterministic() -> None:
    view = _view("alice", "bob")
    query = "How many places have Alice and Bob both visited?"

    assert describe(compile_query(query, view).ast) == describe(compile_query(query, view).ast)


@pytest.mark.parametrize("query,owners", [
    ("How many places have Alice and Bob both visited?", ("alice", "bob")),
    ("Do both Alice and Bob have pets?", ("alice", "bob")),
    ("What was the second book Alice read?", ("alice",)),
    ("Where does Alice currently live?", ("alice",)),
    ("What is the capital of France?", ()),
])
def test_legacy_operator_is_untouched_by_the_shadow_compiler(query, owners) -> None:
    """PR2b must not change execution: ``operator`` stays the V5.5 decision."""
    from graphmem.retrieval.query_ir import _operator

    ir = compile_query(query, _view(*owners))

    assert ir.operator == _operator(query)
    assert ir.operands and all(not item.operand_id.startswith("ast-operand")
                               for item in ir.operands)


def test_divergence_between_legacy_and_ast_is_reported() -> None:
    """The count-over-intersection case is exactly where the two disagree."""
    ir = compile_query("Do both Alice and Bob have pets?", _view("alice", "bob"))

    assert ir.operator == QueryOperator.INTERSECTION_DISTINCT
    assert ir.ast_operator == QueryOperator.EXISTS_ALL
    assert ir.ast_diverges is True
    assert root_operator(ir.ast) == QueryOperator.EXISTS_ALL


def test_promoted_ast_is_the_single_downstream_contract() -> None:
    compiled = compile_query(
        "Do both Alice and Bob have pets?", _view("alice", "bob"))
    active = compiled.promote_ast()

    assert compiled.ast_diverges is True
    assert active.operator == QueryOperator.EXISTS_ALL
    assert active.operands == compiled.ast_operands
    assert active.proof_obligations == compiled.ast_obligations
    assert {row.operand_id for row in active.proof_obligations if row.operand_id} <= {
        row.operand_id for row in active.operands}
