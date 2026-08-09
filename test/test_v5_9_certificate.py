from __future__ import annotations

from graphmem.domain import (
    FactBinding,
    OperandSpec,
    ProofObligation,
    QueryOperator,
)
from graphmem.retrieval import operators as ops
from graphmem.retrieval.ast_algebra import evaluate_ast
from graphmem.retrieval.certificate import finalize_ast_certificate
from graphmem.retrieval.packer import build_proof_units
from graphmem.retrieval.query_ir import QueryIR


def _ir() -> QueryIR:
    operand = OperandSpec("o1", multiplicity="exhaustive_set")
    ast = ops.CountDistinct(ops.FactSet("o1"))
    obligations = (
        ProofObligation("need-binding", "o1", "binding"),
        ProofObligation("need-provenance", "o1", "provenance"),
        ProofObligation("need-collection", "o1", "collection"),
    )
    return QueryIR(
        "How many hats?", QueryOperator.COUNT_DISTINCT, (operand,), (),
        ast=ast, ast_operands=(operand,), ast_obligations=obligations,
    )


def _binding() -> FactBinding:
    return FactBinding(
        binding_id="b1", operand_id="o1", fact_node_id="f1",
        owner_id="alice", predicate="own", scope="hats", value_key="fedora",
        event_instance_id=None, time_interval=None, evidence_group_ids=("g1",),
        confidence=1.0, value="Fedora", session_id="s1", turn_index=1,
    )


def test_post_pack_certificate_closes_only_when_an_atomic_witness_survives() -> None:
    ir = _ir()
    binding = _binding()
    algebra = evaluate_ast(ir.ast, (binding,), collection_closed={"o1": True})
    group_turns = {"g1": ("t1", "t2")}
    units = build_proof_units((binding,), group_turns)

    partial = finalize_ast_certificate(
        ir, algebra, (binding,), group_turns, ("t1",), units=units)
    complete = finalize_ast_certificate(
        ir, algebra, (binding,), group_turns, ("t1", "t2"), units=units)

    assert partial.complete and partial.false_complete
    assert not partial.post_pack_complete
    assert partial.unsatisfied_phase == "witness_packed"
    assert partial.dropped_mandatory_unit_ids == (units[0].unit_id,)
    assert complete.post_pack_complete and not complete.false_complete
    assert complete.phase_status["post_pack_complete"]


def test_unclosed_collection_cannot_be_certified_even_when_the_turn_is_packed() -> None:
    ir = _ir()
    binding = _binding()
    algebra = evaluate_ast(ir.ast, (binding,), collection_closed={})

    certificate = finalize_ast_certificate(
        ir, algebra, (binding,), {"g1": ("t1",)}, ("t1",))

    assert not certificate.complete
    assert not certificate.post_pack_complete
    assert certificate.unsatisfied_phase == "pre_pack_closure"
    assert "need-collection" in certificate.missing_slots
