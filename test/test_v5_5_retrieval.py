from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller
from graphmem.domain import FactBinding, QueryBudget, QueryOperator
from graphmem.retrieval import GraphNavigator, HarnessProfile
from graphmem.retrieval.algebra import evaluate
from graphmem.retrieval.query_ir import compile_query
from graphmem.runtime import GraphReadView
from graphmem.storage import SQLiteGraphStore

from test_v5_1_llm_graph import StrictCompletions, _strict_config
from test_v5_gate_b_core import _store


def _semantic_store(path: Path) -> SQLiteGraphStore:
    store = _store(path); config = _strict_config(); completions = StrictCompletions()
    distiller = QwenSemanticDistiller(
        store, config, "v5.5-test", client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    GraphBuildPipeline(store, dataset_hash="v5.5-test", distiller=distiller).build("travel", config)
    return store


def test_query_read_view_compiles_fact_and_routing_postings(tmp_path: Path) -> None:
    store = _semantic_store(tmp_path / "graph.sqlite")
    view = GraphReadView(store.nodes("travel"), store.edges("travel"))
    assert "alice" in view.owner_alias_index
    alice = view.owner_alias_index["alice"]
    facts = view.lookup_facts(owner_ids=alice, predicates=("visit",))
    assert facts
    assert view.route_children(("alice",))


def test_query_compiler_resolves_possessive_owner_alias(tmp_path: Path) -> None:
    store = _semantic_store(tmp_path / "graph.sqlite")
    view = GraphReadView(store.nodes("travel"), store.edges("travel"))
    ir = compile_query("What did Alice's travel include?", view)
    assert ir.operands[0].owner_aliases == ("alice",)


def test_h6_packs_terminal_proof_units_and_stops_by_certificate(tmp_path: Path) -> None:
    store = _semantic_store(tmp_path / "graph.sqlite")
    result = GraphNavigator(store, harness_profile=HarnessProfile.H6_PROOF_PACKING).navigate(
        "travel", "What did Alice travel to?", QueryBudget(max_evidence_turns=32)
    )
    assert result.retrieved_turn_ids
    assert result.proof_units
    assert result.certificate and result.certificate.complete
    assert result.stop_reason is not None
    assert all(unit.source_turn_ids for unit in result.proof_units)


def test_intersection_requires_a_witness_from_each_operand() -> None:
    rows = (
        FactBinding("a", "left", "f1", "o1", "volunteer", "", "shelter", None, None, ("g1",)),
        FactBinding("b", "right", "f2", "o2", "volunteer", "", "shelter", None, None, ("g2",)),
        FactBinding("c", "left", "f3", "o1", "volunteer", "", "museum", None, None, ("g3",)),
    )
    result = evaluate(QueryOperator.INTERSECTION_DISTINCT, rows, ("left", "right"))
    assert set(result.output_binding_ids) == {"a", "b"}
    assert not result.collection_complete


def test_read_only_store_refuses_mutation(tmp_path: Path) -> None:
    writable = _store(tmp_path / "graph.sqlite"); writable.close()
    readonly = SQLiteGraphStore(tmp_path / "graph.sqlite", read_only=True)
    with pytest.raises(RuntimeError, match="read-only"):
        with readonly.transaction():
            pass
    assert readonly.turns("travel")
    readonly.close()
