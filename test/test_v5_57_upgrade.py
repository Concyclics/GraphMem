from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from graphmem.config import config_hash, load_config, load_runtime_config
from graphmem.domain import (
    CandidateScore,
    EvidenceCertificate,
    FactBinding,
    GraphNode,
    NodeType,
    OperandSpec,
    QueryOperator,
    SourceTurn,
    TruthValue,
    stable_id,
)
from graphmem.projection.config import ARMS
from graphmem.projection.manifest import build_manifests
from graphmem.answer import aggregation_operation
from graphmem.answer.stage import _preference_focus_index
from graphmem.retrieval import operators as ops
from graphmem.retrieval.ast_algebra import evaluate_ast
from graphmem.retrieval.executor import inspect_execution
from graphmem.retrieval.query_ir import compose_operator
from graphmem.retrieval.slots import parse_slots
from graphmem.runtime import GraphReadView


ROOT = Path(__file__).resolve().parents[1]


def _binding(operand: str, value: str, turn: int) -> FactBinding:
    return FactBinding(
        binding_id=f"b:{operand}:{turn}", operand_id=operand,
        fact_node_id=f"f:{turn}", owner_id="alice", predicate="spent",
        scope="travel expenses", value_key=value.casefold(),
        event_instance_id=f"event:{turn}", time_interval=None,
        evidence_group_ids=(f"g:{turn}",), confidence=1.0,
        value=value, value_type="currency", turn_index=turn,
    )


def _fact(value: str, scene_id: str, turn: int) -> GraphNode:
    return GraphNode(
        stable_id("node", "m", "fact", turn), "m", NodeType.CANONICAL_FACT,
        0, f"Alice bought {value}", f"g:{turn}", attributes={
            "owner_id": "owner:alice", "predicate": "bought",
            "scope": "model kits", "collection_key": "model kits",
            "polarity": "positive", "modality": "asserted",
            "value": value, "value_key": value.casefold(),
            "scene_id": scene_id, "session_id": "s1", "turn_index": turn,
        })


def _scene(scene_id: str, *, covered: int, total: int, missing: int = 0,
           unresolved: int = 0) -> GraphNode:
    return GraphNode(
        scene_id, "m", NodeType.SCENE, 0, "model kit discussion", "g:scene",
        attributes={
            "information_unit_total": total,
            "information_unit_covered": covered,
            "information_unit_missing": missing,
            "information_unit_unresolved": unresolved,
        })


def test_v557_profiles_enable_fresh_projection_and_guarded_priority() -> None:
    build = load_config(ROOT / "configs/v5/v5_57_lossless_atomic.json")
    runtime = load_runtime_config(ROOT / "configs/v5/runtime_v5_57_accuracy64.json")
    runtime32 = load_runtime_config(ROOT / "configs/v5/runtime_v5_57_accuracy32.json")

    assert build.projection_profile == "R3"
    assert build.models.semantic_adaptive_fact_cap
    assert build.models.semantic_budget_degrade_at == 1.0
    assert "entity" not in build.models.semantic_atomic_unit_kinds
    assert build.edges.relation_mask_propagation
    assert build.edges.atomic_relation_multiview
    assert runtime.retrieval.exact_lookup_priority
    assert not runtime.retrieval.exact_lookup_fast_path
    assert runtime.retrieval.obligation_aware_packing
    assert runtime.retrieval.fusion_weights["advice_dense"] == 2.0
    assert runtime.retrieval.fusion_weights["advice_session"] == 0.05
    assert runtime32.retrieval.fusion_weights == runtime.retrieval.fusion_weights
    assert runtime.query_budget.max_evidence_turns == 64
    assert runtime32.query_budget.max_evidence_turns == 32
    assert replace(runtime32.query_budget, max_evidence_turns=64) == runtime.query_budget


def test_projection_profile_is_part_of_graph_identity() -> None:
    config = load_config(ROOT / "configs/v5/v5_57_lossless_atomic.json")

    assert config_hash(config) != config_hash(replace(config, projection_profile="P0"))


def test_sum_query_compiles_and_executes_only_over_closed_scope() -> None:
    slots = parse_slots("How much did Alice spend in total?")
    operand = OperandSpec("o1", multiplicity="exhaustive_set")
    ast = compose_operator(slots, (operand,))

    assert slots.is_sum
    assert isinstance(ast, ops.Sum)
    assert ops.root_operator(ast) == QueryOperator.SUM

    rows = (_binding("o1", "$12", 0), _binding("o1", "$30", 1))
    open_result = evaluate_ast(ast, rows, collection_closed={})
    closed_result = evaluate_ast(ast, rows, collection_closed={"o1": True})
    certificate = EvidenceCertificate(
        "sum", (), (), (), True, 1, post_pack_complete=True)

    assert open_result.numeric_total == 42 and not open_result.scope_complete
    assert closed_result.numeric_total == 42 and closed_result.unit == "USD"
    decision = inspect_execution(closed_result, certificate)
    assert decision is not None and decision.safe_to_bypass
    assert decision.text == "42 USD"


def test_per_item_price_is_a_quotient_not_a_sum() -> None:
    slots = parse_slots(
        "How much did I spend on each coffee mug for my coworkers?")
    operand = OperandSpec("o1", multiplicity="exhaustive_set")
    ast = compose_operator(slots, (operand,))

    assert slots.is_unit_rate and not slots.is_sum
    assert isinstance(ast, ops.Lookup)
    assert aggregation_operation(
        "How much did I spend on each coffee mug for my coworkers?") == "unit_rate"


def test_need_item_count_is_not_miscompiled_as_difference() -> None:
    assert aggregation_operation(
        "How many items of clothing do I need to pick up or return from a store?"
    ) == "count_distinct"
    assert aggregation_operation(
        "How many more points do I need to earn to reach my goal?"
    ) == "difference"


def test_preference_focus_quotes_only_packed_personal_facts() -> None:
    def turn(turn_id: str, text: str, role: str = "user") -> SourceTurn:
        return SourceTurn(
            turn_id, "m", "s", 0, role, "", role, None, text, turn_id)

    relevant = turn(
        "t-power",
        "I'm looking for travel advice for my new portable power bank and pad.")
    generic = turn("t-generic", "Can you give me general battery tips?")
    dropped = turn("t-drop", "I have a second battery that was not packed.")
    scores = (
        CandidateScore("t-generic", "s", 0, 0, 1.0, 0, 0, 0, 1, 2.0,
                       ("dense",)),
        CandidateScore("t-power", "s", 0, 0, 0.8, 0, 0, 0, 1, 1.7,
                       ("dense",)),
        CandidateScore("t-drop", "s", 0, 0, 1.0, 0, 0, 0, 1, 2.0,
                       ("dense",)),
    )

    text, ids = _preference_focus_index(
        "What should I use for my phone battery?",
        {row.turn_id: row for row in (relevant, generic, dropped)},
        ("t-generic", "t-power"), scores)

    assert ids == ("t-power",)
    assert text is not None and "portable power bank" in text
    assert "second battery" not in text


def test_existence_miss_is_unknown_until_scope_is_closed() -> None:
    ast = ops.ExistsAll((ops.FactSet("o1"),))

    unknown = evaluate_ast(ast, (), collection_closed={})
    negative = evaluate_ast(ast, (), collection_closed={"o1": True})

    assert unknown.truth_value == TruthValue.UNKNOWN
    assert not unknown.scope_complete
    assert negative.truth_value == TruthValue.FALSE
    assert negative.scope_complete
    decision = inspect_execution(unknown)
    assert decision is not None and "open_world_unknown" in decision.reason_codes
    assert not decision.safe_to_bypass


def test_r3_manifest_requires_atomic_scene_closure_and_keeps_owner_label() -> None:
    scene_id = "scene:1"
    owner = GraphNode(
        "owner:alice", "m", NodeType.CANONICAL_ENTITY, 0, "Alice", "g:owner")
    facts = (_fact("B-29", scene_id, 0), _fact("F-15", scene_id, 1))

    complete_nodes, _edges, _rows = build_manifests(
        "m", (owner, _scene(scene_id, covered=3, total=3), *facts), ARMS["R3"])
    incomplete_nodes, _edges, _rows = build_manifests(
        "m", (owner, _scene(scene_id, covered=2, total=3, missing=1), *facts),
        ARMS["R3"])

    assert complete_nodes[0].attributes["closed"] is True
    assert incomplete_nodes[0].attributes["closed"] is False
    assert complete_nodes[0].attributes["owner_label"] == "Alice"
    assert complete_nodes[0].summary.startswith("Alice")


def test_routing_atoms_are_searchable_without_expanding_display_summary() -> None:
    node = GraphNode(
        "ref:1", "m", NodeType.EVIDENCE_GROUP_REF, 0, "Alice trip notes",
        "g:1", attributes={
            "roles": ("evidence_turn", "terminal"),
            "routing_atoms": ("$317.50", "September 14"),
        })

    view = GraphReadView((node,), ())

    assert view.node_term_index["317"] == (node.node_id,)
    assert view.node_term_index["september"] == (node.node_id,)
