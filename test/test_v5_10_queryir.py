from __future__ import annotations

from graphmem.domain import (
    AlgebraResult,
    EvidenceCertificate,
    GraphEdge,
    GraphNode,
    NodeType,
    OperandSpec,
    ProofObligation,
    QueryBudget,
    QueryOperator,
    RelationType,
    TemporalEndpoint,
    TemporalKey,
)
from graphmem.retrieval.executor import inspect_execution
from graphmem.retrieval.query_ir import QueryIR
from graphmem.retrieval.scheduler import execute
from graphmem.runtime import GraphReadView


def _node(node_id: str, summary: str) -> GraphNode:
    return GraphNode(
        node_id, "m", NodeType.CANONICAL_FACT, 0, summary, f"g:{node_id}")


def _edge(edge_id: str, relation: RelationType, target: str,
          source: str = "test") -> GraphEdge:
    return GraphEdge(
        edge_id, "m", "seed", relation, target, "g:seed",
        True, 0.9, source)


def test_obligation_aware_scheduler_prioritizes_state_relation_in_beam() -> None:
    view = GraphReadView(
        (_node("seed", "Alice"), _node("lexical", "currently"),
         _node("state", "new address")),
        (_edge("e-lexical", RelationType.COARSE_RELATED, "lexical"),
         _edge("e-state", RelationType.CONTRADICTION_UPDATE, "state")))
    operand = OperandSpec("o1", owner_aliases=("alice",))
    ir = QueryIR(
        "Where does Alice currently live?", QueryOperator.LATEST_STATE,
        (operand,), (ProofObligation("need-history", None, "state_history"),))
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=2, max_visited_edges=1,
        max_frontier=4, max_seed_nodes=1)
    common = dict(
        structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,
                             RelationType.CONTRADICTION_UPDATE))

    lexical = execute(view, ir, ("seed",), budget, **common)
    directed = execute(
        view, ir, ("seed",), budget,
        obligation_aware_relations=True, **common)

    assert lexical.proof[0].relation == RelationType.COARSE_RELATED
    assert directed.proof[0].relation == RelationType.CONTRADICTION_UPDATE


def test_relation_mask_metadata_prioritizes_matching_coarse_edge() -> None:
    view = GraphReadView(
        (_node("seed", "Alice"), _node("lexical", "currently"),
         _node("masked", "address history")),
        (_edge("e-lexical", RelationType.COARSE_RELATED, "lexical"),
         _edge("e-masked", RelationType.COARSE_RELATED, "masked",
               "relation_mask:scene_similar,state_compatible")))
    ir = QueryIR(
        "Where does Alice currently live?", QueryOperator.LATEST_STATE,
        (OperandSpec("o1", owner_aliases=("alice",)),),
        (ProofObligation("need-history", None, "state_history"),))
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=2, max_visited_edges=1,
        max_frontier=4, max_seed_nodes=1)

    result = execute(
        view, ir, ("seed",), budget, structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.proof[0].edge_id == "e-masked"


def test_typed_region_arrival_descends_to_fact_on_second_hop() -> None:
    nodes = (
        GraphNode("seed", "m", NodeType.SCENE, 0, "Alice old home", "g:0"),
        GraphNode("region", "m", NodeType.SCENE, 0, "Alice new home", "g:1"),
        _node("fact", "Alice lives in Paris"),
        GraphNode("noise", "m", NodeType.SCENE, 0, "other region", "g:2"),
    )
    edges = (
        GraphEdge("e-cross", "m", "seed", RelationType.COARSE_RELATED,
                  "region", "g:0", False, 0.9,
                  "relation_mask:shared_entity,state_compatible"),
        GraphEdge("e-fact", "m", "region", RelationType.SCENE_CONTAINS,
                  "fact", "g:1", True, 1.0, "structural"),
        GraphEdge("e-noise", "m", "region", RelationType.COARSE_RELATED,
                  "noise", "g:1", False, 0.9, "relation_mask:scene_similar"),
    )
    view = GraphReadView(nodes, edges)
    ir = QueryIR(
        "Where does Alice live?", QueryOperator.LATEST_STATE,
        (OperandSpec("o1", owner_aliases=("alice",)),),
        (ProofObligation("need-history", None, "state_history"),))
    budget = QueryBudget(
        max_hops=2, max_visited_nodes=3, max_visited_edges=2,
        max_frontier=4, max_seed_nodes=1)

    result = execute(
        view, ir, ("seed",), budget, structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.visited_node_ids == ("seed", "region", "fact")
    assert result.proof[-1].relation == RelationType.SCENE_CONTAINS


def test_rare_lexical_bridge_is_conditioned_on_multifact_plan_and_descends() -> None:
    nodes = (
        GraphNode("seed", "m", NodeType.SCENE, 0, "anchor", "g:0"),
        GraphNode("rare-region", "m", NodeType.SCENE, 0,
                  "related region", "g:1"),
        GraphNode("semantic-region", "m", NodeType.SCENE, 0, "nearby region", "g:2"),
        _node("fact", "second required value"),
    )
    edges = (
        GraphEdge("e-rare", "m", "seed", RelationType.COARSE_RELATED,
                  "rare-region", "g:0", False, 0.9,
                  "relation_mask:lexical_rare"),
        GraphEdge("e-scene", "m", "seed", RelationType.COARSE_RELATED,
                  "semantic-region", "g:0", False, 0.9,
                  "relation_mask:scene_similar"),
        GraphEdge("e-fact", "m", "rare-region", RelationType.SCENE_CONTAINS,
                  "fact", "g:1", True, 1.0, "structural"),
    )
    view = GraphReadView(nodes, edges)
    ir = QueryIR(
        "What were the two recorded values?", QueryOperator.UNION_DISTINCT,
        (OperandSpec("o1"), OperandSpec("o2")),
        (ProofObligation("need-collection", None, "collection"),))
    budget = QueryBudget(
        max_hops=2, max_visited_nodes=3, max_visited_edges=2,
        max_frontier=4, max_seed_nodes=1)

    result = execute(
        view, ir, ("seed",), budget, structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.visited_node_ids == ("seed", "rare-region", "fact")
    assert result.proof[0].edge_id == "e-rare"


def test_typed_executor_requires_post_pack_proof_and_reports_units() -> None:
    algebra = AlgebraResult(
        QueryOperator.DATE_DIFFERENCE, (), ("b1", "b2"),
        temporal_endpoints=(
            TemporalEndpoint(
                "start", TemporalKey(
                    start="2025-01-01", precision="day", kind="absolute",
                    confidence=1.0), "b1"),
            TemporalEndpoint(
                "end", TemporalKey(
                    start="2025-01-11", precision="day", kind="absolute",
                    confidence=1.0), "b2")),
        scope_complete=True, answer_kind="date_difference")
    pre_pack = inspect_execution(algebra)
    certificate = EvidenceCertificate(
        "date_difference", (), (), (), True, 1,
        post_pack_complete=True)
    certified = inspect_execution(algebra, certificate)

    assert pre_pack and not pre_pack.safe_to_bypass
    assert certified and certified.safe_to_bypass
    assert certified.value == 10 and certified.unit == "days"
    assert certified.interval_uncertainty == "exact"
    assert certified.provenance_binding_ids == ("b1", "b2")


def test_single_member_count_cannot_establish_closed_world_by_itself() -> None:
    algebra = AlgebraResult(
        QueryOperator.COUNT_DISTINCT, (), (), count=1,
        scope_complete=True, answer_kind="count")
    certificate = EvidenceCertificate(
        "count", (), (), (), True, 1, post_pack_complete=True)

    decision = inspect_execution(algebra, certificate)

    assert decision and not decision.safe_to_bypass
    assert "single_member_closed_world_unproven" in decision.reason_codes
