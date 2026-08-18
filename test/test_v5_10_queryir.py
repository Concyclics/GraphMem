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
from graphmem.retrieval.navigator import exact_lookup_eligible
from graphmem.retrieval.query_ir import QueryIR
from graphmem.retrieval.scheduler import execute
from graphmem.retrieval.slots import QuerySlots
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


def test_exact_lookup_gate_excludes_plural_and_preference_plans() -> None:
    operand = OperandSpec("o1", owner_aliases=("alice",))
    obligation = ProofObligation("need-value", "o1", "value")
    scalar = QueryIR(
        "Where did Alice buy the camera?", QueryOperator.LOOKUP,
        (operand,), (obligation,), slots=QuerySlots())
    plural = QueryIR(
        "What places did Alice visit?", QueryOperator.LOOKUP,
        (operand,), (obligation,), slots=QuerySlots(expects_multiple=True))
    preference = QueryIR(
        "What book should Alice read?", QueryOperator.LOOKUP,
        (operand,), (obligation,), slots=QuerySlots())
    tips = QueryIR(
        "My phone battery is weak. Any tips?", QueryOperator.LOOKUP,
        (operand,), (obligation,), slots=QuerySlots(is_advice=True))

    assert exact_lookup_eligible(scalar)
    assert not exact_lookup_eligible(plural)
    assert not exact_lookup_eligible(preference)
    assert not exact_lookup_eligible(tips)


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


def test_relation_witness_prunes_same_type_edge_for_wrong_entity() -> None:
    nodes = (
        _node("seed", "current address"),
        _node("alice", "Alice address history"),
        _node("bob", "Bob address history"),
    )
    edges = (
        _edge(
            "e-bob", RelationType.COARSE_RELATED, "bob",
            'relation_mask:state_compatible|relation_witness:{"state_compatible":["bob\\u001flives in"]}'),
        _edge(
            "e-alice", RelationType.COARSE_RELATED, "alice",
            'relation_mask:state_compatible|relation_witness:{"state_compatible":["alice\\u001flives in"]}'),
    )
    ir = QueryIR(
        "Where does Alice currently live?", QueryOperator.LATEST_STATE,
        (OperandSpec("o1", owner_aliases=("alice",),
                     predicate_candidates=("lives in",)),),
        (ProofObligation("need-history", None, "state_history"),))
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=2, max_visited_edges=1,
        max_frontier=4, max_seed_nodes=1)

    result = execute(
        GraphReadView(nodes, edges), ir, ("seed",), budget,
        structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.visited_node_ids == ("seed", "alice")
    assert result.proof[0].edge_id == "e-alice"


def test_predicate_family_witness_matches_inflection_and_polarity() -> None:
    nodes = (
        _node("seed", "Alice event history"),
        _node("positive", "Alice attended events"),
        _node("negative", "Alice did not attend events"),
    )
    edges = (
        _edge(
            "e-negative", RelationType.COARSE_RELATED, "negative",
            'relation_mask:state_compatible|relation_witness:'
            '{"state_compatible":["alice\\u001fattend\\u001fnegative\\u001fasserted"]}'),
        _edge(
            "e-positive", RelationType.COARSE_RELATED, "positive",
            'relation_mask:state_compatible|relation_witness:'
            '{"state_compatible":["alice\\u001fattend\\u001fpositive\\u001fasserted"]}'),
    )
    ir = QueryIR(
        "Which events has Alice attended?", QueryOperator.UNION_DISTINCT,
        (OperandSpec("o1", owner_aliases=("alice",),
                     predicate_candidates=("attending events",),
                     polarity="positive"),),
        (ProofObligation("binding", "o1", "binding"),))
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=2, max_visited_edges=1,
        max_frontier=4, max_seed_nodes=1)

    result = execute(
        GraphReadView(nodes, edges), ir, ("seed",), budget,
        structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.visited_node_ids == ("seed", "positive")
    assert result.proof[0].edge_id == "e-positive"


def test_relation_mask_without_witness_remains_backward_compatible() -> None:
    view = GraphReadView(
        (_node("seed", "Alice"), _node("state", "new address")),
        (_edge("e-state", RelationType.COARSE_RELATED, "state",
               "relation_mask:state_compatible"),))
    ir = QueryIR(
        "Where does Alice currently live?", QueryOperator.LATEST_STATE,
        (OperandSpec("o1", owner_aliases=("alice",)),),
        (ProofObligation("need-history", None, "state_history"),))
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=2, max_visited_edges=1,
        max_frontier=2, max_seed_nodes=1)

    result = execute(
        view, ir, ("seed",), budget, structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.visited_node_ids == ("seed", "state")


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

    disabled = execute(
        view, ir, ("seed",), budget, structured=True, expansion_beam=1,
        rare_lexical_relations=False,
        preferred_relations=(RelationType.COARSE_RELATED,))
    result = execute(
        view, ir, ("seed",), budget, structured=True, expansion_beam=1,
        rare_lexical_relations=True,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert disabled.visited_node_ids == ("seed", "semantic-region")
    assert result.visited_node_ids == ("seed", "rare-region", "fact")
    assert result.proof[0].edge_id == "e-rare"


def test_query_gated_rare_lexical_rejects_lookup_but_keeps_multifact() -> None:
    nodes = (
        GraphNode("seed", "m", NodeType.SCENE, 0, "anchor", "g:0"),
        GraphNode("rare", "m", NodeType.SCENE, 0, "same topic", "g:1"),
        GraphNode("semantic", "m", NodeType.SCENE, 0, "direct match", "g:2"),
    )
    edges = (
        GraphEdge("e-rare", "m", "seed", RelationType.COARSE_RELATED,
                  "rare", "g:0", False, 0.9,
                  "relation_mask:lexical_rare"),
        GraphEdge("e-semantic", "m", "seed", RelationType.COARSE_RELATED,
                  "semantic", "g:0", False, 0.9,
                  "relation_mask:scene_similar"),
    )
    view = GraphReadView(nodes, edges)
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=2, max_visited_edges=1,
        max_frontier=4, max_seed_nodes=1)
    common = dict(
        structured=True, expansion_beam=1, rare_lexical_relations=True,
        query_gated_rare_lexical=True,
        preferred_relations=(RelationType.COARSE_RELATED,))

    lookup = execute(
        view,
        QueryIR("What is the value?", QueryOperator.LOOKUP,
                (OperandSpec("o1"),),
                (ProofObligation("need-value", "o1", "value"),)),
        ("seed",), budget, **common)
    multifact = execute(
        view,
        QueryIR("What are both values?", QueryOperator.UNION_DISTINCT,
                (OperandSpec("o1"), OperandSpec("o2")),
                (ProofObligation("need-collection", None, "collection"),)),
        ("seed",), budget, **common)

    assert lookup.proof[0].edge_id == "e-semantic"
    assert multifact.proof[0].edge_id == "e-rare"


def test_disabling_rare_lexical_keeps_edges_with_nonlexical_signals() -> None:
    nodes = (
        GraphNode("seed", "m", NodeType.SCENE, 0, "anchor", "g:0"),
        GraphNode("mixed", "m", NodeType.SCENE, 0,
                  "entity-related region", "g:1"),
    )
    edges = (GraphEdge(
        "e-mixed", "m", "seed", RelationType.COARSE_RELATED,
        "mixed", "g:0", False, 0.9,
        "relation_mask:lexical_rare,shared_entity"),)
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=2, max_visited_edges=1,
        max_frontier=2, max_seed_nodes=1)

    result = execute(
        GraphReadView(nodes, edges),
        QueryIR("related entity", QueryOperator.LOOKUP, (OperandSpec("o1"),),
                (ProofObligation("need-value", "o1", "value"),)),
        ("seed",), budget, structured=True, expansion_beam=1,
        rare_lexical_relations=False,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.visited_node_ids == ("seed", "mixed")


def test_layered_search_reranks_each_level_and_reaches_leaf_at_hop_cap() -> None:
    nodes = (
        GraphNode("seed", "m", NodeType.SCENE, 0, "query anchor", "g:0"),
        GraphNode(
            "route", "m", NodeType.ROUTING_CARD, 2, "candidate region", "g:1",
            attributes={"child_postings": {"sapphire": ("card-good",)}}),
        GraphNode(
            "card-good", "m", NodeType.ROUTING_CARD, 1, "selected session", "g:1",
            attributes={"child_postings": {"sapphire": ("scene-good",)}}),
        GraphNode("card-bad", "m", NodeType.ROUTING_CARD, 1,
                  "unrelated session", "g:2"),
        GraphNode("scene-good", "m", NodeType.SCENE, 0,
                  "selected scene", "g:1"),
        GraphNode("scene-bad", "m", NodeType.SCENE, 0,
                  "unrelated scene", "g:2"),
        _node("fact-good", "sapphire project reached the final stage"),
        _node("fact-bad", "ordinary unrelated detail"),
    )
    edges = (
        GraphEdge("e-cross", "m", "seed", RelationType.COARSE_RELATED,
                  "route", "g:0", False, 0.9,
                  "relation_mask:lexical_rare"),
        GraphEdge("e-card-good", "m", "route", RelationType.REFINES_TO,
                  "card-good", "g:1", True, 1.0, "structural"),
        GraphEdge("e-card-bad", "m", "route", RelationType.REFINES_TO,
                  "card-bad", "g:2", True, 1.0, "structural"),
        GraphEdge("e-scene-good", "m", "card-good", RelationType.REFINES_TO,
                  "scene-good", "g:1", True, 1.0, "structural"),
        GraphEdge("e-scene-bad", "m", "card-bad", RelationType.REFINES_TO,
                  "scene-bad", "g:2", True, 1.0, "structural"),
        GraphEdge("e-fact-good", "m", "scene-good", RelationType.SCENE_CONTAINS,
                  "fact-good", "g:1", True, 1.0, "structural"),
        GraphEdge("e-fact-bad", "m", "scene-bad", RelationType.SCENE_CONTAINS,
                  "fact-bad", "g:2", True, 1.0, "structural"),
    )
    view = GraphReadView(nodes, edges)
    ir = QueryIR(
        "What happened to the sapphire project?", QueryOperator.LOOKUP,
        (OperandSpec("o1"),), (ProofObligation("need-value", "o1", "value"),))
    budget = QueryBudget(
        max_hops=1, max_visited_nodes=5, max_visited_edges=4,
        max_frontier=4, max_seed_nodes=1)

    result = execute(
        view, ir, ("seed",), budget, structured=True, expansion_beam=1,
        rare_lexical_relations=True,
        preferred_relations=(RelationType.COARSE_RELATED,
                             RelationType.REFINES_TO))

    assert result.visited_node_ids == (
        "seed", "route", "card-good", "scene-good", "fact-good")
    assert result.node_hops["route"] == 1
    assert result.node_hops["fact-good"] == 1
    assert "card-bad" not in result.visited_node_ids


def test_layered_search_separates_relation_hops_from_structural_depth() -> None:
    nodes = (
        GraphNode("seed", "m", NodeType.SCENE, 0, "anchor", "g:0"),
        GraphNode("route-a", "m", NodeType.ROUTING_CARD, 2,
                  "first related region", "g:1"),
        GraphNode("card-a", "m", NodeType.ROUTING_CARD, 1,
                  "first session", "g:1"),
        GraphNode("route-b", "m", NodeType.ROUTING_CARD, 1,
                  "second related region", "g:2"),
        GraphNode("scene-b", "m", NodeType.SCENE, 0,
                  "target scene", "g:2"),
        _node("fact-b", "the target evidence"),
    )
    edges = (
        GraphEdge("e-rel-1", "m", "seed", RelationType.COARSE_RELATED,
                  "route-a", "g:0", False, 0.9,
                  "relation_mask:shared_entity"),
        GraphEdge("e-down-a", "m", "route-a", RelationType.REFINES_TO,
                  "card-a", "g:1", True, 1.0, "structural"),
        GraphEdge("e-rel-2", "m", "card-a", RelationType.COARSE_RELATED,
                  "route-b", "g:1", False, 0.9,
                  "relation_mask:temporal_near"),
        GraphEdge("e-down-b", "m", "route-b", RelationType.REFINES_TO,
                  "scene-b", "g:2", True, 1.0, "structural"),
        GraphEdge("e-fact-b", "m", "scene-b", RelationType.SCENE_CONTAINS,
                  "fact-b", "g:2", True, 1.0, "structural"),
    )
    budget = QueryBudget(
        max_hops=2, max_visited_nodes=6, max_visited_edges=5,
        max_frontier=4, max_seed_nodes=1)

    result = execute(
        GraphReadView(nodes, edges),
        QueryIR("target evidence", QueryOperator.LOOKUP, (OperandSpec("o1"),),
                (ProofObligation("need-value", "o1", "value"),)),
        ("seed",), budget, structured=True, expansion_beam=1,
        preferred_relations=(RelationType.COARSE_RELATED,))

    assert result.visited_node_ids == (
        "seed", "route-a", "card-a", "route-b", "scene-b", "fact-b")
    assert result.node_hops["route-a"] == 1
    assert result.node_hops["card-a"] == 1
    assert result.node_hops["route-b"] == 2
    assert result.node_hops["fact-b"] == 2


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
