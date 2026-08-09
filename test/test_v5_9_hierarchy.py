from __future__ import annotations

from graphmem.domain import (
    GraphEdge,
    GraphNode,
    NodeType,
    OperandSpec,
    QueryOperator,
    RelationType,
)
from graphmem.retrieval import operators as ops
from graphmem.retrieval.hierarchy import compile_physical_route, route_hierarchy
from graphmem.retrieval.query_ir import QueryIR
from graphmem.runtime import GraphReadView


def _node(node_id: str, node_type: NodeType, level: int, summary: str, **attrs) -> GraphNode:
    return GraphNode(node_id, "m", node_type, level, summary, f"g:{node_id}",
                     attributes=attrs)


def _edge(edge_id: str, source: str, target: str) -> GraphEdge:
    return GraphEdge(edge_id, "m", source, RelationType.REFINES_TO, target,
                     f"g:{source}", True, 1.0, "test")


def _view() -> GraphReadView:
    nodes = (
        _node("root", NodeType.ROUTING_CARD, 2, "memory root", roles=("route",),
              child_postings={"kyoto": ("travel",), "cat": ("pets",)}),
        _node("travel", NodeType.ROUTING_CARD, 1, "Alice travelled to Kyoto",
              roles=("route",), child_postings={"kyoto": ("f-kyoto",)}),
        _node("pets", NodeType.ROUTING_CARD, 1, "Bob adopted a cat",
              roles=("route",), child_postings={"cat": ("f-cat",)}),
        _node("f-kyoto", NodeType.CANONICAL_FACT, 0, "Alice visited Kyoto",
              owner_id="alice", predicate="visit", value="Kyoto"),
        _node("f-cat", NodeType.CANONICAL_FACT, 0, "Bob owns a cat",
              owner_id="bob", predicate="own", value="cat"),
    )
    edges = (
        _edge("e1", "root", "travel"), _edge("e2", "root", "pets"),
        _edge("e3", "travel", "f-kyoto"), _edge("e4", "pets", "f-cat"),
    )
    return GraphReadView(nodes, edges)


def test_read_view_exposes_directional_hierarchy_apis() -> None:
    view = _view()

    assert view.routing_roots() == ("root",)
    assert set(view.hierarchy_children("root")) == {"travel", "pets"}
    assert view.hierarchy_parents("travel") == ("root",)
    assert view.hierarchy_children("f-kyoto") == ()


def test_top_down_route_never_opens_an_unselected_sibling_subtree() -> None:
    view = _view()
    operand = OperandSpec("o1", predicate_candidates=("visit",))
    ast = ops.Lookup(ops.FactSet("o1"))
    ir = QueryIR("Where did Alice visit in Kyoto?", QueryOperator.LOOKUP,
                 (operand,), (), ast=ast, ast_operands=(operand,))
    plan = compile_physical_route(
        ir, max_nodes=8, root_beam=1, child_beam=1,
        operator_aware=False)

    result = route_hierarchy(view, plan)

    assert "travel" in result.selected_node_ids
    assert "f-kyoto" in result.terminal_node_ids
    assert "pets" not in result.selected_node_ids
    assert "f-cat" not in result.visited_node_ids
    assert result.candidate_count == 4  # root, its two children, then one leaf


def test_inverted_node_index_replaces_a_full_graph_scan() -> None:
    view = _view()

    assert view.lexical_nodes(frozenset({"kyoto"}), limit=2) == (
        "f-kyoto", "travel")


def test_operator_aware_portal_rescues_a_term_lost_by_parent_summary() -> None:
    nodes = (
        _node("root", NodeType.ROUTING_CARD, 3, "memory root", roles=("route",)),
        _node("a", NodeType.ROUTING_CARD, 2, "common partition", roles=("route",)),
        _node("b", NodeType.ROUTING_CARD, 2, "common partition", roles=("route",)),
        _node("leaf-a", NodeType.ROUTING_CARD, 1, "ordinary update",
              roles=("route",), session_id="s-a"),
        _node("leaf-b", NodeType.ROUTING_CARD, 1, "cobalt launch",
              roles=("route",), session_id="s-b"),
    )
    edges = (
        _edge("e1", "root", "a"), _edge("e2", "root", "b"),
        _edge("e3", "a", "leaf-a"), _edge("e4", "b", "leaf-b"),
    )
    view = GraphReadView(nodes, edges)
    operand = OperandSpec("o1", predicate_candidates=("cobalt",))
    ast = ops.Lookup(ops.FactSet("o1"))
    ir = QueryIR("What was the cobalt launch?", QueryOperator.LOOKUP,
                 (operand,), (), ast=ast, ast_operands=(operand,))

    fixed = route_hierarchy(view, compile_physical_route(
        ir, max_nodes=8, root_beam=1, child_beam=1,
        operator_aware=False))
    adaptive = route_hierarchy(view, compile_physical_route(
        ir, max_nodes=8, root_beam=1, child_beam=1,
        operator_aware=True))

    assert "leaf-b" not in fixed.terminal_node_ids
    assert "leaf-b" in adaptive.terminal_node_ids
    assert adaptive.portal_node_ids == ("leaf-b",)
    assert "b" in adaptive.selected_node_ids
