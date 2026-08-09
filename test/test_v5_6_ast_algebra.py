from __future__ import annotations

from graphmem.domain import (
    FactBinding, GraphNode, NodeType, QueryOperator, TemporalKey, stable_id,
)
from graphmem.projection.config import ProjectionConfig
from graphmem.projection.manifest import build_manifests, manifest_stats
from graphmem.retrieval import operators as ops
from graphmem.retrieval.ast_algebra import evaluate_ast
from graphmem.runtime import GraphReadView


def _binding(operand: str, value: str, *, owner: str = "alice", start: str | None = None,
             event: str | None = None, session: str = "s1", turn: int = 0,
             predicate: str = "own") -> FactBinding:
    return FactBinding(
        binding_id=f"b:{operand}:{value}:{session}:{turn}", operand_id=operand,
        fact_node_id=f"f:{value}:{turn}", owner_id=owner, predicate=predicate,
        scope="things", value_key=value.casefold(), event_instance_id=event,
        time_interval=TemporalKey(start=start, kind="point", confidence=1.0) if start else None,
        evidence_group_ids=(f"g:{value}",), confidence=1.0, value=value,
        session_id=session, turn_index=turn)


def _fact(value: str, *, owner: str = "alice", predicate: str = "own", turn: int = 0,
          collection: str = "things", event: str | None = None) -> GraphNode:
    attrs = {"owner_id": owner, "predicate": predicate, "scope": "things",
             "collection_key": collection, "polarity": "positive", "modality": "asserted",
             "value": value, "value_key": value.casefold(), "session_id": "s1",
             "turn_index": turn}
    if event:
        attrs["event_instance_id"] = event
    return GraphNode(stable_id("node", "m", "fact", value, turn), "m", NodeType.CANONICAL_FACT,
                     0, f"{owner} {predicate} {value}", f"g:{value}", (), attributes=attrs)


# --- collection manifest ------------------------------------------------------

ON = ProjectionConfig(collection_manifest=True)


def test_no_manifests_are_built_when_the_arm_is_off() -> None:
    """P0 must reproduce the frozen graph exactly."""
    nodes, edges, rows = build_manifests("m", [_fact("hat"), _fact("coat", turn=1)],
                                         ProjectionConfig())

    assert (nodes, edges, rows) == ([], [], [])


def test_a_manifest_enumerates_its_members() -> None:
    facts = [_fact("hat"), _fact("coat", turn=1), _fact("boots", turn=2)]

    nodes, edges, rows = build_manifests("m", facts, ON)

    assert len(nodes) == 1 and nodes[0].node_type == NodeType.COLLECTION_MANIFEST
    assert nodes[0].attributes["member_count"] == 3
    assert nodes[0].attributes["closed"] is True
    assert len(edges) == 3 and all(str(edge.relation) == "member_of" for edge in edges)
    assert rows[0].member_count == 3


def test_manifest_values_are_compiled_into_the_read_view() -> None:
    nodes, edges, _rows = build_manifests(
        "m", [_fact("hat"), _fact("coat", turn=1)], ON)

    view = GraphReadView(nodes, edges)

    assert view.manifest_value_index["hat"] == (nodes[0].node_id,)
    assert view.manifest_value_index["coat"] == (nodes[0].node_id,)


def test_a_single_member_collection_gets_a_manifest() -> None:
    """The frozen build required >=2 rows, so 'how many X' over one X was blind."""
    nodes, _edges, rows = build_manifests("m", [_fact("hat")], ON)

    assert len(nodes) == 1 and rows[0].member_count == 1
    assert manifest_stats(rows)["single_member_manifests"] == 1


def test_min_members_two_reproduces_the_frozen_blind_spot() -> None:
    config = ProjectionConfig(collection_manifest=True, manifest_min_members=2)

    nodes, _edges, _rows = build_manifests("m", [_fact("hat")], config)

    assert nodes == []


def test_distinct_chains_do_not_merge() -> None:
    facts = [_fact("hat", predicate="own"), _fact("kyoto", predicate="visit", turn=1)]

    nodes, _edges, _rows = build_manifests("m", facts, ON)

    assert len(nodes) == 2


#: R1: group by the class of thing, not by the verb.
BY_CATEGORY = ProjectionConfig(collection_manifest=True, chain_includes_scope=False,
                               chain_includes_predicate=False)


def test_one_class_reached_by_different_verbs_is_one_collection() -> None:
    """The defect this arm exists to fix.

    Measured over 149 gold-annotated aggregation questions: 91.5% of a question's
    gold facts land in different chains.  "How many model kits have I worked on
    or bought" has gold facts whose predicates are just/start/finish -- three
    chains, so a count has no set to range over and the answer stage faithfully
    counts whatever subset survived ranking.
    """
    facts = [_fact("B-29 bomber kit", predicate="just got", collection="model kit"),
             _fact("diorama kit", predicate="started working on", collection="model kit", turn=1),
             _fact("Revell F-15 kit", predicate="finished", collection="model kit", turn=2)]

    split = build_manifests("m", facts, ON)[0]
    merged = build_manifests("m", facts, BY_CATEGORY)[0]

    assert len(split) == 3, "the verb-keyed chain splits one collection three ways"
    assert len(merged) == 1
    assert merged[0].attributes["member_count"] == 3
    assert merged[0].attributes["collection_key"] == "model kit"
    # The class name has to survive into the summary, because that and
    # collection_key are what a question's noun is matched against once the
    # predicate has left the chain key.
    assert "model kit" in merged[0].summary


def test_different_classes_still_do_not_merge_without_the_predicate() -> None:
    facts = [_fact("B-29 kit", predicate="bought", collection="model kit"),
             _fact("kyoto", predicate="bought", collection="trip", turn=1)]

    nodes, _edges, _rows = build_manifests("m", facts, BY_CATEGORY)

    assert len(nodes) == 2


def test_repeated_occurrences_stay_separate_members() -> None:
    """Counting occurrences must not deduplicate two rides into one."""
    facts = [_fact("rollercoaster", turn=0, event="e1"),
             _fact("rollercoaster", turn=1, event="e2")]

    nodes, _edges, rows = build_manifests("m", facts, ON)

    assert nodes[0].attributes["member_count"] == 2
    assert nodes[0].attributes["distinct_values"] == 1
    assert rows[0].occurrence and nodes[0].attributes["collection_semantics"] == "event_instances"


def test_manifests_are_deterministic() -> None:
    facts = [_fact("hat"), _fact("coat", turn=1)]

    left = build_manifests("m", facts, ON)
    right = build_manifests("m", list(reversed(facts)), ON)

    assert [row.manifest_id for row in left[2]] == [row.manifest_id for row in right[2]]
    assert left[0][0].attributes["member_ids"] == right[0][0].attributes["member_ids"]


def test_stats_report_what_the_frozen_build_could_not_see() -> None:
    rows = build_manifests("m", [_fact("hat"), _fact("coat", turn=1),
                                 _fact("kyoto", predicate="visit", turn=2)], ON)[2]

    stats = manifest_stats(rows)

    # own{hat,coat} is visible to the frozen build; visit{kyoto} is not.
    assert stats["manifests"] == 2 and stats["invisible_to_frozen_build"] == 1


# --- AST algebra --------------------------------------------------------------

def _count_ast(operand: str = "o1", distinct: str = "value"):
    return ops.CountDistinct(ops.FactSet(operand), distinct_by=distinct)


def test_count_distinct_produces_one_member_per_distinct_value() -> None:
    bindings = [_binding("o1", "hat"), _binding("o1", "coat", turn=1),
                _binding("o1", "hat", turn=2)]

    result = evaluate_ast(_count_ast(), bindings, collection_closed={"o1": True})

    assert result.count == 2
    assert {row.value for row in result.members} == {"hat", "coat"}
    assert result.answer_kind == "count"


def test_a_count_over_an_unclosed_collection_is_not_exact() -> None:
    """Reporting a floor as exact is how an aggregation answer becomes confidently wrong."""
    bindings = [_binding("o1", "hat"), _binding("o1", "coat", turn=1)]

    result = evaluate_ast(_count_ast(), bindings, collection_closed={})

    assert result.count == 2 and not result.scope_complete


def test_a_count_over_a_closed_collection_is_exact() -> None:
    bindings = [_binding("o1", "hat"), _binding("o1", "coat", turn=1)]

    result = evaluate_ast(_count_ast(), bindings, collection_closed={"o1": True})

    assert result.scope_complete


def test_counting_occurrences_does_not_deduplicate_repeats() -> None:
    """Measured failure: 'how many times did I ride' answered 5 when gold was 10."""
    bindings = [_binding("o1", "ride", event="e1"), _binding("o1", "ride", event="e2", turn=1)]

    result = evaluate_ast(_count_ast(distinct="event_instance"), bindings,
                          collection_closed={"o1": True})

    assert result.count == 2


def test_every_member_carries_the_bindings_that_witness_it() -> None:
    bindings = [_binding("o1", "hat"), _binding("o1", "hat", turn=3)]

    result = evaluate_ast(_count_ast(), bindings, collection_closed={"o1": True})

    assert result.count == 1
    assert len(result.members[0].witness_binding_ids) == 2
    assert result.witness_map["hat"] == result.members[0].witness_binding_ids


def test_intersection_keeps_only_values_every_operand_witnesses() -> None:
    node = ops.IntersectionDistinct((ops.FactSet("o1"), ops.FactSet("o2")))
    bindings = [_binding("o1", "kyoto"), _binding("o1", "osaka", turn=1),
                _binding("o2", "kyoto", owner="bob", turn=2)]

    result = evaluate_ast(node, bindings)

    assert [row.value for row in result.members] == ["kyoto"]


def test_a_one_sided_intersection_yields_nothing_and_says_why() -> None:
    node = ops.IntersectionDistinct((ops.FactSet("o1"), ops.FactSet("o2")))
    bindings = [_binding("o1", "kyoto"), _binding("o1", "osaka", turn=1)]

    result = evaluate_ast(node, bindings)

    assert result.members == () and not result.scope_complete


def test_group_by_owner_buckets_values_per_owner() -> None:
    node = ops.GroupByOwner((ops.FactSet("o1"),))
    bindings = [_binding("o1", "kyoto"), _binding("o1", "osaka", owner="bob", turn=1)]

    result = evaluate_ast(node, bindings, collection_closed={"o1": True})

    assert result.groups == {"alice": ("kyoto",), "bob": ("osaka",)}


def test_exists_all_needs_a_witness_for_every_operand() -> None:
    node = ops.ExistsAll((ops.FactSet("o1"), ops.FactSet("o2")))
    bindings = [_binding("o1", "kyoto")]

    result = evaluate_ast(node, bindings)

    assert result.members == ()


def test_ordinal_selects_the_kth_element_by_time_not_the_first() -> None:
    """V5.5 parsed the ordinal word but never carried k."""
    node = ops.Ordinal(ops.FactSet("o1"), index=2)
    bindings = [_binding("o1", "a", start="2023-01-01"),
                _binding("o1", "b", start="2023-02-01", turn=1),
                _binding("o1", "c", start="2023-03-01", turn=2)]

    result = evaluate_ast(node, bindings)

    assert [row.value for row in result.members] == ["b"]


def test_argmax_time_takes_the_latest() -> None:
    node = ops.ArgMaxTime(ops.FactSet("o1"))
    bindings = [_binding("o1", "a", start="2023-01-01"),
                _binding("o1", "c", start="2023-03-01", turn=2)]

    result = evaluate_ast(node, bindings)

    assert [row.value for row in result.members] == ["c"]


def test_unresolved_intervals_are_excluded_and_recorded() -> None:
    node = ops.ArgMaxTime(ops.FactSet("o1"))
    bindings = [_binding("o1", "a", start="2023-01-01"), _binding("o1", "b", turn=1)]

    result = evaluate_ast(node, bindings)

    assert "unresolved_intervals_excluded" in result.degradations


def test_an_ordering_with_no_resolved_time_degrades_rather_than_guessing() -> None:
    node = ops.ArgMaxTime(ops.FactSet("o1"))

    result = evaluate_ast(node, [_binding("o1", "a")])

    assert result.members == ()
    assert "no_resolved_time_interval" in result.degradations


def test_date_difference_reports_both_endpoints() -> None:
    node = ops.DateDifference(ops.FactSet("o1"), ops.FactSet("o1"))
    bindings = [_binding("o1", "a", start="2023-01-01"),
                _binding("o1", "b", start="2023-03-02", turn=1)]

    result = evaluate_ast(node, bindings)

    assert len(result.temporal_endpoints) == 2
    assert result.temporal_endpoints[0].key.days_between(
        result.temporal_endpoints[1].key) == -60


def test_latest_state_reports_the_current_value_and_its_predecessors() -> None:
    node = ops.LatestState(ops.FactSet("o1"))
    bindings = [_binding("o1", "boston", start="2022-01-01"),
                _binding("o1", "kyoto", start="2024-01-01", turn=1)]

    result = evaluate_ast(node, bindings)

    assert result.state_result is not None
    assert result.state_result.current_value == "kyoto"
    assert result.state_result.superseded


def test_evaluation_is_deterministic() -> None:
    bindings = [_binding("o1", "hat"), _binding("o1", "coat", turn=1)]

    left = evaluate_ast(_count_ast(), bindings, collection_closed={"o1": True})
    right = evaluate_ast(_count_ast(), list(reversed(bindings)), collection_closed={"o1": True})

    assert left == right


def test_count_executes_its_intersection_child_instead_of_counting_the_union() -> None:
    node = ops.CountDistinct(ops.IntersectionDistinct((
        ops.FactSet("o1"), ops.FactSet("o2"))))
    bindings = [_binding("o1", "kyoto"), _binding("o1", "osaka", turn=1),
                _binding("o2", "kyoto", owner="bob", turn=2),
                _binding("o2", "nagoya", owner="bob", turn=3)]

    result = evaluate_ast(node, bindings,
                          collection_closed={"o1": True, "o2": True})

    assert result.count == 1
    assert [row.value for row in result.members] == ["kyoto"]
    assert set(result.members[0].operand_ids) == {"o1", "o2"}
    assert result.scope_complete


def test_count_executes_its_union_child() -> None:
    node = ops.CountDistinct(ops.UnionDistinct((
        ops.FactSet("o1"), ops.FactSet("o2"))))
    bindings = [_binding("o1", "kyoto"), _binding("o1", "osaka", turn=1),
                _binding("o2", "kyoto", owner="bob", turn=2),
                _binding("o2", "nagoya", owner="bob", turn=3)]

    result = evaluate_ast(node, bindings,
                          collection_closed={"o1": True, "o2": True})

    assert result.count == 3
    assert {row.value for row in result.members} == {"kyoto", "osaka", "nagoya"}


def test_date_difference_respects_left_and_right_operand_scopes() -> None:
    node = ops.DateDifference(ops.FactSet("o1"), ops.FactSet("o2"))
    bindings = [
        _binding("o1", "left-a", start="2023-03-01"),
        _binding("o1", "left-b", start="2023-12-01", turn=1),
        _binding("o2", "right-a", start="2023-01-01", turn=2),
        _binding("o2", "right-b", start="2023-06-01", turn=3),
    ]

    result = evaluate_ast(node, bindings)

    assert result.temporal_endpoints[0].binding_id.startswith("b:o1:left-a")
    assert result.temporal_endpoints[1].binding_id.startswith("b:o2:right-b")


def test_descending_ordinal_reverses_the_time_order() -> None:
    node = ops.Ordinal(ops.FactSet("o1"), index=2, order="descending")
    bindings = [_binding("o1", "a", start="2023-01-01"),
                _binding("o1", "b", start="2023-02-01", turn=1),
                _binding("o1", "c", start="2023-03-01", turn=2)]

    result = evaluate_ast(node, bindings, collection_closed={"o1": True})

    assert [row.value for row in result.members] == ["b"]
    assert result.scope_complete


def test_latest_state_marks_overlapping_conflicting_values_unsafe() -> None:
    node = ops.LatestState(ops.FactSet("o1"))
    bindings = [
        _binding("o1", "kyoto", start="2023-03-01"),
        _binding("o1", "osaka", start="2023-03-01", turn=1),
    ]

    result = evaluate_ast(node, bindings, collection_closed={"o1": True})

    assert result.state_result is not None
    assert result.state_result.contradiction_status == "conflicting_latest"
    assert "conflicting_latest_state" in result.degradations
