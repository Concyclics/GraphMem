from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from graphmem.domain import GraphNode, NodeType, OperandSpec, ProofObligation, QueryOperator, stable_id
from graphmem.retrieval.facts import (
    ACTIVE_PER_OPERAND,
    build_fact_reservoir,
    select_active_facts,
    turn_group_index,
)
from graphmem.retrieval.query_ir import QueryIR
from graphmem.runtime import GraphReadView


def _fact(node_id: str, owner: str, predicate: str, value: str, group: str) -> GraphNode:
    return GraphNode(node_id, "m", NodeType.CANONICAL_FACT, 0, f"{owner} {predicate} {value}",
                     group, attributes={"owner_id": owner, "predicate": predicate,
                                        "value": value, "value_key": value, "scope": "travel"})


def _entity(node_id: str, name: str) -> GraphNode:
    return GraphNode(node_id, "m", NodeType.CANONICAL_ENTITY, 0, name, "group-e",
                     attributes={"aliases": (name,)})


@dataclass
class _Seed:
    operand_turns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source_turn_ids: tuple[str, ...] = ()


def _ir(operands: Sequence[OperandSpec]) -> QueryIR:
    return QueryIR("where did alice volunteer", QueryOperator.LOOKUP, tuple(operands),
                   tuple(ProofObligation(stable_id("obligation", item.operand_id, "binding"),
                                         item.operand_id, "binding") for item in operands))


def _view_and_groups(fact_count: int = 400):
    """One memory with many facts, only one of which the gold turn produced."""
    nodes = [_entity("owner-alice", "alice"), _entity("owner-bob", "bob")]
    for index in range(fact_count):
        nodes.append(_fact(f"fact:noise{index:04d}", "owner-bob", "eat", f"snack {index}",
                           f"group:noise{index:04d}"))
    nodes.append(_fact("fact:gold", "owner-alice", "volunteer", "homeless shelter", "group:gold"))
    view = GraphReadView(nodes, [])
    groups_by_turn = {"turn:gold": ("group:gold",),
                      **{f"turn:noise{index:04d}": (f"group:noise{index:04d}",)
                         for index in range(fact_count)}}
    return view, groups_by_turn


def test_reverse_projection_recovers_a_fact_the_postings_would_miss() -> None:
    """The channel that matters: the gold turn is reachable, so its fact must be.

    The operand names a predicate the fact does not use, so owner/predicate
    postings alone cannot return it; only projecting back through the source
    turn's evidence group finds it.
    """
    view, groups_by_turn = _view_and_groups()
    operand = OperandSpec(operand_id="only", owner_aliases=(), predicate_candidates=("unrelated",))
    seed = _Seed({"only": ("turn:gold",)}, ("turn:gold",))

    reservoir = build_fact_reservoir(view, _ir([operand]), seed, groups_by_turn,
                                     channels=("source_projection",))

    assert "fact:gold" in {row.fact_id for row in reservoir.entries}


def test_structured_channel_alone_misses_it() -> None:
    view, groups_by_turn = _view_and_groups()
    operand = OperandSpec(operand_id="only", owner_aliases=(), predicate_candidates=("unrelated",))
    seed = _Seed({"only": ("turn:gold",)}, ("turn:gold",))

    reservoir = build_fact_reservoir(view, _ir([operand]), seed, groups_by_turn,
                                     channels=("structured",))

    assert "fact:gold" not in {row.fact_id for row in reservoir.entries}


def test_a_highly_ranked_source_turn_admits_its_fact_to_the_shortlist() -> None:
    """Rank carries the evidence: a fact from the top source turn is admissible."""
    view, groups_by_turn = _view_and_groups()
    operand = OperandSpec(operand_id="only", owner_aliases=(), predicate_candidates=("unrelated",))
    seed = _Seed({"only": ("turn:gold",)}, ("turn:gold",))

    reservoir = build_fact_reservoir(view, _ir([operand]), seed, groups_by_turn,
                                     channels=("source_projection",))
    active = select_active_facts(reservoir, view, _ir([operand]))

    assert "fact:gold" in set(active.active)


def test_a_deeply_ranked_unrelated_fact_stays_out_of_the_shortlist() -> None:
    """The reservoir may be wide; the shortlist must not be."""
    view, groups_by_turn = _view_and_groups()
    operand = OperandSpec(operand_id="only", owner_aliases=("alice",),
                          predicate_candidates=("volunteer",))
    turns = tuple(f"turn:noise{index:04d}" for index in range(200))
    seed = _Seed({"only": ("turn:gold", *turns)}, ("turn:gold", *turns))
    ir = _ir([operand])

    active = select_active_facts(
        build_fact_reservoir(view, ir, seed, groups_by_turn, channels=("source_projection",)),
        view, ir)

    assert "fact:gold" in set(active.active)
    assert len(active.active) < 200


def test_core_facts_are_never_displaced_by_rescues() -> None:
    view, groups_by_turn = _view_and_groups()
    operand = OperandSpec(operand_id="only", owner_aliases=("alice",),
                          predicate_candidates=("volunteer",))
    turns = tuple(f"turn:noise{index:04d}" for index in range(300))
    seed = _Seed({"only": turns}, turns)
    ir = _ir([operand])

    reservoir = build_fact_reservoir(view, ir, seed, groups_by_turn,
                                     core_fact_ids=("fact:gold",),
                                     channels=("source_projection", "structured"))
    active = select_active_facts(reservoir, view, ir)

    assert "fact:gold" in set(active.active)
    assert active.stats["active_core"] >= 1


def test_each_operand_gets_its_own_share_of_the_active_budget() -> None:
    view, groups_by_turn = _view_and_groups()
    left = OperandSpec(operand_id="left", owner_aliases=("alice",), predicate_candidates=("volunteer",))
    right = OperandSpec(operand_id="right", owner_aliases=("bob",), predicate_candidates=("eat",))
    turns = tuple(f"turn:noise{index:04d}" for index in range(200))
    seed = _Seed({"left": ("turn:gold",), "right": turns}, ("turn:gold", *turns))
    ir = _ir([left, right])

    active = select_active_facts(
        build_fact_reservoir(view, ir, seed, groups_by_turn,
                             channels=("source_projection", "structured")), view, ir)

    per_operand = active.stats["per_operand_active"]
    assert per_operand["left"] >= 1
    assert per_operand["right"] >= 1
    assert all(value <= ACTIVE_PER_OPERAND for value in per_operand.values())


def test_reservoir_holds_ids_only_and_hydrates_no_text() -> None:
    view, groups_by_turn = _view_and_groups()
    operand = OperandSpec(operand_id="only", owner_aliases=("alice",), predicate_candidates=("volunteer",))
    seed = _Seed({"only": ("turn:gold",)}, ("turn:gold",))

    reservoir = build_fact_reservoir(view, _ir([operand]), seed, groups_by_turn)

    for row in reservoir.entries:
        assert not hasattr(row, "raw_text")
        assert isinstance(row.evidence_group_ids, tuple)


def test_reservoir_is_deterministic() -> None:
    view, groups_by_turn = _view_and_groups()
    operand = OperandSpec(operand_id="only", owner_aliases=("alice",), predicate_candidates=("volunteer",))
    turns = tuple(f"turn:noise{index:04d}" for index in range(50))
    seed = _Seed({"only": ("turn:gold", *turns)}, ("turn:gold", *turns))
    ir = _ir([operand])

    first = select_active_facts(build_fact_reservoir(view, ir, seed, groups_by_turn), view, ir)
    second = select_active_facts(build_fact_reservoir(view, ir, seed, groups_by_turn), view, ir)

    assert first.active == second.active


def test_turn_group_index_maps_every_member(tmp_path) -> None:
    from graphmem.domain import Conversation, EvidenceGroup, EvidenceMember, Session, SourceTurn
    from graphmem.storage import SQLiteGraphStore
    import hashlib

    store = SQLiteGraphStore(tmp_path / "graph.sqlite")
    text = "I volunteered at the shelter"
    turn = SourceTurn(stable_id("turn", "m", "s1", 0), "m", "s1", 0, "Alice", "Bob", "user", None,
                      text, hashlib.sha256(text.encode()).hexdigest())
    store.ingest_conversation(Conversation("m", "golden", "m", "hash"),
                              [Session("s1", "m", 0, None, "sh")], [turn])
    group = EvidenceGroup("group:1", "m", (EvidenceMember(turn.turn_id, 0, 5, "quote"),), "chash")
    store.replace_graph("m", [], [], [group])

    index = turn_group_index(store, "m")

    assert index[turn.turn_id] == ("group:1",)
    store.close()
