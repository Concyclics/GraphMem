from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from graphmem.domain import OperandSpec, ProofObligation, QueryOperator, SourceTurn, stable_id
from graphmem.retrieval.query_ir import QueryIR
from graphmem.retrieval.seeding import TurnSearchIndex, _interleave, build_views, seed_operands
from graphmem.retrieval.slots import QuerySlots


def _turn(session_id: str, index: int, text: str) -> SourceTurn:
    turn_id = stable_id("turn", "m", session_id, index)
    return SourceTurn(turn_id, "m", session_id, index, "Alice", "Bob", "user", None, text,
                      hashlib.sha256(text.encode()).hexdigest())


@dataclass
class _StubStore:
    """Returns a fixed BM25 ranking so channel tails are easy to assert on."""

    ranking: Sequence[tuple[str, float]] = ()

    def search_turns(self, memory_id: str, query: str, *, limit: int = 64):
        return list(self.ranking)[:limit]


@dataclass
class _StubNode:
    node_id: str
    summary: str = ""
    confidence: float = 1.0
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass
class _StubView:
    nodes: Mapping[str, _StubNode] = field(default_factory=dict)
    owner_alias_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    facts: tuple[str, ...] = ()

    def lookup_facts(self, **_kwargs) -> tuple[str, ...]:
        return self.facts

    def lookup_collections(self, **_kwargs) -> tuple[str, ...]:
        return ()

    def route_children(self, _keys, *, limit: int = 48) -> tuple[str, ...]:
        return ()


def _ir(query: str, operands: Sequence[OperandSpec]) -> QueryIR:
    obligations = tuple(ProofObligation(stable_id("obligation", item.operand_id, "binding"),
                                        item.operand_id, "binding") for item in operands)
    return QueryIR(query, QueryOperator.LOOKUP, tuple(operands), obligations)


def _operand(name: str, owner: str, predicate: str) -> OperandSpec:
    return OperandSpec(operand_id=name, owner_aliases=(owner,), predicate_candidates=(predicate,))


def test_build_views_gives_each_operand_its_own_angles() -> None:
    """A shared whole-question ranking cannot express two different owners."""
    ir = _ir("Where did John and Maria both volunteer?",
             [_operand("left", "john", "volunteer"), _operand("right", "maria", "volunteer")])

    views = build_views(ir, max_per_operand=6)

    assert sum(1 for row in views if row.operand_id is None) == 1
    for operand_id in ("left", "right"):
        texts = {row.text for row in views if row.operand_id == operand_id}
        assert any("john" in text or "maria" in text for text in texts)
    assert {row.text for row in views if row.operand_id == "left"} != {
        row.text for row in views if row.operand_id == "right"}
    assert views == build_views(ir, max_per_operand=6)


def test_query_relation_view_removes_owners_and_answer_head() -> None:
    operand = _operand("only", "tim", "suggest")
    ir = QueryIR(
        "What city did Tim suggest for the team trip next month?",
        QueryOperator.LOOKUP, (operand,), (),
        slots=QuerySlots(content_terms=(
            "city", "tim", "suggest", "team", "trip", "next", "month")))

    views = build_views(ir, max_per_operand=6, query_relation_view=True)
    relation = next(row for row in views if row.kind == "query_relation")

    assert relation.text == "suggest team trip next month"
    assert relation.dense is False


def test_relational_view_scoring_keeps_reach_but_demotes_owner_only_exact() -> None:
    turns = [
        _turn("s1", 0, "Unrelated photo. [Media shared by Alice; caption: a tree]"),
        _turn("s2", 0, "Alice volunteered at the shelter"),
    ]
    ir = _ir("Where did Alice volunteer?", [
        _operand("only", "alice", "volunteer")])
    store = _StubStore()
    baseline = seed_operands(
        store, _StubView(), "m", ir, turns, dense_search=None,
        use_rrf=True, use_postings=False, wide_reservoir=True,
        native_bm25=True)
    relational = seed_operands(
        store, _StubView(), "m", ir, turns, dense_search=None,
        use_rrf=True, use_postings=False, wide_reservoir=True,
        native_bm25=True, relational_view_scoring=True)

    assert set(relational.source_turn_ids) == set(baseline.source_turn_ids)
    assert relational.raw_scores[turns[0].turn_id]["exact"] \
        < baseline.raw_scores[turns[0].turn_id]["exact"]
    assert relational.raw_scores[turns[1].turn_id]["exact"] \
        > relational.raw_scores[turns[0].turn_id]["exact"]
    by_turn = {row.turn_id: row for row in relational.reservoir}
    assert by_turn[turns[1].turn_id].relational_consensus \
        > by_turn[turns[0].turn_id].relational_consensus
    assert relational.stats["relational_view_scoring"] is True


def test_dense_views_use_one_batch_without_changing_per_view_results() -> None:
    turns = [_turn("s1", index, f"unrelated evidence number {index}")
             for index in range(4)]
    ir = _ir("Where did Alice volunteer?", [_operand("only", "alice", "volunteer")])
    requests = []

    def dense_many(memory_id, rows):
        requests.append((memory_id, tuple(rows)))
        return tuple(
            [(turns[index % len(turns)].turn_id, 1.0)]
            for index, _request in enumerate(rows)
        )

    result = seed_operands(
        _StubStore(), _StubView(), "m", ir, turns,
        dense_search=lambda *_args: (_ for _ in ()).throw(
            AssertionError("batch path must replace individual dense calls")),
        dense_search_many=dense_many,
        use_rrf=True, use_postings=False, wide_reservoir=True,
    )

    assert len(requests) == 1
    assert len(requests[0][1]) == result.stats["dense_views"] == 2
    assert {turns[0].turn_id, turns[1].turn_id} <= set(result.source_turn_ids)


def test_turn_search_index_uses_postings_and_preserves_exact_ranking() -> None:
    turns = [_turn("s1", 0, "Alice visited Kyoto in spring"),
             _turn("s1", 1, "Alice visited Osaka"),
             _turn("s2", 0, "Bob adopted a cat")]
    index = TurnSearchIndex(turns)

    ranked = index.exact("Alice Kyoto", limit=8)

    assert ranked[0][0] == turns[0].turn_id
    assert turns[2].turn_id not in {turn_id for turn_id, _ in ranked}
    assert index.signature == (3, turns[-1].turn_id, turns[-1].content_hash)


def test_turn_search_index_ranks_sessions_without_rebuilding_term_counters() -> None:
    turns = [_turn("travel", 0, "Kyoto train reservation"),
             _turn("travel", 1, "Kyoto hotel"),
             _turn("pets", 0, "adopted a cat")]
    index = TurnSearchIndex(turns)

    assert index.rank_sessions("Kyoto trip", 1) == ("travel",)


def test_native_bm25_is_deterministic_and_avoids_store_fts() -> None:
    turns = [_turn("travel", 0, "Kyoto Kyoto train reservation"),
             _turn("travel", 1, "Kyoto hotel"),
             _turn("pets", 0, "adopted a cat")]
    index = TurnSearchIndex(turns)

    ranked = index.bm25("Kyoto train", limit=8)

    assert ranked == index.bm25("Kyoto train", limit=8)
    assert ranked[0][0] == turns[0].turn_id

    class NoFtsStore:
        def search_turns(self, *_args, **_kwargs):
            raise AssertionError("native seed fusion must not call SQLite FTS")

    result = seed_operands(
        NoFtsStore(), _StubView(), "m",
        _ir("Kyoto train", [_operand("only", "", "train")]), turns,
        dense_search=None, use_rrf=True, use_postings=False,
        wide_reservoir=True, turn_index=index, native_bm25=True)
    assert result.stats["bm25_backend"] == "immutable_memory_index"


def test_reservoir_keeps_the_weakest_hit_of_every_channel() -> None:
    """Min-max normalization sends each channel's worst hit to exactly 0.0.

    Deriving pool membership from the normalized score therefore dropped the
    tail of every channel, which is how turns the legacy navigator retrieved
    went missing from the harness pool.
    """
    turns = [_turn("s1", index, f"volunteer shift number {index}") for index in range(4)]
    weakest = turns[-1].turn_id
    store = _StubStore(ranking=[(turn.turn_id, float(10 - index))
                                for index, turn in enumerate(turns)])
    view = _StubView(nodes={}, owner_alias_index={})
    ir = _ir("volunteer", [_operand("only", "", "volunteer")])

    result = seed_operands(store, view, "m", ir, turns, dense_search=None, use_rrf=True,
                           use_postings=False, wide_reservoir=True)

    assert weakest in set(result.source_turn_ids)
    assert result.raw_scores[weakest]["bm25"] == 0.0
    entry = next(row for row in result.reservoir if row.turn_id == weakest)
    assert entry.parity is True


def test_reservoir_floods_every_session_holding_a_hit() -> None:
    """The legacy navigator floods its eight strongest sessions; dominate that."""
    turns = [_turn("s1", 0, "volunteer at the shelter"), _turn("s1", 1, "unrelated chatter"),
             _turn("s2", 0, "volunteer again"), _turn("s2", 1, "more unrelated chatter"),
             _turn("s3", 0, "nothing relevant here"), _turn("s3", 1, "still nothing")]
    store = _StubStore(ranking=[(turns[0].turn_id, 5.0), (turns[2].turn_id, 4.0)])
    ir = _ir("volunteer", [_operand("only", "", "volunteer")])

    result = seed_operands(store, _StubView(), "m", ir, turns, dense_search=None, use_rrf=True,
                           use_postings=False, wide_reservoir=True)
    pool = set(result.source_turn_ids)

    assert turns[1].turn_id in pool and turns[3].turn_id in pool
    assert result.stats["flooded_sessions"] == 2


def test_narrow_mode_does_not_flood() -> None:
    turns = [_turn("s1", 0, "volunteer at the shelter"), _turn("s1", 1, "unrelated chatter")]
    store = _StubStore(ranking=[(turns[0].turn_id, 5.0)])
    ir = _ir("volunteer", [_operand("only", "", "volunteer")])

    result = seed_operands(store, _StubView(), "m", ir, turns, dense_search=None, use_rrf=True,
                           use_postings=False, wide_reservoir=False)

    assert result.stats["flooded_sessions"] == 0


def test_interleave_gives_every_operand_a_share_of_the_node_budget() -> None:
    """A shared cap consumed in operand order starved the second operand."""
    rows = {"left": tuple(f"L{index}" for index in range(40)),
            "right": tuple(f"R{index}" for index in range(40))}

    selected = _interleave(rows, 20)

    assert len(selected) == 20
    assert sum(1 for item in selected if item.startswith("L")) == 10
    assert sum(1 for item in selected if item.startswith("R")) == 10


def test_interleave_is_deterministic_and_dedupes() -> None:
    rows = {"a": ("x", "y", "z"), "b": ("y", "z", "w")}

    assert _interleave(rows, 10) == _interleave(rows, 10)
    assert len(set(_interleave(rows, 10))) == len(_interleave(rows, 10))


def test_lookup_facts_ranks_before_truncating(tmp_path) -> None:
    """Truncating by node id discarded facts for reasons unrelated to the query.

    The relevant fact is given a node id that sorts last, so an alphabetical
    slice drops it while a relevance-ranked slice keeps it.
    """
    from graphmem.domain import GraphNode, NodeType
    from graphmem.runtime import GraphReadView

    def fact(node_id: str, predicate: str, value: str) -> GraphNode:
        return GraphNode(node_id, "m", NodeType.CANONICAL_FACT, 0, f"Alice {predicate} {value}",
                         "group-1", attributes={"owner_id": "owner-1", "predicate": predicate,
                                                "value": value, "value_key": value, "scope": "travel"})

    nodes = [fact(f"node:a{index:02d}", "eat", f"snack {index}") for index in range(40)]
    nodes.append(fact("node:zzz", "volunteer", "homeless shelter"))
    view = GraphReadView(nodes, [])

    ranked = view.lookup_facts(owner_ids=("owner-1",), predicates=("volunteer",), limit=8)

    assert "node:zzz" in ranked
    assert ranked[0] == "node:zzz"


def test_narrow_mode_preserves_the_v5_5_channel_caps() -> None:
    """H2-H6 must keep the frozen V5.5 seeding shape.

    The wide reservoir is a new rung on the ablation ladder, not a change to the
    existing ones; if H6 moves, nothing measured against it can be attributed.
    """
    turns = [_turn("s1", index, f"volunteer shift {index}") for index in range(40)]
    store = _StubStore(ranking=[(turn.turn_id, float(100 - index))
                                for index, turn in enumerate(turns)])
    ir = _ir("volunteer", [_operand("only", "", "volunteer")])

    narrow = seed_operands(store, _StubView(), "m", ir, turns, dense_search=None, use_rrf=True,
                           use_postings=False, wide_reservoir=False)
    wide = seed_operands(store, _StubView(), "m", ir, turns, dense_search=None, use_rrf=True,
                         use_postings=False, wide_reservoir=True)

    assert narrow.stats["mode"] == "narrow_v5_5"
    # 24 fused turns plus up to 8 local turns from each of at most four sessions.
    assert len(narrow.source_turn_ids) <= 24 + 4 * 8
    assert len(wide.source_turn_ids) > len(narrow.source_turn_ids)
    assert set(narrow.source_turn_ids) <= set(wide.source_turn_ids)
