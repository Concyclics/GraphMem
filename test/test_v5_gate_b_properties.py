from __future__ import annotations

import hashlib
import itertools
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from graphmem.build import GraphBuildPipeline
from graphmem.config import GraphMemV5Config
from graphmem.domain import Conversation, QueryBudget, Session, SourceTurn, stable_id
from graphmem.retrieval import GraphNavigator, NavigatorVariant
from graphmem.storage import SQLiteGraphStore


ROWS = (
    ("s1", 0, "Alice", "Bob", "I booked a train to Paris on January 3."),
    ("s1", 1, "Bob", "Alice", "The booking is confirmed."),
    ("s2", 0, "Alice", "Bob", "I cancelled the train after the meeting."),
    ("s2", 1, "Bob", "Alice", "You are taking a bus instead."),
)


def make_store(path: Path, order: tuple[int, ...] = (0, 1, 2, 3)) -> SQLiteGraphStore:
    store = SQLiteGraphStore(path)
    memory_id = "property-memory"
    sessions = [Session("s1", memory_id, 0, None, "s1"),
                Session("s2", memory_id, 1, None, "s2")]
    turns = []
    for position in order:
        session_id, index, speaker, listener, text = ROWS[position]
        turns.append(SourceTurn(
            stable_id("turn", memory_id, session_id, index), memory_id, session_id, index,
            speaker, listener, "user" if index == 0 else "assistant", None, text,
            hashlib.sha256(text.encode()).hexdigest(),
        ))
    store.ingest_conversation(
        Conversation(memory_id, "property", memory_id, "hash"), sessions, turns
    )
    return store


@pytest.mark.parametrize("order", tuple(itertools.permutations((0, 1, 2, 3))))
def test_rebuild_checksum_is_independent_of_turn_insertion_order(order: tuple[int, ...]) -> None:
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        first = make_store(Path(first_dir) / "graph.sqlite")
        second = make_store(Path(second_dir) / "graph.sqlite", order)
        config = replace(GraphMemV5Config(), profile="b5")
        left = GraphBuildPipeline(first, dataset_hash="dataset").build("property-memory", config)
        right = GraphBuildPipeline(second, dataset_hash="dataset").build("property-memory", config)
        assert left.graph_checksum == right.graph_checksum
        first.close(); second.close()


@pytest.mark.parametrize(
    "max_nodes,max_edges,max_turns",
    ((1, 1, 1), (2, 3, 2), (4, 6, 3), (8, 12, 4), (12, 24, 2), (24, 48, 4)),
)
def test_navigation_never_exceeds_generated_hard_budgets(
    max_nodes: int, max_edges: int, max_turns: int
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = make_store(Path(directory) / "graph.sqlite")
        GraphBuildPipeline(store, dataset_hash="dataset").build(
            "property-memory", replace(GraphMemV5Config(), profile="b5")
        )
        budget = QueryBudget(
            max_visited_nodes=max_nodes, max_visited_edges=max_edges,
            max_evidence_turns=max_turns, max_evidence_tokens=100,
        )
        result = GraphNavigator(store, variant=NavigatorVariant.N5_SET_COVER).navigate(
            "property-memory", "What replaced the Paris train?", budget
        )
        assert result.visited_nodes <= max_nodes
        assert result.visited_edges <= max_edges
        assert len(result.retrieved_turn_ids) <= max_turns
        assert result.evidence_tokens <= budget.max_evidence_tokens
        store.close()
