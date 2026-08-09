from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from graphmem.build import GraphBuildPipeline, Qwen30BRefiner, RefineCandidate
from graphmem.config import GraphMemV5Config
from graphmem.domain import (
    Conversation,
    QueryBudget,
    Session,
    SourceTurn,
    logical_graph_checksum,
    stable_id,
)
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.retrieval import GraphNavigator, NavigatorVariant
from graphmem.runtime import GraphReadView, SQLiteSnapshotRuntime
from graphmem.storage import SQLiteGraphStore


def _store(path: Path) -> SQLiteGraphStore:
    store = SQLiteGraphStore(path)
    memory_id = "travel"
    sessions = [
        Session("s1", memory_id, 0, "2025-01-01", "s1h"),
        Session("s2", memory_id, 1, "2025-02-01", "s2h"),
    ]
    rows = [
        ("s1", 0, "Alice", "Bob", "I booked a train to Paris on January 3."),
        ("s1", 1, "Bob", "Alice", "The train booking sounds exciting."),
        ("s2", 0, "Alice", "Bob", "I cancelled the Paris train after the meeting."),
        ("s2", 1, "Bob", "Alice", "Now you are taking a bus instead."),
    ]
    turns = [SourceTurn(
        stable_id("turn", memory_id, session_id, index), memory_id, session_id, index,
        speaker, listener, "user" if index % 2 == 0 else "assistant", None, text,
        hashlib.sha256(text.encode()).hexdigest(),
    ) for session_id, index, speaker, listener, text in rows]
    store.ingest_conversation(
        Conversation(memory_id, "golden", memory_id, "memory-hash"), sessions, turns
    )
    return store


def test_sqlite_ingestion_is_idempotent_and_fts_is_memory_scoped(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    original = list(store.turns("travel"))
    assert store.ingest_turns(original) == 4
    assert store.turns("travel") == original
    assert store.search_turns("travel", "Paris train")
    assert not store.search_turns("missing", "Paris train")


def test_build_is_deterministic_and_read_view_has_inverse_csr(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    config = replace(GraphMemV5Config(), profile="b5")
    builder = GraphBuildPipeline(store, dataset_hash="dataset")
    first = builder.build("travel", config)
    first_nodes, first_edges = store.nodes("travel"), store.edges("travel")
    second = builder.build("travel", config)
    assert first.graph_checksum == second.graph_checksum
    assert logical_graph_checksum(first_nodes, first_edges) == second.graph_checksum
    view = GraphReadView(first_nodes, first_edges)
    assert view.forward and view.inverse
    assert {node.level for node in first_nodes if node.node_type.value == "routing_card"} == {1, 2, 3}
    for node in first_nodes:
        assert view.provenance_bitset[node.node_id]


def test_n5_navigation_respects_budget_and_closes_to_source_turns(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    config = replace(GraphMemV5Config(), profile="b5")
    GraphBuildPipeline(store, dataset_hash="dataset").build("travel", config)
    budget = QueryBudget(max_evidence_turns=3, max_evidence_tokens=60)
    result = GraphNavigator(store, variant=NavigatorVariant.N5_SET_COVER).navigate(
        "travel", "What replaced Alice's Paris train and when?", budget
    )
    source_ids = {turn.turn_id for turn in store.turns("travel")}
    assert set(result.retrieved_turn_ids) <= source_ids
    assert len(result.retrieved_turn_ids) <= budget.max_evidence_turns
    assert result.evidence_tokens <= budget.max_evidence_tokens
    assert result.certificate is not None
    assert result.trace["semantic_navigation_excludes_provenance_edges"] is True


def test_runtime_refreshes_after_graph_version_change(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    runtime = SQLiteSnapshotRuntime(store)
    builder = GraphBuildPipeline(store, dataset_hash="dataset")
    builder.build("travel", replace(GraphMemV5Config(), profile="b0"))
    first = runtime.view("travel")
    builder.build("travel", replace(GraphMemV5Config(), profile="b5"))
    second = runtime.view("travel")
    assert first is not second
    assert len(second.nodes) > len(first.nodes)


def test_snapshot_runtime_singleflights_concurrent_cold_compilation(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    GraphBuildPipeline(store, dataset_hash="dataset").build(
        "travel", replace(GraphMemV5Config(), profile="b5"))
    runtime = SQLiteSnapshotRuntime(store)
    original = store.graph_snapshot
    calls = 0

    def counted(memory_id: str):
        nonlocal calls
        calls += 1
        return original(memory_id)

    store.graph_snapshot = counted  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=8) as pool:
        views = list(pool.map(lambda _index: runtime.view("travel"), range(16)))

    assert len({id(view) for view in views}) == 1
    assert calls == 1
    assert runtime.cache_stats()["views"] == 1


def test_sqlite_read_pool_serves_concurrent_snapshot_queries(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    assert store.enable_read_pool(4) == 4

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(
            lambda _index: (len(store.turns("travel")),
                            len(store.search_turns("travel", "Paris"))),
            range(64)))

    assert rows == [(4, 2)] * 64


def test_sqlite_control_read_cannot_be_starved_by_full_read_pool(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    assert store.enable_read_pool(2) == 2
    assert store._read_pool is not None
    held = [store._read_pool.get(), store._read_pool.get()]
    started = __import__("time").perf_counter()
    try:
        assert store.graph_version("travel") == 0
    finally:
        for connection in held:
            store._read_pool.put(connection)
    assert __import__("time").perf_counter() - started < 0.5


def test_online_modules_do_not_import_gold_or_answers() -> None:
    roots = [Path("src/graphmem/build"), Path("src/graphmem/retrieval"),
             Path("src/graphmem/runtime"), Path("src/graphmem/storage")]
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "graphmem.eval" not in text
            assert "gold_turn" not in text


def test_embedding_index_is_cached_and_searches_only_memory_turns(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")

    class Embeddings:
        calls = 0

        def create(self, *, model, input):
            self.calls += 1
            data = []
            for index, text in enumerate(input):
                lowered = text.casefold()
                data.append(SimpleNamespace(
                    index=index,
                    embedding=[float("paris" in lowered), float("bus" in lowered), 0.1],
                ))
            return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(input)))

    embeddings = Embeddings()
    client = SimpleNamespace(embeddings=embeddings)
    index = QwenEmbeddingIndex(store, GraphMemV5Config(), client=client, batch_size=2)
    assert index.index_memory("travel") == 4
    assert index.index_memory("travel") == 0
    results = index.search("travel", "Paris", 2)
    assert len(results) == 2
    assert all(turn_id in {turn.turn_id for turn in store.turns("travel")} for turn_id, _ in results)


def test_refiner_value_gate_cache_and_zero_reasoning(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")

    class Completions:
        calls = 0

        def create(self, **request):
            self.calls += 1
            candidate_id = __import__("json").loads(request["messages"][1]["content"])["candidate_id"]
            content = __import__("json").dumps({
                "candidate_id": candidate_id, "decision": "portal", "confidence": 0.9
            })
            message = SimpleNamespace(content=content, reasoning_content=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], model="qwen30b",
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17),
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    refiner = Qwen30BRefiner(store, GraphMemV5Config(), "dataset", client=client)
    eligible = RefineCandidate(
        "candidate:1", "edge", "left", "right", "Alice booked Paris",
        "She later cancelled it", ("portal", "NONE"), 0.05, True, True, True,
    )
    local = replace(eligible, candidate_id="candidate:2", cross_scene=False,
                    cross_session=False, affects_portal=False)
    first, truncated = refiner.refine("travel", [eligible, local])
    second, _ = refiner.refine("travel", [eligible])
    assert not truncated
    assert first[0].decision == second[0].decision == "portal"
    assert completions.calls == 1
    usages = [__import__("json").loads(row[0]) for row in store._connection.execute(
        "SELECT usage_json FROM llm_calls"
    )]
    assert all(row["reasoning_tokens"] == 0 for row in usages)
