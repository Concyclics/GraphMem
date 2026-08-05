from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller
from graphmem.config import GraphMemV5Config
from graphmem.domain import NodeType, QueryBudget, RelationType
from graphmem.retrieval import GraphDiagnosticProbe
from graphmem.storage import SQLiteGraphStore
from graphmem.runtime import GraphReadView

from test_v5_gate_b_core import _store


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def create(self, **request):
        self.calls += 1
        self.requests.append(request)
        system = request["messages"][0]["content"]
        payload = json.loads(request["messages"][1]["content"])
        if "Compress supplied" in system:
            children = payload["children"]
            content = json.dumps({"summary": "Alice Paris travel facts", "owners": ["Alice"],
                "predicates": ["travel"], "values": ["Paris"], "scopes": ["travel"],
                "times": [], "child_postings": {"alice": [row["child_id"] for row in children]}})
        else:
            scenes = []
            for scene in payload.get("s", payload.get("scenes", [])):
                scene_id = scene.get("i", scene.get("scene_id")); turns = scene.get("r", scene.get("turns", []))
                turn = next(row for row in turns if "Paris" in row.get("t", row.get("text", "")))
                text = turn.get("t", turn.get("text")); turn_id = turn.get("i", turn.get("turn_id"))
                start = text.index("Paris")
                scenes.append({"scene_id": scene_id, "summary": "Alice travel in Paris",
                    "mentions": [], "facts": [{"owner": "Alice", "predicate": "travel",
                    "value": "Paris", "value_type": "place", "scope": "travel",
                    "polarity": "positive", "time": None, "confidence": 0.95,
                    "evidence": [{"turn_id": turn_id, "start": start, "end": start + 5}]}],
                    "unresolved": []})
            content = json.dumps({"scenes": scenes})
        message = SimpleNamespace(content=content, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], model="qwen30b",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=25, total_tokens=125))


def _semantic_config():
    base = GraphMemV5Config()
    return replace(base, profile="b5", scenes=replace(base.scenes,
        llm_semantic_extraction=True, llm_hierarchy_compression=True),
        edges=replace(base.edges, refine_mode="none", max_refine_calls_per_1000_turns=0))


def test_semantic_distillation_builds_grounded_fact_graph_and_reuses_cache(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite"); completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions)); config = _semantic_config()
    distiller = QwenSemanticDistiller(store, config, "dataset", client=client)
    builder = GraphBuildPipeline(store, dataset_hash="dataset", distiller=distiller)
    first = builder.build("travel", config); calls = completions.calls
    nodes, edges = store.nodes("travel"), store.edges("travel")
    assert {NodeType.CANONICAL_FACT, NodeType.CANONICAL_VALUE, NodeType.VIRTUAL_REGION} <= {x.node_type for x in nodes}
    assert {RelationType.HAS_FACT, RelationType.FACT_VALUE, RelationType.SHARED_VALUE} <= {x.relation for x in edges}
    assert RelationType.PORTAL not in {x.relation for x in edges}
    second = builder.build("travel", config)
    assert first.graph_checksum == second.graph_checksum
    assert completions.calls == calls
    assert second.build_token_usage["reasoning_tokens"] == 0
    for node in nodes:
        assert node.evidence_group_id


def test_semantic_parser_accepts_concatenated_objects() -> None:
    rows = QwenSemanticDistiller._parse_objects('{"a":1},{"b":2}\n{"c":3}')
    assert rows == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_constrained_semantic_profile_requests_json_and_changes_cache_identity(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite"); completions = FakeCompletions()
    base = _semantic_config()
    config = replace(base, models=replace(base.models, semantic_max_facts_per_scene=4,
        semantic_summary_tokens=32, semantic_batch_output_tokens=1024,
        semantic_constrained_json=True))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    GraphBuildPipeline(store, dataset_hash="dataset",
        distiller=QwenSemanticDistiller(store, config, "dataset", client=client)).build("travel", config)
    scene_requests = [row for row in completions.requests
                      if "Extract grounded" in row["messages"][0]["content"]]
    assert scene_requests
    assert all(row["response_format"] == {"type": "json_object"} for row in scene_requests)
    assert any("at most 4 facts" in row["messages"][0]["content"] for row in scene_requests)


def test_graph_variants_remove_noisy_edges_and_add_collection_routes(tmp_path: Path) -> None:
    relations = {}
    for variant in ("g0", "g1", "g3"):
        store = _store(tmp_path / f"{variant}.sqlite"); completions = FakeCompletions()
        base = _semantic_config()
        config = replace(base, edges=replace(base.edges, graph_variant=variant),
                         coarsen=replace(base.coarsen, cross_session_merge=variant == "g0"))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        GraphBuildPipeline(store, dataset_hash="dataset",
            distiller=QwenSemanticDistiller(store, config, "dataset", client=client)).build("travel", config)
        relations[variant] = {edge.relation for edge in store.edges("travel")}
    assert RelationType.FACT_VALUE in relations["g0"]
    assert RelationType.FACT_VALUE not in relations["g1"]
    assert RelationType.SAME_ACTIVITY not in relations["g1"]
    assert RelationType.FACT_VALUE not in relations["g3"]


def test_relation_probe_is_bounded_deterministic_and_supports_ablation(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite"); completions = FakeCompletions()
    config = _semantic_config(); client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    GraphBuildPipeline(store, dataset_hash="dataset",
        distiller=QwenSemanticDistiller(store, config, "dataset", client=client)).build("travel", config)
    probe = GraphDiagnosticProbe(store); budget = QueryBudget(max_visited_nodes=20, max_visited_edges=20)
    relation = probe.run("travel", "Alice travel Paris", budget, mode="relation_only")
    shuffled_a = probe.run("travel", "Alice travel Paris", budget, mode="shuffled", shuffle_seed=7)
    shuffled_b = probe.run("travel", "Alice travel Paris", budget, mode="shuffled", shuffle_seed=7)
    without = probe.run("travel", "Alice travel Paris", budget, mode="relation_only",
                        excluded_relations=(RelationType.SHARED_VALUE,))
    assert len(relation.visited_node_ids) <= budget.max_visited_nodes
    assert len(relation.proof) <= budget.max_visited_edges
    assert shuffled_a == shuffled_b
    assert all(step.relation != RelationType.SHARED_VALUE for step in without.proof)


def test_per_memory_sqlite_shard_is_isolated_and_merges_atomically(tmp_path: Path) -> None:
    authority = _store(tmp_path / "authority.sqlite")
    shard_path = tmp_path / "shards" / "travel.sqlite"
    authority.prepare_memory_shard(shard_path, "travel")
    shard = SQLiteGraphStore(shard_path)
    assert shard.conversation("travel") is not None
    assert {row.memory_id for row in shard.turns("travel")} == {"travel"}
    completions = FakeCompletions(); config = _semantic_config()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    GraphBuildPipeline(shard, dataset_hash="dataset",
        distiller=QwenSemanticDistiller(shard, config, "dataset", client=client)).build("travel", config)
    checksum = shard.graph_checksum("travel"); shard.close()

    version = authority.merge_memory_shard(shard_path, "travel")
    assert version == authority.graph_version("travel")
    assert authority.graph_checksum("travel") == checksum
    assert any(node.node_type == NodeType.CANONICAL_FACT for node in authority.nodes("travel"))
    assert authority._connection.execute(
        "SELECT count(*) FROM llm_calls WHERE memory_id='travel'"
    ).fetchone()[0] > 0


class StrictCompletions:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls = 0; self.requests = []; self.fail_first = fail_first

    def create(self, **request):
        self.calls += 1; self.requests.append(request)
        payload = json.loads(request["messages"][1]["content"])
        if self.fail_first and self.calls == 1:
            content = '{}'
        else:
            scenes = []
            for scene in payload["s"]:
                turn = next(row for row in scene["r"] if "Paris" in row["t"])
                start = turn["t"].index("Paris")
                scenes.append({"i": scene["i"], "f": [{"o": "Alice", "p": "travel to",
                    "v": "Paris", "y": "place", "g": "travel", "n": "positive",
                    "t": None, "r": [turn["i"]], "q": "Paris"}]})
            content = json.dumps({"s": scenes})
        message = SimpleNamespace(content=content, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="qwen30b", usage=SimpleNamespace(prompt_tokens=80, completion_tokens=20,
                                                    total_tokens=100))


def _strict_config(*, variant="g5", retries=1, temporal=True):
    base = _semantic_config()
    return replace(base, models=replace(base.models, semantic_extraction_mode="strict_single",
        semantic_batch_scenes=1, semantic_batch_output_tokens=768,
        semantic_max_facts_per_scene=4, semantic_max_retries=retries,
        semantic_retry_output_tokens=1024, semantic_compile_summary=True),
        scenes=replace(base.scenes, llm_hierarchy_compression=False),
        edges=replace(base.edges, graph_variant=variant, temporal_normalization=temporal),
        coarsen=replace(base.coarsen, cross_session_merge=False))


def test_strict_prompt_uses_local_aliases_schema_and_one_retry(tmp_path: Path) -> None:
    store = _store(tmp_path / "strict.sqlite"); completions = StrictCompletions(fail_first=True)
    config = _strict_config(); client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    manifest = GraphBuildPipeline(store, dataset_hash="dataset",
        distiller=QwenSemanticDistiller(store, config, "dataset", client=client)).build("travel", config)
    assert completions.calls >= 2
    first = completions.requests[0]
    assert first["response_format"]["type"] == "json_schema"
    fact_schema = first["response_format"]["json_schema"]["schema"]["properties"]["s"]["items"]["properties"]["f"]["items"]
    assert set(fact_schema["properties"]) == {"o", "p", "v", "g", "n", "r", "q"}
    payload = json.loads(first["messages"][1]["content"])
    assert all(row["i"].startswith("s") for row in payload["s"])
    assert all(turn["i"].startswith("s") for row in payload["s"] for turn in row["r"])
    assert manifest.build_diagnostics["extraction_retry_calls"] >= 1
    assert manifest.build_token_usage["reasoning_tokens"] == 0


def test_strict_value_type_is_derived_without_model_tokens() -> None:
    assert QwenSemanticDistiller._infer_value_type("$800") == "currency"
    assert QwenSemanticDistiller._infer_value_type("three options") == "text"
    assert QwenSemanticDistiller._infer_value_type("2024-05-01") == "time"


def test_g5_lean_graph_has_terminal_facts_and_no_value_nodes(tmp_path: Path) -> None:
    store = _store(tmp_path / "lean.sqlite"); completions = StrictCompletions()
    config = _strict_config(); client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    GraphBuildPipeline(store, dataset_hash="dataset",
        distiller=QwenSemanticDistiller(store, config, "dataset", client=client)).build("travel", config)
    nodes = store.nodes("travel"); edges = store.edges("travel")
    assert NodeType.CANONICAL_FACT in {node.node_type for node in nodes}
    assert NodeType.CANONICAL_VALUE not in {node.node_type for node in nodes}
    assert NodeType.EVIDENCE_GROUP_REF in {node.node_type for node in nodes}
    assert RelationType.SCENE_CONTAINS in {edge.relation for edge in edges}
    assert all(node.attributes.get("provenance_scope") == "route"
               for node in nodes if node.node_type in {
                   NodeType.ROUTING_CARD, NodeType.SCENE, NodeType.CANONICAL_ENTITY})
    view = GraphReadView(nodes, edges)
    route_ids = [node.node_id for node in nodes if node.attributes.get("provenance_scope") == "route"]
    assert not view.evidence_group_ids_for_nodes(route_ids)
    assert view.evidence_group_ids_for_nodes(route_ids, terminal_only=False)
    evidence_refs = [node for node in nodes if node.node_type == NodeType.EVIDENCE_GROUP_REF]
    assert all(node.attributes.get("provenance_scope") == "terminal" for node in evidence_refs)
    assert all(node.attributes.get("turn_id") for node in evidence_refs)
    assert all(len(node.summary.split()) <= 13 for node in evidence_refs)
