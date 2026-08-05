from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller
from graphmem.config import GraphMemV5Config
from graphmem.domain import NodeType, QueryBudget, RelationType
from graphmem.retrieval import GraphDiagnosticProbe

from test_v5_gate_b_core import _store


class FakeCompletions:
    def __init__(self) -> None: self.calls = 0

    def create(self, **request):
        self.calls += 1
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
