from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller
from graphmem.build.semantic import SemanticFact, frozen_semantic_request_fingerprint
from graphmem.config import GraphMemV5Config
from graphmem.domain import NodeType, QueryBudget, RelationType
from graphmem.retrieval import GraphDiagnosticProbe
from graphmem.storage import SQLiteGraphStore
from graphmem.runtime import GraphReadView

from test_v5_gate_b_core import _store


class _NoopCallStore:
    def cache_get(self, _key):
        return None

    def cache_put(self, *_args, **_kwargs):
        return None

    def _read_one(self, *_args, **_kwargs):
        return (0,)

    def log_llm_call(self, **_kwargs):
        return None


class _ConcurrencyProbeCompletions:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def create(self, **_request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        message = SimpleNamespace(content='{"s": []}', reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="qwen30b",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )


def test_process_wide_request_gate_caps_multiple_distillers() -> None:
    gate = threading.BoundedSemaphore(2)
    completions = _ConcurrencyProbeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = GraphMemV5Config()
    distillers = [
        QwenSemanticDistiller(
            _NoopCallStore(), config, "dataset", client=client,
            request_gate=gate, worker_limit=1)
        for _ in range(8)
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda row: row[1]._call(
                f"memory-{row[0]}", "probe", "system", {"i": row[0]}, 1),
            enumerate(distillers),
        ))

    assert completions.max_active == 2


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


def test_frozen_semantic_cache_replays_dynamic_fact_cap_without_api(tmp_path: Path) -> None:
    store = _store(tmp_path / "frozen.sqlite")
    class _ReplaySeedCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            message = SimpleNamespace(
                content=json.dumps({"s": []}), reasoning_content=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                model="qwen30b",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            )

    completions = _ReplaySeedCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = GraphMemV5Config()
    first = QwenSemanticDistiller(store, config, "dataset", client=client)
    system4 = "contract. This call's schema permits at most 4 facts per scene; obey each scene's k."
    payload4 = {"s": [{"i": "0", "k": 4, "r": []}]}
    response, _ = first._call(
        "travel", "scene_semantic", system4, payload4, 1, max_facts=4)
    assert completions.calls == 1

    class _ForbiddenCompletions:
        def create(self, **_request):
            raise AssertionError("frozen replay attempted an external API call")

    frozen = QwenSemanticDistiller(
        store, config, "dataset",
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=_ForbiddenCompletions())),
        frozen_cache_only=True)
    system8 = "contract. This call's schema permits at most 8 facts per scene; obey each scene's k."
    payload8 = {"s": [{"i": "0", "k": 8, "r": []}]}
    replayed, usage = frozen._call(
        "travel", "scene_semantic", system8, payload8, 1, max_facts=8)

    assert replayed == response
    assert completions.calls == 1
    assert usage["uncached_input_tokens"] == 0
    assert store._read_one(
        "SELECT cached FROM llm_calls ORDER BY created_at DESC, rowid DESC LIMIT 1"
    )[0] == 1


def test_frozen_semantic_fingerprint_keeps_source_text_but_ignores_fact_cap() -> None:
    def request(cap: int, text: str) -> dict:
        return {"model": "qwen", "messages": [
            {"content": (
                "contract. This call's schema permits at most "
                f"{cap} facts per scene; obey each scene's k.")},
            {"content": json.dumps({
                "s": [{"i": "0", "k": cap, "r": [{"t": text}]}]})},
        ]}

    assert frozen_semantic_request_fingerprint(
        "scene_semantic", request(4, "alpha")) == frozen_semantic_request_fingerprint(
            "scene_semantic", request(8, "alpha"))
    assert frozen_semantic_request_fingerprint(
        "scene_semantic", request(4, "alpha")) != frozen_semantic_request_fingerprint(
        "scene_semantic", request(4, "beta"))


def test_frozen_semantic_cache_disambiguates_dynamic_fact_caps(tmp_path: Path) -> None:
    """Same source at different cold-build caps is not response ambiguity."""
    store = _store(tmp_path / "frozen-caps.sqlite")
    config = GraphMemV5Config()

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            content = '{"s":[{"i":"0","f":[{"o":"u","p":"has","v":"x"}]}]}'
            message = SimpleNamespace(content=content, reasoning_content=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=self.calls)
            return SimpleNamespace(choices=[choice], usage=usage, model="qwen")

    completions = _Completions()
    live = QwenSemanticDistiller(
        store, config, "dataset",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    for cap in (1, 2):
        live._call(
            "m", "scene_semantic",
            f"contract. This call's schema permits at most {cap} facts per scene; obey each scene's k.",
            {"s": [{"i": "0", "k": cap, "r": [{"t": "same source"}]}]},
            1, max_facts=cap)

    frozen = QwenSemanticDistiller(
        store, config, "dataset",
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_request: (_ for _ in ()).throw(
                AssertionError("external call"))))),
        frozen_cache_only=True)
    response, usage = frozen._call(
        "m", "scene_semantic",
        "contract. This call's schema permits at most 3 facts per scene; obey each scene's k.",
        {"s": [{"i": "0", "k": 3, "r": [{"t": "same source"}]}]},
        1, max_facts=3)

    assert response["content"]
    assert usage["uncached_input_tokens"] == 0
    assert completions.calls == 2


def test_frozen_semantic_replays_full_budget_fallback_without_api(tmp_path: Path) -> None:
    store = _store(tmp_path / "frozen-fallback.sqlite")

    class _ForbiddenCompletions:
        def create(self, **_request):
            raise AssertionError("frozen budget fallback attempted an API call")

    distiller = QwenSemanticDistiller(
        store, GraphMemV5Config(), "dataset",
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=_ForbiddenCompletions())),
        frozen_cache_only=True, frozen_fallback_calls=1)
    response, usage = distiller._call(
        "travel", "scene_semantic", "contract",
        {"s": [{"i": "0", "k": 8, "r": [{"t": "not cached"}]}]},
        1, max_facts=8)

    assert response["model"] == "frozen_full_budget_fallback"
    assert response["content"] == '{"s":[]}'
    assert usage["total_tokens"] == 0
    assert store._read_one(
        "SELECT cached FROM llm_calls ORDER BY created_at DESC, rowid DESC LIMIT 1"
    )[0] == 1


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
                index, turn = next((i, row) for i, row in enumerate(scene["r"]) if "Paris" in row["t"])
                start = turn["t"].index("Paris")
                # `p` is the bare relation the prompt asks for -- the build no
                # longer rewrites "travel to" into "visit" through a verb list.
                scenes.append({"i": scene["i"], "f": [{"o": "Alice", "p": "visit",
                    "v": "Paris", "y": "place", "g": "travel", "n": "positive",
                    "t": None, "r": [index], "q": "Paris",
                    }]})
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
        semantic_retry_output_tokens=1024, semantic_compile_summary=True,
),
        scenes=replace(base.scenes, llm_hierarchy_compression=False),
        edges=replace(base.edges, graph_variant=variant, temporal_normalization=temporal),
        coarsen=replace(base.coarsen, cross_session_merge=False))


def test_strict_prompt_exposes_only_role_time_text_and_one_retry(tmp_path: Path) -> None:
    store = _store(tmp_path / "strict.sqlite"); completions = StrictCompletions(fail_first=True)
    config = _strict_config(); client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    manifest = GraphBuildPipeline(store, dataset_hash="dataset",
        distiller=QwenSemanticDistiller(store, config, "dataset", client=client)).build("travel", config)
    assert completions.calls >= 2
    first = completions.requests[0]
    assert first["response_format"]["type"] == "json_schema"
    fact_schema = first["response_format"]["json_schema"]["schema"]["properties"]["s"]["items"]["properties"]["f"]["items"]
    assert set(fact_schema["properties"]) == {"o", "p", "v", "g", "n", "r", "q"}
    scene_array = first["response_format"]["json_schema"]["schema"]["properties"]["s"]
    assert scene_array["minItems"] == scene_array["maxItems"] == 1
    payload = json.loads(first["messages"][1]["content"])
    # Scene ids are array positions now, not "s0"-style labels the model could copy.
    assert [row["i"] for row in payload["s"]] == [str(i) for i in range(len(payload["s"]))]
    # A turn shows the model only who spoke, when, and what was said.
    assert all(set(turn) == {"s", "d", "t"} for row in payload["s"] for turn in row["r"])
    assert manifest.build_diagnostics["extraction_retry_calls"] >= 1
    assert manifest.build_token_usage["reasoning_tokens"] == 0


def test_strict_rows_recovers_duplicate_scene_alias_by_position() -> None:
    distiller = QwenSemanticDistiller.__new__(QwenSemanticDistiller)
    response = {"finish_reason": "stop", "content": json.dumps({"s": [
        {"i": "s0", "f": []}, {"i": "s0", "f": []},
    ]})}
    rows = distiller._strict_rows(
        response, {"s0": "scene-a", "s1": "scene-b"}, {})
    assert [row["i"] for row in rows] == ["scene-a", "scene-b"]


def test_strict_rows_salvages_complete_scene_from_truncated_root() -> None:
    distiller = QwenSemanticDistiller.__new__(QwenSemanticDistiller)
    content = '{"s":[{"i":"s1","f":[]} , {"i":"s0","f":['
    rows = distiller._strict_rows(
        {"finish_reason": "length", "content": content},
        {"s0": "scene-a", "s1": "scene-b"}, {})
    assert [row["i"] for row in rows] == ["scene-b"]


def test_strict_value_type_is_derived_without_model_tokens() -> None:
    assert QwenSemanticDistiller._infer_value_type("$800") == "currency"
    assert QwenSemanticDistiller._infer_value_type("three options") == "text"
    assert QwenSemanticDistiller._infer_value_type("2024-05-01") == "time"


def _semantic_fact(**overrides):
    fields = {"owner": "user", "predicate": "won_tournament", "value": "Valorant final",
              "value_type": "text", "scope": "competition", "polarity": "positive",
              "time": None, "confidence": 1.0, "evidence": ()}
    fields.update(overrides)
    return SemanticFact(**fields)


def test_v58_scene_summary_and_entities_enter_the_schema() -> None:
    from graphmem.build.semantic import strict_scene_schema

    off = strict_scene_schema(1, 4)["properties"]["s"]["items"]
    on = strict_scene_schema(1, 4, scene_summary_chars=160, scene_entities=True)
    scene = on["properties"]["s"]["items"]

    assert "m" not in off["properties"] and "e" not in off["properties"]
    assert scene["properties"]["m"]["maxLength"] == 160
    assert "m" in scene["required"] and "e" in scene["required"]
    assert scene["properties"]["e"]["items"]["maxLength"] == 40


def test_v58_a_real_summary_survives_compile_summary() -> None:
    """The sentence must not be overwritten by the fact-triple concatenation.

    `semantic_compile_summary` replaced the model's summary with
    "owner predicate value owner predicate value ...", which is what routing
    cards were built from and why they read as duplicated term soup.
    """
    from graphmem.build.semantic import QwenSemanticDistiller

    base = _strict_config()
    config = replace(base, models=replace(
        base.models, semantic_compile_summary=True, semantic_scene_summary_chars=160))
    distiller = QwenSemanticDistiller.__new__(QwenSemanticDistiller)
    distiller.config = config
    scene = SimpleNamespace(scene_id="s0", summary="fallback", turns=[])
    row = {"i": "s0", "f": [], "m": "Alice visited Paris in July and stayed near the Louvre.",
           "e": ["Alice", "Paris", "Louvre", "Alice"]}

    packet = distiller._validate_scene(scene, row)

    assert packet.summary == "Alice visited Paris in July and stayed near the Louvre."
    # Deduplicated, order preserved -- the same entity string must recur across
    # scenes for cross-session routing to join them.
    assert packet.entities == ("Alice", "Paris", "Louvre")


def test_v58_scene_and_turn_labels_never_survive_into_free_text() -> None:
    """The input protocol's own labels must not come back as content.

    Measured on a 20-memory build without this guard: 68.5% of scene summaries
    were the bare string "s1", and the six most frequent "entities" across the
    corpus were s1t1, s1t0, s1t2, s1t3, s0t1, s0t0 -- which recur in every
    session by construction and so faked a cross-session entity link.  `scope`
    has been filtered since V5.4 and the V5.7 category field leaked the same
    labels at 2.9%; this is the third occurrence, so it is pinned here.
    """
    from graphmem.build.semantic import QwenSemanticDistiller, strip_aliases

    for label in ("s1", "s10", "s1t0", "S0T12", "  s1  "):
        assert strip_aliases(label) == ""
    for real in ("Trello", "Patti Smith", "s1 is where Alice went"):
        assert strip_aliases(real) == " ".join(real.split())

    base = _strict_config()
    config = replace(base, models=replace(
        base.models, semantic_scene_summary_chars=160, semantic_scene_entities=True))
    distiller = QwenSemanticDistiller.__new__(QwenSemanticDistiller)
    distiller.config = config
    scene = SimpleNamespace(scene_id="s0", summary="fallback", turns=[])

    packet = distiller._validate_scene(
        scene, {"i": "s0", "f": [], "m": "s1", "e": ["s1t0", "Trello", "s0", "Patti Smith"]})

    assert packet.summary == "fallback", "an echoed label must fall through to the scene text"
    assert packet.entities == ("Trello", "Patti Smith")


def test_v58_compiled_summary_still_used_when_the_field_is_off() -> None:
    from graphmem.build.semantic import QwenSemanticDistiller

    base = _strict_config()
    config = replace(base, models=replace(
        base.models, semantic_compile_summary=True, semantic_scene_summary_chars=0))
    distiller = QwenSemanticDistiller.__new__(QwenSemanticDistiller)
    distiller.config = config
    scene = SimpleNamespace(scene_id="s0", summary="fallback", turns=[])

    packet = distiller._validate_scene(scene, {"i": "s0", "f": [], "m": "ignored"})

    assert packet.summary == "fallback" and packet.entities == ()


def test_v57_predicate_and_scope_come_from_extraction_not_regex_ladders() -> None:
    # The predicate is normalized, not rewritten: no verb list decides that
    # "won_tournament" means "win".  Extraction supplies the relation itself.
    assert GraphBuildPipeline._predicate_key(_semantic_fact()) == "won tournament"
    assert GraphBuildPipeline._predicate_key(
        _semantic_fact(predicate="recommends reading")) == "recommends reading"
    assert GraphBuildPipeline._scope_key(_semantic_fact(scope="Travel")) == "travel"
    assert GraphBuildPipeline._scope_key(_semantic_fact(scope="")) == "general"
    # collection_key falls back to value_type: the seven regex families it used
    # to consult matched none of model kit, fitness class or film festival, and
    # the extraction-supplied category that replaced them measured 390 distinct
    # values per memory at 6.3% reuse, so that route is closed.
    assert GraphBuildPipeline._collection_key(_semantic_fact(value_type="time")) == "time"
    assert GraphBuildPipeline._collection_key(_semantic_fact(value_type="")) == "value"


def test_v54_modality_rules_survive_the_ladder_removal() -> None:
    assert GraphBuildPipeline._fact_modality("It is definitely on my to-do list") == "planned"
    assert GraphBuildPipeline._fact_modality("I visited Chicago last week") == "asserted"


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
