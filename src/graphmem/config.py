from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .domain import QueryBudget, canonical_json


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    dataset_hash: str
    model_id: str
    prompt_hash: str
    schema_version: str
    config_hash: str
    stage: str

    def key(self) -> str:
        values = asdict(self)
        if any(not str(value).strip() for value in values.values()):
            raise ValueError("cache identity fields must all be non-empty")
        return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    llm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    llm_base_url: str = "http://127.0.0.1:8002/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    thinking_enabled: bool = False
    max_concurrency: int = 128
    refine_input_tokens_per_endpoint: int = 96
    refine_output_tokens: int = 256
    bridge_refine_output_tokens: int = 512
    semantic_batch_scenes: int = 4
    semantic_batch_input_tokens: int = 12000
    semantic_scene_input_tokens: int = 16000
    semantic_batch_output_tokens: int = 4096
    semantic_average_tokens_per_memory: int = 180000


@dataclass(frozen=True, slots=True)
class SceneConfig:
    min_turns: int = 2
    max_turns: int = 8
    topic_similarity_threshold: float = 0.55
    max_events_per_scene: int = 3
    coreference_margin: float = 0.08
    refine_batch_size: int = 24
    llm_semantic_extraction: bool = False
    llm_hierarchy_compression: bool = False


@dataclass(frozen=True, slots=True)
class CoarsenConfig:
    fanout: int = 8
    max_levels: int = 3
    summary_tokens: int = 320
    cross_session_merge: bool = True


@dataclass(frozen=True, slots=True)
class EdgeConfig:
    embedding_k: int = 8
    max_candidates_per_node: int = 24
    max_degree_per_relation: int = 12
    low_threshold: float = 0.45
    high_threshold: float = 0.78
    refine_mode: str = "ambiguous_only"
    refine_batch_size: int = 24
    max_refine_calls_per_1000_turns: int = 20
    relation_degree_caps: Mapping[str, int] = field(default_factory=lambda: {
        "same_event": 4, "same_activity": 8, "state_next": 4,
        "state_transition": 4, "temporal_before": 8, "shared_entity": 32,
        "shared_value": 32, "collection_co_member": 128,
        "has_fact": 64, "fact_value": 64, "coreference": 32,
    })


@dataclass(frozen=True, slots=True)
class StorageConfig:
    runtime_mode: str = "sqlite_snapshot"
    sqlite_path: str = "artifacts/v5/graphmem.sqlite"
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_batch_nodes: int = 1000
    neo4j_batch_edges: int = 2000


@dataclass(frozen=True, slots=True)
class GraphMemV5Config:
    schema_version: str = "graphmem-v5"
    profile: str = "full_balanced"
    random_seed: int = 42
    models: ModelConfig = field(default_factory=ModelConfig)
    scenes: SceneConfig = field(default_factory=SceneConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    coarsen: CoarsenConfig = field(default_factory=CoarsenConfig)
    edges: EdgeConfig = field(default_factory=EdgeConfig)
    query_budget: QueryBudget = field(default_factory=QueryBudget)

    def __post_init__(self) -> None:
        if self.schema_version != "graphmem-v5":
            raise ValueError("unsupported GraphMem schema version")
        if self.models.thinking_enabled:
            raise ValueError("GraphMem V5 memory backbone thinking must remain disabled")
        if self.storage.runtime_mode not in {
            "neo4j_direct", "neo4j_cached", "sqlite_snapshot"
        }:
            raise ValueError("invalid graph runtime mode")
        if self.edges.refine_mode not in {
            "none", "ambiguous_only", "high_value_only", "all_bounded_candidates"
        }:
            raise ValueError("invalid edge refine mode")
        if not 0 <= self.edges.low_threshold < self.edges.high_threshold <= 1:
            raise ValueError("edge thresholds must satisfy 0 <= low < high <= 1")
        positive = {
            "fanout": self.coarsen.fanout,
            "max_levels": self.coarsen.max_levels,
            "summary_tokens": self.coarsen.summary_tokens,
            "embedding_k": self.edges.embedding_k,
            "max_candidates_per_node": self.edges.max_candidates_per_node,
            "max_degree_per_relation": self.edges.max_degree_per_relation,
            "refine_batch_size": self.edges.refine_batch_size,
            "scene_min_turns": self.scenes.min_turns,
            "scene_max_turns": self.scenes.max_turns,
            "max_events_per_scene": self.scenes.max_events_per_scene,
            "coreference_batch_size": self.scenes.refine_batch_size,
            "max_concurrency": self.models.max_concurrency,
            "refine_input_tokens_per_endpoint": self.models.refine_input_tokens_per_endpoint,
            "refine_output_tokens": self.models.refine_output_tokens,
            "bridge_refine_output_tokens": self.models.bridge_refine_output_tokens,
            "semantic_batch_scenes": self.models.semantic_batch_scenes,
            "semantic_batch_input_tokens": self.models.semantic_batch_input_tokens,
            "semantic_scene_input_tokens": self.models.semantic_scene_input_tokens,
            "semantic_batch_output_tokens": self.models.semantic_batch_output_tokens,
            "semantic_average_tokens_per_memory": self.models.semantic_average_tokens_per_memory,
            "neo4j_batch_nodes": self.storage.neo4j_batch_nodes,
            "neo4j_batch_edges": self.storage.neo4j_batch_edges,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"positive configuration values required: {positive}")
        if self.edges.max_refine_calls_per_1000_turns < 0:
            raise ValueError("refine call limit cannot be negative")
        if self.scenes.min_turns > self.scenes.max_turns:
            raise ValueError("scene min_turns cannot exceed max_turns")
        if not 0.0 <= self.scenes.topic_similarity_threshold <= 1.0:
            raise ValueError("scene similarity threshold must be in [0, 1]")
        if not 0.0 <= self.scenes.coreference_margin <= 1.0:
            raise ValueError("coreference margin must be in [0, 1]")


def config_hash(config: GraphMemV5Config) -> str:
    return hashlib.sha256(canonical_json(asdict(config)).encode("utf-8")).hexdigest()


def _section(cls: type[Any], payload: Mapping[str, Any] | None) -> Any:
    return cls(**dict(payload or {}))


def config_from_dict(payload: Mapping[str, Any]) -> GraphMemV5Config:
    value = dict(payload)
    return GraphMemV5Config(
        schema_version=str(value.get("schema_version", "graphmem-v5")),
        profile=str(value.get("profile", "full_balanced")),
        random_seed=int(value.get("random_seed", 42)),
        models=_section(ModelConfig, value.get("models")),
        scenes=_section(SceneConfig, value.get("scenes")),
        storage=_section(StorageConfig, value.get("storage")),
        coarsen=_section(CoarsenConfig, value.get("coarsen")),
        edges=_section(EdgeConfig, value.get("edges")),
        query_budget=_section(QueryBudget, value.get("query_budget")),
    )


def load_config(path: Path) -> GraphMemV5Config:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("PyYAML is required to load YAML V5 configs") from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ValueError("V5 config root must be a mapping")
    return config_from_dict(payload)
