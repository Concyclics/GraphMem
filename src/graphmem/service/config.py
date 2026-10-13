"""Portable product configuration; packaged profiles never reference experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from importlib.resources import files
import json
from pathlib import Path
from typing import Any, Mapping

from ..config import config_from_dict, runtime_config_from_dict


@dataclass(frozen=True)
class ServiceConfig:
    data_dir: str = "./data/graphmem"
    semantic_build: bool = True
    dense_retrieval: bool = True
    generate_answers: bool = False
    require_exact_tokenizer: bool = True
    llm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    llm_base_url: str = "http://127.0.0.1:8002/v1"
    llm_api_key_env: str = ""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_api_key_env: str = ""
    request_profile: str = "qwen"
    answer_model: str = ""
    answer_base_url: str = ""
    answer_api_key_env: str = ""
    answer_request_profile: str = "qwen"
    answer_reasoning_effort: str = "none"
    max_output_tokens: int = 2000
    answer_frequency_penalty: float = 0.2
    model_timeout_seconds: float = 180.0
    readout: str = "v5_81"
    build_overrides: Mapping[str, Any] = field(default_factory=dict)
    runtime_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, str) or not self.data_dir.strip() or self.data_dir == ":memory:":
            raise ValueError("data_dir must name a persistent directory")
        for name in ("semantic_build", "dense_retrieval", "generate_answers", "require_exact_tokenizer"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.request_profile not in {"qwen", "openai"}:
            raise ValueError("request_profile must be qwen or openai")
        if self.answer_request_profile not in {"qwen", "openai", "omit"}:
            raise ValueError("invalid answer_request_profile")
        if self.answer_reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("invalid answer_reasoning_effort")
        if self.answer_reasoning_effort != "none" and self.answer_request_profile != "openai":
            raise ValueError("reasoning requires answer_request_profile=openai")
        if self.readout not in {"v5_81", "v5_73"}:
            raise ValueError("readout must be v5_81 or v5_73")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if not 0.0 <= self.answer_frequency_penalty <= 2.0:
            raise ValueError("answer_frequency_penalty must be between 0 and 2")
        if not 0 < self.model_timeout_seconds <= 600:
            raise ValueError("model_timeout_seconds must be in (0, 600]")
        for name in ("llm_model", "embedding_model", "llm_base_url", "embedding_base_url"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")

    @classmethod
    def offline(cls, data_dir: str | Path) -> ServiceConfig:
        """Local persistence/API smoke profile, without model calls or accuracy claims."""
        return cls(data_dir=str(data_dir), semantic_build=False,
                   dense_retrieval=False, require_exact_tokenizer=False)

    @classmethod
    def from_file(cls, path: str | Path) -> ServiceConfig:
        path = Path(path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("service config must be a JSON object")
        config = cls(**payload)
        directory = Path(config.data_dir).expanduser()
        if not directory.is_absolute():
            directory = path.parent / directory
        return replace(config, data_dir=str(directory.resolve()))

    def profiles(self):
        root = files("graphmem.service").joinpath("profiles")
        build = asdict(config_from_dict(json.loads(root.joinpath("build_v5_81.json").read_text())))
        runtime = asdict(runtime_config_from_dict(json.loads(root.joinpath("runtime_v5_81.json").read_text())))
        _merge(build, self.build_overrides)
        _merge(runtime, self.runtime_overrides)
        directory = Path(self.data_dir).expanduser().resolve()
        build["storage"].update(sqlite_path=str(directory / "memory.sqlite"))
        build["models"].update(llm_model=self.llm_model, llm_base_url=self.llm_base_url,
                               embedding_model=self.embedding_model,
                               embedding_base_url=self.embedding_base_url)
        build["scenes"]["llm_semantic_extraction"] = self.semantic_build
        if not self.semantic_build:
            build["edges"]["graph_variant"] = "g0"
            build["projection_profile"] = "P0"
        if not self.dense_retrieval:
            build["coarsen"]["assignment_method"] = "bounded_semantic_partition"
            build["edges"]["relation_candidate_method"] = "bounded_sparse"
            build["edges"]["atomic_relation_multiview"] = False
        runtime["retrieval"].update(
            dense_search_enabled=self.dense_retrieval,
            embedding_model=self.embedding_model, embedding_base_url=self.embedding_base_url,
            compiled_cache_dir=str(directory / "compiled"),
            dense_sidecar_dir=str(directory / "dense"),
            query_embedding_cache_path=str(directory / "query_embeddings.sqlite"))
        return config_from_dict(build), runtime_config_from_dict(runtime)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(target: dict, overrides: Mapping) -> None:
    if not isinstance(overrides, Mapping):
        raise ValueError("profile overrides must be objects")
    for key, value in overrides.items():
        if key not in target:
            raise ValueError(f"unknown profile option: {key}")
        if isinstance(target[key], dict):
            _merge(target[key], value)
        else:
            target[key] = value
