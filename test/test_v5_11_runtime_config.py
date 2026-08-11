from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphmem.build import GraphBuildPipeline
from graphmem.config import (
    GraphMemV5Config,
    RetrievalRuntimeConfig,
    ServingConfig,
    load_runtime_config,
    runtime_config_hash,
)
from graphmem.retrieval.compiled_memory import CompiledMemorySidecar
from graphmem.serving import sync_compiled_sidecars

from test_v5_9_coarsening import _store


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", [
    "runtime_v5_11_balanced.json",
    "runtime_v5_11_low_latency.json",
    "runtime_v5_11_low_memory.json",
    "runtime_v5_11_report_8w.json",
])
def test_runtime_profiles_load_and_translate_to_online_options(name: str) -> None:
    config = load_runtime_config(ROOT / "configs/v5" / name)
    navigator = config.retrieval.navigator_options()
    pool = config.serving.pool_options()

    assert config.schema_version == "graphmem-runtime-v5.11"
    assert len(runtime_config_hash(config)) == 64
    assert navigator["harness_profile"] == "h11"
    assert navigator["obligation_aware_packing"] is True
    assert navigator["native_seed_fusion"] is True
    assert navigator["hierarchy_descent_beam"] == 1
    assert navigator["rare_lexical_relations"] is False
    assert navigator["compiled_cache_admission"] is True
    assert pool["workers"] == config.serving.workers
    assert pool["affinity_replicas"] == config.serving.affinity_replicas
    assert config.query_budget.max_evidence_turns == 32


def test_v5_17_accuracy_profile_keeps_wider_evidence_budget() -> None:
    config = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_17_accuracy64.json")

    assert config.profile == "v5_17_accuracy64"
    assert config.query_budget.max_evidence_turns == 64
    assert config.query_budget.max_evidence_tokens == 12_000
    options = config.retrieval.navigator_options(compiled_cache_dir=None)
    assert options["obligation_aware_packing"] is True
    assert options["precision_aware_packing"] is False
    assert options["rare_lexical_relations"] is True
    assert options["queryir_soft_fallback"] is True
    assert config.retrieval.dense_search_enabled is True
    embedding = config.retrieval.embedding_options()
    assert embedding is not None
    assert embedding["dense_backend"] == "auto"
    assert embedding["query_cache_entries"] == 8192


def test_v5_54_accuracy64_profile_freezes_the_winner_query_path() -> None:
    config = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_54_accuracy64.json")

    assert config.profile == "v5_54_accuracy64"
    assert config.query_budget.max_evidence_turns == 64
    assert config.query_budget.max_evidence_tokens == 12_000
    assert config.query_budget.max_answer_tokens == 10_000
    options = config.retrieval.navigator_options(compiled_cache_dir=None)
    assert options["obligation_aware_packing"] is True
    assert options["native_seed_fusion"] is True
    assert options["relational_view_scoring"] is True
    assert options["query_relation_view"] is True
    assert options["relational_view_named_speakers_only"] is True
    assert options["speaker_owner_bonus"] == 1.0
    assert options["query_witness_bonus"] == 1.0
    assert options["queryir_soft_fallback"] is True
    # Coarse lexical graph traversal was rejected; BM25/dense seeding remains.
    assert options["rare_lexical_relations"] is False


def test_runtime_profile_matches_report_pareto_worker_configuration() -> None:
    config = load_runtime_config(
        ROOT / "configs/v5/runtime_v5_11_report_8w.json")
    assert config.serving.workers == 8
    assert config.serving.worker_cpu_ids == tuple(range(8))
    assert config.serving.affinity_replicas == 2
    assert config.serving.max_queued == 248
    assert config.serving.per_tenant_outstanding == 1
    assert config.retrieval.snapshot_cache_memories == 8
    assert config.retrieval.snapshot_cache_bytes == 256 * 1024 * 1024


def test_serving_cpu_affinity_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="one CPU ID per worker"):
        ServingConfig(workers=2, worker_cpu_ids=(0,))
    with pytest.raises(ValueError, match="must be unique"):
        ServingConfig(workers=2, worker_cpu_ids=(0, 0))


def test_speaker_owner_bonus_is_validated_and_translated() -> None:
    config = RetrievalRuntimeConfig(speaker_owner_bonus=1.0)

    assert config.navigator_options(compiled_cache_dir=None)[
        "speaker_owner_bonus"] == 1.0
    with pytest.raises(ValueError, match="speaker_owner_bonus"):
        RetrievalRuntimeConfig(speaker_owner_bonus=-0.1)


def test_query_witness_closure_is_validated_and_translated() -> None:
    config = RetrievalRuntimeConfig(
        query_witness_bonus=1.0,
        query_witness_seed_count=16,
        query_witness_rare_df=4,
        query_witness_min_shared_terms=2,
    )
    options = config.navigator_options(compiled_cache_dir=None)
    assert options["query_witness_bonus"] == 1.0
    assert options["query_witness_seed_count"] == 16
    assert options["query_witness_rare_df"] == 4
    assert options["query_witness_min_shared_terms"] == 2
    with pytest.raises(ValueError, match="query_witness_bonus"):
        RetrievalRuntimeConfig(query_witness_bonus=-0.1)
    with pytest.raises(ValueError, match="positive retrieval runtime"):
        RetrievalRuntimeConfig(query_witness_seed_count=0)


def test_sidecar_sync_compiles_once_then_uses_lightweight_publish_record(
        tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite"
    output = tmp_path / "compiled"
    store = _store(database)
    GraphBuildPipeline(store, dataset_hash="dataset").build(
        "m", GraphMemV5Config(profile="b5"))
    version, checksum = store.graph_identity("m")
    store.close()

    first = sync_compiled_sidecars(database, output, workers=1)
    assert first["compiled"] == 1 and first["current"] == 0
    assert first["failed"] == 0

    sidecar = CompiledMemorySidecar(output)
    assert sidecar.is_current("m", version, checksum)
    metadata = sidecar.metadata_path_for("m")
    record = json.loads(metadata.read_text(encoding="utf-8"))
    assert record["graph_version"] == version
    assert record["graph_checksum"] == checksum

    second = sync_compiled_sidecars(database, output, workers=1)
    assert second["compiled"] == 0 and second["current"] == 1
    assert second["failed"] == 0

    record["graph_checksum"] = "stale"
    metadata.write_text(json.dumps(record), encoding="utf-8")
    repaired = sync_compiled_sidecars(database, output, workers=1)
    assert repaired["compiled"] == 1 and repaired["failed"] == 0
    assert sidecar.is_current("m", version, checksum)
