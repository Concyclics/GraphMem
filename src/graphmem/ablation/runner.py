from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..build import GraphBuildPipeline, Qwen30BRefiner
from ..config import GraphMemV5Config, config_from_dict, config_hash
from ..domain import RunManifest, canonical_json, dataclass_dict, stable_id
from ..eval import (
    DevQuestion,
    aggregate_metrics,
    calibration40,
    ingest_questions,
    load_dev_questions,
    load_gold_turns,
    navigation_metrics,
)
from ..embedding import QwenEmbeddingIndex
from ..manifests import combined_dataset_hash, create_run_manifest
from ..retrieval import GraphNavigator, NavigatorVariant
from ..storage import SQLiteGraphStore


class GateBAblationRunner:
    """Reproducible N0-N5/B0-B6 funnel; answers and gold stay offline here."""

    def __init__(self, *, repo: Path, artifact_root: Path,
                 lme_path: Path, locomo_path: Path, gold_path: Path,
                 config: GraphMemV5Config, enable_llm: bool = False,
                 enable_embedding: bool = False,
                 embedding_cache_db: Path | None = None,
                 profiles: Sequence[str] = ("b0", "b1", "b2", "b3", "b4", "b5"),
                 forced_navigator: NavigatorVariant | str | None = None) -> None:
        self.repo = repo
        self.artifact_root = artifact_root
        self.lme_path = lme_path
        self.locomo_path = locomo_path
        self.gold_path = gold_path
        self.config = config
        self.enable_llm = enable_llm
        self.enable_embedding = enable_embedding
        self.embedding_cache_db = embedding_cache_db
        self.profiles = tuple(profiles)
        self.forced_navigator = NavigatorVariant(forced_navigator) if forced_navigator else None

    def run(self, matrix: Path, dataset: Path) -> RunManifest:
        matrix_payload = json.loads(matrix.read_text(encoding="utf-8"))
        dataset_payload = json.loads(dataset.read_text(encoding="utf-8"))
        config = config_from_dict(matrix_payload.get("config", {}))
        delegate = type(self)(
            repo=self.repo, artifact_root=self.artifact_root,
            lme_path=Path(dataset_payload["longmemeval"]),
            locomo_path=Path(dataset_payload["locomo"]),
            gold_path=Path(dataset_payload["longmemeval_gold_turns"]),
            config=config, enable_llm=bool(matrix_payload.get("enable_llm", False)),
            enable_embedding=bool(matrix_payload.get("enable_embedding", False)),
            embedding_cache_db=Path(matrix_payload["embedding_cache_db"])
            if matrix_payload.get("embedding_cache_db") else None,
            profiles=tuple(matrix_payload.get("profiles", ("b0", "b1", "b2", "b3", "b4", "b5"))),
            forced_navigator=matrix_payload.get("forced_navigator"),
        )
        return delegate.run_funnel(full_confirm=bool(matrix_payload.get("full_confirm", False)))

    def run_funnel(self, *, full_confirm: bool = False) -> RunManifest:
        questions = load_dev_questions(
            self.lme_path, self.locomo_path, load_gold_turns(self.gold_path)
        )
        selected = questions if full_confirm else calibration40(questions, self.config.random_seed)
        dataset_hash = combined_dataset_hash([self.lme_path, self.locomo_path, self.gold_path])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = self.artifact_root / f"gate_b_{'full200' if full_confirm else 'cal40'}_{stamp}"
        if run_root.exists():
            raise FileExistsError(f"refusing to overwrite run directory: {run_root}")
        run_root.mkdir(parents=True)
        store = SQLiteGraphStore(run_root / "graphmem.sqlite")
        memory_ids = ingest_questions(store, selected)
        embedding_index = QwenEmbeddingIndex(store, self.config) if self.enable_embedding else None
        if embedding_index:
            source_cache = SQLiteGraphStore(self.embedding_cache_db) if self.embedding_cache_db else None
            for memory_id in memory_ids:
                if source_cache:
                    embedding_index.import_memory(memory_id, source_cache)
                embedding_index.index_memory(memory_id)
            if source_cache:
                source_cache.close()
        refiner = Qwen30BRefiner(store, self.config, dataset_hash) if self.enable_llm else None
        vector_provider = (embedding_index.embed_graph_nodes
                           if embedding_index is not None else None)
        builder = GraphBuildPipeline(
            store, dataset_hash=dataset_hash, refiner=refiner,
            coarsen_vector_provider=vector_provider,
            relation_vector_provider=vector_provider)
        all_rows: list[dict[str, Any]] = []
        nav_rows: list[dict[str, Any]] = []
        graph_manifests: list[dict[str, Any]] = []
        summary: dict[str, Any] = {"navigation": {}, "build": {}}

        # Navigator funnel uses the richest deterministic graph so navigation is
        # selected before construction changes are compared.
        if self.forced_navigator:
            best_nav = self.forced_navigator
            summary["navigation"]["selection"] = {
                "forced_from_calibration": str(best_nav), "calibration_seed": self.config.random_seed,
            }
        else:
            nav_profile = replace(
                self.config, profile="b5",
                edges=replace(self.config.edges, refine_mode="none"),
            )
            for memory_id in memory_ids:
                graph_manifests.append(dataclass_dict(builder.build(memory_id, nav_profile)))
            best_nav = NavigatorVariant.N1_RAW_FUSION
            best_nav_score = -1.0
            for variant in (
                NavigatorVariant.N1_RAW_FUSION, NavigatorVariant.N2_PROVENANCE,
                NavigatorVariant.N3_PRIORITY, NavigatorVariant.N4_CERTIFICATE,
                NavigatorVariant.N5_SET_COVER,
            ):
                metrics, results = self._evaluate(store, selected, variant, "nav:" + str(variant), embedding_index)
                aggregate = aggregate_metrics(metrics)
                summary["navigation"][str(variant)] = aggregate
                all_rows.extend(metrics)
                nav_rows.extend(results)
                score = float(aggregate["equal_stratum_turn_all_hit"])
                if score > best_nav_score or (score == best_nav_score and str(variant) > str(best_nav)):
                    best_nav, best_nav_score = variant, score

        for profile_name in self.profiles:
            if profile_name not in {"b0", "b1", "b2", "b3", "b4", "b5"}:
                raise ValueError(f"unsupported ablation profile: {profile_name}")
            profile = replace(self.config, profile=profile_name)
            if profile_name == "b4" and not self.enable_llm:
                profile = replace(profile, edges=replace(profile.edges, refine_mode="none"))
            current_manifests = []
            for memory_id in memory_ids:
                manifest = builder.build(memory_id, profile)
                payload = dataclass_dict(manifest)
                payload["ablation_profile"] = profile_name
                current_manifests.append(payload)
                graph_manifests.append(payload)
            metrics, results = self._evaluate(store, selected, best_nav, "build:" + profile_name, embedding_index)
            aggregate = aggregate_metrics(metrics)
            aggregate["build_tokens"] = sum(
                int(row["build_token_usage"].get("total_tokens", 0)) for row in current_manifests
            )
            aggregate["uncached_build_tokens"] = sum(
                int(row["build_token_usage"].get("uncached_input_tokens", 0))
                + int(row["build_token_usage"].get("output_tokens", 0))
                for row in current_manifests
            )
            aggregate["cached_build_tokens"] = sum(
                int(row["build_token_usage"].get("cached_input_tokens", 0))
                for row in current_manifests
            )
            aggregate["node_count"] = sum(int(row["node_count"]) for row in current_manifests)
            aggregate["edge_count"] = sum(int(row["edge_count"]) for row in current_manifests)
            summary["build"][profile_name] = aggregate
            all_rows.extend(metrics)
            nav_rows.extend(results)

        if "b4" in summary["build"] and "b5" in summary["build"]:
            refine_clean_tokens = int(summary["build"]["b4"]["uncached_build_tokens"])
            summary["build"]["b4"]["clean_build_backbone_tokens"] = refine_clean_tokens
            summary["build"]["b5"]["clean_build_backbone_tokens"] = refine_clean_tokens

        reference = self._legacy_reference(selected)
        if reference:
            summary["build"]["b6"] = reference

        manifest = create_run_manifest(
            repo=self.repo, dataset_paths=[self.lme_path, self.locomo_path, self.gold_path],
            config=self.config,
            graph_artifact_ids=[row["graph_artifact_id"] for row in graph_manifests],
            prompt_hashes={"selective_refine": refiner.prompt_hash if refiner else "disabled"},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write_outputs(run_root, manifest, graph_manifests, nav_rows, all_rows, summary,
                            selected, best_nav, store)
        store.close()
        return manifest

    def _evaluate(self, store: SQLiteGraphStore, questions: Sequence[DevQuestion],
                  variant: NavigatorVariant, experiment: str,
                  embedding_index: QwenEmbeddingIndex | None = None):
        navigator = GraphNavigator(
            store, variant=variant,
            dense_search=embedding_index.search if embedding_index else None,
        )
        metrics, results = [], []
        for question in questions:
            result = navigator.navigate(question.memory_id, question.query, self.config.query_budget)
            metric = navigation_metrics(question, result, store)
            metric.update({"experiment": experiment, "navigator": str(variant)})
            metrics.append(metric)
            payload = dataclass_dict(result)
            payload.update({"experiment": experiment, "original_question_id": question.question_id,
                            "stratum": question.stratum})
            results.append(payload)
        return metrics, results

    def _legacy_reference(self, selected: Sequence[DevQuestion]) -> dict[str, Any] | None:
        path = self.repo.parent / "artifacts/v5/gate_a_qwen30_20260804/research/research_summary.json"
        per_question_path = path.parent / "question_research_log.jsonl"
        if not path.exists() or not per_question_path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected_ids = {row.question_id for row in selected}
        rows = [json.loads(line) for line in per_question_path.read_text(encoding="utf-8").splitlines()
                if line and json.loads(line).get("question_id") in selected_ids]
        by_stratum = {}
        for stratum in ("lme_multi_session", "lme_temporal", "locomo_multihop", "locomo_temporal"):
            subset = [row for row in rows if row["stratum"] == stratum]
            by_stratum[stratum] = {
                "questions": len(subset),
                "turn_all_hit": sum(bool(row["turn_all_hit"]) for row in subset) / max(1, len(subset)),
                "turn_any_hit": sum(bool(row["turn_any_hit"]) for row in subset) / max(1, len(subset)),
            }
        values = [float(by_stratum[key]["turn_all_hit"]) for key in by_stratum]
        return {
            "source": str(per_question_path), "questions": len(rows),
            "equal_stratum_turn_all_hit": sum(values) / 4, "strata": by_stratum,
            "build_tokens": payload["llm_tokens"], "reasoning_tokens": payload["reasoning_tokens"],
        }

    def _write_outputs(self, root, manifest, graphs, navigation, metrics, summary,
                       questions, best_nav, store):
        run_payload = dataclass_dict(manifest)
        run_payload.update({
            "questions": len(questions), "unique_question_ids": len({q.question_id for q in questions}),
            "selected_navigator": str(best_nav), "reasoning_tokens": self._reasoning_tokens(store),
            "embedding_enabled": self.enable_embedding,
            "config_hash_verified": config_hash(self.config),
        })
        (root / "run_manifest.json").write_text(json.dumps(run_payload, indent=2) + "\n")
        (root / "graph_manifest.json").write_text(json.dumps(graphs, indent=2) + "\n")
        self._jsonl(root / "navigation_results.jsonl", navigation)
        self._jsonl(root / "error_cases.jsonl", [row for row in metrics if not row["turn_all_hit"]])
        self._csv(root / "metrics.csv", metrics)
        try:
            import pandas as pd
            pd.DataFrame(metrics).to_parquet(root / "metrics.parquet", index=False)
        except (ImportError, ValueError):
            (root / "metrics.parquet.unavailable.txt").write_text(
                "Install pandas and pyarrow to emit parquet. metrics.csv is authoritative.\n"
            )
        token_rows = [dict(row) for row in store._connection.execute(
            "SELECT memory_id,stage,cached,usage_json,latency_ms,retry_count,batch_size,prompt_hash FROM llm_calls"
        )]
        for row in token_rows:
            usage = json.loads(row.pop("usage_json"))
            row.update(usage)
        self._csv(root / "token_usage.csv", token_rows, fallback_fields=(
            "memory_id", "stage", "cached", "cached_input_tokens", "uncached_input_tokens",
            "output_tokens", "reasoning_tokens", "total_tokens", "latency_ms", "retry_count",
            "batch_size", "prompt_hash",
        ))
        with gzip.open(root / "llm_calls.jsonl.gz", "wt", encoding="utf-8") as handle:
            for row in store._connection.execute("SELECT * FROM llm_calls ORDER BY created_at,call_id"):
                handle.write(canonical_json(dict(row)) + "\n")
        with gzip.open(root / "graph_snapshot.jsonl.gz", "wt", encoding="utf-8") as handle:
            for memory_id in sorted({row["memory_id"] for row in graphs}):
                for node in store.nodes(memory_id):
                    handle.write(canonical_json({"record_type": "node", **dataclass_dict(node)}) + "\n")
                for edge in store.edges(memory_id):
                    handle.write(canonical_json({"record_type": "edge", **dataclass_dict(edge)}) + "\n")
                for group in store.evidence_groups(memory_id):
                    handle.write(canonical_json({"record_type": "evidence_group", **dataclass_dict(group)}) + "\n")
        embedding_rows = [dict(row) for row in store._connection.execute(
            "SELECT * FROM embedding_calls ORDER BY created_at,call_id"
        )]
        self._csv(root / "embedding_usage.csv", embedding_rows, fallback_fields=(
            "call_id", "memory_id", "model_id", "item_count", "input_tokens",
            "latency_ms", "heartbeat", "created_at",
        ))
        report = (
            "<!doctype html><meta charset='utf-8'><title>GraphMem V5 Gate B ablation</title>"
            "<h1>GraphMem V5 Gate B ablation</h1>"
            f"<p>Questions: {len(questions)}; selected navigator: {best_nav}</p>"
            "<h2>Machine-readable summary</h2><pre>" +
            json.dumps(summary, indent=2).replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
        )
        (root / "ablation_report.html").write_text(report, encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    @staticmethod
    def _reasoning_tokens(store):
        total = 0
        for row in store._connection.execute("SELECT usage_json FROM llm_calls"):
            total += int(json.loads(row[0]).get("reasoning_tokens", 0))
        return total

    @staticmethod
    def _jsonl(path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")

    @staticmethod
    def _csv(path, rows, fallback_fields=()):
        fields = sorted({key for row in rows for key in row}) or list(fallback_fields)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)
