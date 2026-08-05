#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from graphmem.build import GraphBuildPipeline, PredicateCanonicalizer, QwenSemanticDistiller
from graphmem.config import config_hash, load_config
from graphmem.domain import RelationType, dataclass_dict
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval import (aggregate_metrics, conversation_holdout_split, load_dev_questions,
                           load_gold_turns, navigation_metrics)
from graphmem.retrieval import GraphDiagnosticProbe, GraphNavigator, NavigatorVariant
from graphmem.storage import SQLiteGraphStore


def diagnostic_metric(question, result, store):
    turns = {turn.turn_id: turn for turn in store.turns(question.memory_id)}
    predicted = {(turns[item].session_id, turns[item].turn_index) for item in result.candidate_turn_ids if item in turns}
    gold = {(item.session_id, item.turn_index) for item in question.gold_turns}
    sessions = set(result.candidate_session_ids); gold_sessions = set(question.gold_sessions)
    nodes = {node.node_id: node for node in store.nodes(question.memory_id)}
    adjacency = {}
    for edge in store.edges(question.memory_id):
        adjacency.setdefault(edge.src_id, []).append(edge.dst_id)
        adjacency.setdefault(edge.dst_id, []).append(edge.src_id)
    distance = {node_id: 0 for node_id in result.seed_node_ids}; queue = deque(result.seed_node_ids)
    while queue:
        node_id = queue.popleft()
        if distance[node_id] >= 2:
            continue
        for neighbor in adjacency.get(node_id, ()):
            if neighbor not in distance:
                distance[neighbor] = distance[node_id] + 1; queue.append(neighbor)
    gold_distance = {}
    terminal_turn_ids = set()
    for node_id, node in nodes.items():
        if node.attributes.get("provenance_scope", "terminal") != "terminal":
            continue
        for group_id in node.all_evidence_group_ids:
            group = store.evidence_group(group_id)
            if not group:
                continue
            for member in group.members:
                terminal_turn_ids.add(member.turn_id)
                turn = turns.get(member.turn_id)
                key = (turn.session_id, turn.turn_index) if turn else None
                if key in gold and node_id in distance:
                    gold_distance[key] = min(gold_distance.get(key, 99), distance[node_id])
    return {"question_id": question.question_id, "stratum": question.stratum, "mode": result.mode,
            "session_all_hit": gold_sessions <= sessions,
            "session_recall": len(sessions & gold_sessions) / max(1, len(gold_sessions)),
            "turn_all_hit": gold <= predicted, "turn_recall": len(gold & predicted) / max(1, len(gold)),
            "visited_nodes": len(result.visited_node_ids), "visited_edges": len(result.proof),
            "candidate_turns": len(result.candidate_turn_ids), "relation_counts": result.relation_counts,
            "terminal_reachable_all_hit": gold <= set(gold_distance),
            "terminal_reachable_recall": len(set(gold_distance) & gold) / max(1, len(gold)),
            "gold_turn_shortest_path": (sum(gold_distance.values()) / len(gold_distance)
                                          if gold_distance else None),
            "abstract_provenance_leakage": len(set(result.candidate_turn_ids) - terminal_turn_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True); parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation", "final", "full"), default="development")
    parser.add_argument("--profiles", default="c0,c1,c2,c3,c4"); parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--memory-workers", type=int, default=1,
                        help="Build independent per-memory SQLite shards concurrently")
    parser.add_argument("--e-graph-variant", choices=("g0", "g1", "g2", "g3", "g4"), default="g3")
    parser.add_argument("--f-extraction-mode", choices=("strict_single", "strict_pair"),
                        default="strict_single")
    parser.add_argument("--stratum", choices=("lme_multi_session", "lme_temporal",
                                               "locomo_multihop", "locomo_temporal"))
    parser.add_argument("--per-stratum", type=int)
    args = parser.parse_args(); stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root / f"v5_1_graph_ablation_{args.split}_{stamp}"; root.mkdir(parents=True)
    db_path = root / "graphmem.sqlite"; shutil.copy2(args.source_db, db_path)
    store = SQLiteGraphStore(db_path); base = load_config(args.config)
    all_questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    selected = all_questions if args.split == "full" else conversation_holdout_split(all_questions, base.random_seed)[args.split]
    if args.stratum: selected = [row for row in selected if row.stratum == args.stratum]
    if args.per_stratum:
        selected = [row for name in ("lme_multi_session", "lme_temporal", "locomo_multihop", "locomo_temporal")
                    for row in [next(iter([item for item in selected if item.stratum == name][index:index+1]), None)
                                for index in range(args.per_stratum)] if row is not None]
    if args.max_questions: selected = selected[:args.max_questions]
    memory_ids = sorted({row.memory_id for row in selected}); embedding = QwenEmbeddingIndex(store, base) if args.embedding else None
    definitions = {
        "c0": replace(base, profile="b5", scenes=replace(base.scenes, llm_semantic_extraction=False,
             llm_hierarchy_compression=False), coarsen=replace(base.coarsen, cross_session_merge=True)),
        "c1": replace(base, profile="b1", scenes=replace(base.scenes, llm_semantic_extraction=True,
             llm_hierarchy_compression=False), coarsen=replace(base.coarsen, cross_session_merge=False)),
        "c2": replace(base, profile="b2", scenes=replace(base.scenes, llm_semantic_extraction=True,
             llm_hierarchy_compression=False), coarsen=replace(base.coarsen, cross_session_merge=False)),
        "c3": replace(base, profile="b5", scenes=replace(base.scenes, llm_semantic_extraction=True,
             llm_hierarchy_compression=True), coarsen=replace(base.coarsen, cross_session_merge=False)),
        "c4": replace(base, profile="b5", scenes=replace(base.scenes, llm_semantic_extraction=True,
             llm_hierarchy_compression=True), coarsen=replace(base.coarsen, cross_session_merge=True)),
    }
    for graph_variant in ("g0", "g1", "g2", "g3", "g4"):
        definitions[graph_variant] = replace(base, profile="b5",
            scenes=replace(base.scenes, llm_semantic_extraction=True, llm_hierarchy_compression=True),
            coarsen=replace(base.coarsen, cross_session_merge=graph_variant == "g4"),
            edges=replace(base.edges, graph_variant=graph_variant))
    extraction_models = {
        "e0": base.models,
        "e1": replace(base.models, semantic_max_facts_per_scene=4,
                      semantic_summary_tokens=32, semantic_batch_output_tokens=1024),
        "e2": replace(base.models, semantic_max_facts_per_scene=4,
                      semantic_summary_tokens=32, semantic_batch_output_tokens=1024,
                      semantic_constrained_json=True),
        "e3": replace(base.models, semantic_batch_scenes=2, semantic_batch_output_tokens=768,
                      semantic_max_facts_per_scene=4, semantic_summary_tokens=32,
                      semantic_repair_output_tokens=256, semantic_constrained_json=True,
                      semantic_individual_repair=True),
        "e4": replace(base.models, semantic_batch_scenes=2, semantic_batch_output_tokens=768,
                      semantic_max_facts_per_scene=4, semantic_summary_tokens=32,
                      semantic_repair_output_tokens=256, semantic_constrained_json=True,
                      semantic_individual_repair=True),
    }
    for extraction_variant, models in extraction_models.items():
        definitions[extraction_variant] = replace(base, profile="b5", models=models,
            scenes=replace(base.scenes, llm_semantic_extraction=True,
                           llm_hierarchy_compression=extraction_variant != "e4"),
            coarsen=replace(base.coarsen, cross_session_merge=args.e_graph_variant == "g4"),
            edges=replace(base.edges, graph_variant=args.e_graph_variant))
    f0_models = replace(base.models, semantic_batch_scenes=2, semantic_batch_output_tokens=768,
        semantic_max_facts_per_scene=4, semantic_summary_tokens=32,
        semantic_repair_output_tokens=256, semantic_constrained_json=True,
        semantic_individual_repair=True, semantic_extraction_mode="legacy_batch",
        semantic_max_retries=0, semantic_compile_summary=False)
    strict_single = replace(base.models, semantic_extraction_mode="strict_single",
        semantic_batch_scenes=1, semantic_batch_output_tokens=768,
        semantic_max_facts_per_scene=3, semantic_summary_tokens=48,
        semantic_max_retries=1, semantic_retry_output_tokens=1024,
        semantic_compile_summary=True)
    strict_pair = replace(strict_single, semantic_extraction_mode="strict_pair",
                          semantic_batch_scenes=2, semantic_batch_output_tokens=1024)
    selected_strict = strict_single if args.f_extraction_mode == "strict_single" else strict_pair
    definitions["f0"] = replace(base, profile="b5", models=f0_models,
        scenes=replace(base.scenes, llm_semantic_extraction=True, llm_hierarchy_compression=False),
        coarsen=replace(base.coarsen, cross_session_merge=False),
        edges=replace(base.edges, graph_variant="g3"))
    for label, models in (("f1", strict_single), ("f2", strict_pair)):
        definitions[label] = replace(base, profile="b5", models=models,
            scenes=replace(base.scenes, llm_semantic_extraction=True, llm_hierarchy_compression=False),
            coarsen=replace(base.coarsen, cross_session_merge=False),
            edges=replace(base.edges, graph_variant="g3"))
    for label, temporal, portals in (("f3", False, False), ("f4", True, False), ("f5", True, True)):
        definitions[label] = replace(base, profile="b5", models=selected_strict,
            scenes=replace(base.scenes, llm_semantic_extraction=True, llm_hierarchy_compression=False),
            coarsen=replace(base.coarsen, cross_session_merge=False),
            edges=replace(base.edges, graph_variant="g5", temporal_normalization=temporal,
                          cross_session_portals=portals))
    summary = {}; metric_rows = []; diagnostic_rows = []; trace_rows = []
    navigation_rows = []; manifests = []
    for label in (item.strip() for item in args.profiles.split(",") if item.strip()):
        profile = definitions[label]; distiller = QwenSemanticDistiller(store, profile, "v5.1-holdout") \
            if profile.scenes.llm_semantic_extraction else None
        if args.memory_workers < 1:
            raise ValueError("--memory-workers must be positive")

        def build_one(memory_id: str):
            shard_dir = root / "memory_shards" / label
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / (memory_id.replace(":", "_").replace("/", "_") + ".sqlite")
            store.prepare_memory_shard(shard_path, memory_id)
            worker_store = SQLiteGraphStore(shard_path)
            worker_distiller = QwenSemanticDistiller(
                worker_store, profile, "v5.1-holdout"
            ) if profile.scenes.llm_semantic_extraction else None
            predicate_canonicalizer = (PredicateCanonicalizer(worker_store, profile)
                                       if profile.edges.graph_variant == "g5" else None)
            try:
                manifest = GraphBuildPipeline(
                    worker_store, dataset_hash="v5.1-holdout", distiller=worker_distiller,
                    predicate_canonicalizer=predicate_canonicalizer,
                ).build(memory_id, profile)
            finally:
                worker_store.close()
            merged_version = store.merge_memory_shard(shard_path, memory_id)
            return replace(manifest, graph_version=merged_version)

        current = []
        workers = min(args.memory_workers, len(memory_ids))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_one, memory_id): memory_id for memory_id in memory_ids}
            for completed, future in enumerate(as_completed(futures), 1):
                current.append(future.result())
                print(f"[{label}] built {completed}/{len(memory_ids)} memories", flush=True)
        current.sort(key=lambda item: item.memory_id)
        manifests.extend({"configuration": label, **dataclass_dict(item)} for item in current)
        navigator = GraphNavigator(store, variant=NavigatorVariant.N5_SET_COVER,
                                   dense_search=embedding.search if embedding else None)
        probe = GraphDiagnosticProbe(store, dense_search=embedding.search if embedding else None)
        nav_metrics = []
        for question in selected:
            nav = navigator.navigate(question.memory_id, question.query, profile.query_budget)
            navigation_rows.append({"configuration": label, "question_id": question.question_id,
                                    **dataclass_dict(nav)})
            metric = navigation_metrics(question, nav, store); metric["configuration"] = label
            nav_metrics.append(metric); metric_rows.append(metric)
            probe_results = {}
            for mode in ("seed_only", "relation_only", "shuffled"):
                result = probe.run(question.memory_id, question.query, profile.query_budget, mode=mode)
                probe_results[mode] = result
                row = diagnostic_metric(question, result, store); row["configuration"] = label
                diagnostic_rows.append(row)
                trace_rows.append({"configuration": label, "question_id": question.question_id,
                                   **dataclass_dict(result)})
            turns = list(store.turns(question.memory_id)); gold_keys = {
                (item.session_id, item.turn_index) for item in question.gold_turns}
            gold_turn_ids = {turn.turn_id for turn in turns if (turn.session_id, turn.turn_index) in gold_keys}
            seed_ids = set(probe_results["seed_only"].candidate_turn_ids)
            relation_ids = set(probe_results["relation_only"].candidate_turn_ids)
            diagnostic_rows[-2]["graph_unique_gold_turn_recall"] = len(
                (relation_ids - seed_ids) & gold_turn_ids) / max(1, len(gold_turn_ids))
            oracle_nodes = []
            for node in store.nodes(question.memory_id):
                if node.node_type.value in {"routing_card", "virtual_region"}: continue
                if any(member.turn_id in gold_turn_ids for group_id in node.all_evidence_group_ids
                       for group in [store.evidence_group(group_id)] if group for member in group.members):
                    oracle_nodes.append(node.node_id)
            oracle = probe.run(question.memory_id, question.query, profile.query_budget,
                               mode="relation_only", oracle_seed_ids=tuple(sorted(oracle_nodes)[:8]))
            oracle_row = diagnostic_metric(question, oracle, store); oracle_row.update(
                {"configuration": label, "mode": "oracle_seed"}); diagnostic_rows.append(oracle_row)
            trace_rows.append({"configuration": label, "question_id": question.question_id,
                               "mode": "oracle_seed", **dataclass_dict(oracle)})
            if label in {"c2", "c3", "c4", "g0", "g1", "g2", "g3", "g4",
                         "e0", "e1", "e2", "e3", "e4", "f0", "f1", "f2", "f3", "f4", "f5"}:
                for relation in (RelationType.SHARED_VALUE, RelationType.SAME_ACTIVITY,
                                 RelationType.HAS_FACT, RelationType.FACT_VALUE,
                                 RelationType.SCENE_CONTAINS, RelationType.AT_TIME,
                                 RelationType.STATE_NEXT, RelationType.TEMPORAL_BEFORE,
                                 RelationType.COLLECTION_CO_MEMBER, RelationType.PORTAL):
                    result = probe.run(question.memory_id, question.query, profile.query_budget,
                                       mode="relation_only", excluded_relations=(relation,))
                    row = diagnostic_metric(question, result, store); row.update(
                        {"configuration": label, "mode": "without:" + str(relation)})
                    diagnostic_rows.append(row)
        usage = [item.build_token_usage for item in current]
        # Use the cache ledger attached to the exact calls made by this profile. Summing only
        # uncached calls in the merged authority DB makes resumed/cached profiles look free and
        # also mixes earlier profiles that share a prompt hash. The cache record retains the
        # original cold usage, so this is reproducible across interrupted and resumed runs.
        cold_usage = {}; stage_tokens = {}
        for memory_id in memory_ids:
            shard_path = root / "memory_shards" / label / (
                memory_id.replace(":", "_").replace("/", "_") + ".sqlite")
            connection = sqlite3.connect(f"file:{shard_path}?mode=ro", uri=True)
            rows = connection.execute(
                "SELECT DISTINCT c.cache_key,c.stage,k.usage_json FROM llm_calls c "
                "JOIN llm_cache k ON k.cache_key=c.cache_key WHERE c.stage IN "
                "('scene_semantic','scene_semantic_retry','scene_semantic_repair',"
                "'hierarchy_l1','hierarchy_l2','hierarchy_l3')"
            ).fetchall()
            connection.close()
            for cache_key, stage, usage_json in rows:
                cold_usage.setdefault(cache_key, json.loads(usage_json))
                stage_tokens.setdefault(stage, {})[cache_key] = json.loads(usage_json)
        cold_tokens = sum(int(row.get("total_tokens", 0)) for row in cold_usage.values())
        cold_reasoning = sum(int(row.get("reasoning_tokens", 0)) for row in cold_usage.values())
        summary[label] = {"navigation": aggregate_metrics(nav_metrics), "memories": len(current),
            "average_uncached_backbone_tokens": cold_tokens / max(1, len(memory_ids)),
            "average_cold_equivalent_backbone_tokens": cold_tokens / max(1, len(memory_ids)),
            "cold_equivalent_stage_tokens": {stage: sum(int(row.get("total_tokens", 0))
                for row in rows.values()) for stage, rows in sorted(stage_tokens.items())},
            "reasoning_tokens": max(cold_reasoning,
                sum(int(x.get("reasoning_tokens", 0)) for x in usage)),
            "extraction_retry_rate": sum(int(item.build_diagnostics.get("extraction_retry_calls", 0))
                for item in current) / max(1, sum(int(item.build_diagnostics.get("extraction_scenes", 0))
                for item in current)),
            "terminal_turn_coverage": sum(float(item.build_diagnostics.get("terminal_turn_coverage", 0))
                for item in current) / max(1, len(current)),
            "config_hash": config_hash(profile)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "graph_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    for name, rows in (("metrics.jsonl", metric_rows), ("relation_diagnostics.jsonl", diagnostic_rows),
                       ("relation_traces.jsonl", trace_rows),
                       ("navigation_results.jsonl", navigation_rows)):
        with (root / name).open("w") as handle:
            for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
    if metric_rows:
        columns = sorted({key for row in metric_rows for key in row})
        with (root / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple))
                              else value for key, value in row.items()} for row in metric_rows)
        try:
            import pandas as pd
            pd.DataFrame(metric_rows).to_parquet(root / "metrics.parquet", index=False)
        except ImportError:
            pass
    with (root / "token_usage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("configuration", "average_cold_equivalent_tokens",
                                                     "retry_rate", "reasoning_tokens"))
        writer.writeheader()
        for label, row in summary.items():
            writer.writerow({"configuration": label,
                "average_cold_equivalent_tokens": row["average_cold_equivalent_backbone_tokens"],
                "retry_rate": row["extraction_retry_rate"],
                "reasoning_tokens": row["reasoning_tokens"]})
    with (root / "error_cases.jsonl").open("w") as handle:
        for row in metric_rows:
            if not row.get("turn_all_hit"):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    hashes = {}
    for path in (args.config, args.lme, args.locomo, args.gold):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "run_manifest.json").write_text(json.dumps({
        "created_at": stamp, "split": args.split, "profiles": list(summary),
        "questions": len(selected), "unique_question_ids": len({row.question_id for row in selected}),
        "memories": len(memory_ids), "input_hashes": hashes,
        "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in summary.values()),
    }, indent=2) + "\n")
    (root / "ablation_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>GraphMem V5 ablation</title>"
        "<h1>GraphMem V5 ablation</h1><pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>")
    store.close(); print(root)


if __name__ == "__main__": main()
