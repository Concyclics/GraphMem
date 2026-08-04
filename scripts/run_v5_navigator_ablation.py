#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from graphmem.config import load_config
from graphmem.build import GraphBuildPipeline
from graphmem.domain import dataclass_dict
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval import (
    aggregate_metrics, calibration40, load_dev_questions, load_gold_turns,
    navigation_metrics,
)
from graphmem.retrieval import GraphNavigator, NavigatorVariant
from graphmem.storage import SQLiteGraphStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--full-200", action="store_true")
    parser.add_argument("--rebuild-profile", choices=[f"b{i}" for i in range(6)])
    parser.add_argument("--navigator", choices=[str(item) for item in NavigatorVariant])
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.resume_dir:
        output = args.resume_dir
        target_db = output / "graphmem.sqlite"
        if not target_db.exists():
            raise FileNotFoundError(target_db)
    else:
        if not args.source_db:
            parser.error("--source-db is required unless --resume-dir is used")
        output = args.output_root / f"navigator_only_{stamp}"
        output.mkdir(parents=True)
        target_db = output / "graphmem.sqlite"
        shutil.copy2(args.source_db, target_db)
    store = SQLiteGraphStore(target_db)
    config = load_config(args.config)
    all_questions = load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)
    )
    questions = all_questions if args.full_200 else calibration40(all_questions, config.random_seed)
    embedding = QwenEmbeddingIndex(store, config) if args.embedding else None
    graph_manifests = []
    if args.rebuild_profile:
        profile = replace(config, profile=args.rebuild_profile)
        builder = GraphBuildPipeline(store, dataset_hash="existing-snapshot-confirm")
        for memory_id in sorted({question.memory_id for question in questions}):
            if args.resume_dir and store.graph_version(memory_id) >= 3:
                graph_manifests.append({
                    "memory_id": memory_id, "graph_version": store.graph_version(memory_id),
                    "graph_checksum": store.graph_checksum(memory_id),
                    "node_count": len(store.nodes(memory_id)), "edge_count": len(store.edges(memory_id)),
                    "resumed_existing": True,
                })
            else:
                graph_manifests.append(dataclass_dict(builder.build(memory_id, profile)))
    summary = {}
    rows = []
    variants = (NavigatorVariant(args.navigator),) if args.navigator else (
        NavigatorVariant.N1_RAW_FUSION, NavigatorVariant.N2_PROVENANCE,
        NavigatorVariant.N3_PRIORITY, NavigatorVariant.N4_CERTIFICATE,
        NavigatorVariant.N5_SET_COVER,
    )
    for variant in variants:
        navigator = GraphNavigator(
            store, variant=variant, dense_search=embedding.search if embedding else None
        )
        metrics = []
        for question in questions:
            result = navigator.navigate(question.memory_id, question.query, config.query_budget)
            row = navigation_metrics(question, result, store)
            row["variant"] = str(variant)
            metrics.append(row)
            rows.append(row)
        summary[str(variant)] = aggregate_metrics(metrics)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "graph_manifest.json").write_text(json.dumps(graph_manifests, indent=2) + "\n")
    with (output / "metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    store.close()
    print(output)


if __name__ == "__main__":
    main()
