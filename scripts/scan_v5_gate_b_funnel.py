#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from graphmem.build import GraphBuildPipeline
from graphmem.config import load_config
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval import (
    aggregate_metrics, calibration40, load_dev_questions, load_gold_turns,
    navigation_metrics,
)
from graphmem.retrieval import GraphNavigator, NavigatorVariant
from graphmem.storage import SQLiteGraphStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / f"funnel_scan40_{stamp}"
    output.mkdir(parents=True)
    target = output / "graphmem.sqlite"
    shutil.copy2(args.source_db, target)
    store = SQLiteGraphStore(target)
    config = load_config(args.config)
    questions = calibration40(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)
    ), config.random_seed)
    memories = sorted({question.memory_id for question in questions})
    embedding = QwenEmbeddingIndex(store, config)
    rows, summary = [], {}

    def run(label: str, fanout: int, merge: bool) -> dict:
        profile = replace(
            config, profile="b5",
            coarsen=replace(config.coarsen, fanout=fanout, cross_session_merge=merge),
            edges=replace(config.edges, refine_mode="none"),
        )
        builder = GraphBuildPipeline(store, dataset_hash="funnel-scan40")
        manifests = [builder.build(memory_id, profile) for memory_id in memories]
        navigator = GraphNavigator(
            store, variant=NavigatorVariant.N5_SET_COVER, dense_search=embedding.search
        )
        metrics = []
        for question in questions:
            result = navigator.navigate(question.memory_id, question.query, config.query_budget)
            metric = navigation_metrics(question, result, store)
            metric["experiment"] = label
            metrics.append(metric); rows.append(metric)
        aggregate = aggregate_metrics(metrics)
        aggregate.update({
            "fanout": fanout, "cross_session_merge": merge,
            "nodes": sum(item.node_count for item in manifests),
            "edges": sum(item.edge_count for item in manifests),
            "build_backbone_tokens": sum(item.build_token_usage.get("total_tokens", 0)
                                             for item in manifests),
        })
        summary[label] = aggregate
        return aggregate

    fanout_results = [(fanout, run(f"fanout_{fanout}", fanout, True)) for fanout in (4, 8, 16)]
    # Apply the declared tie-break order exactly: quality, build LLM tokens,
    # visited nodes, evidence tokens, storage, then the smaller fanout.
    best_fanout = max(fanout_results, key=lambda item: (
        item[1]["equal_stratum_turn_all_hit"],
        -item[1]["build_backbone_tokens"],
        -item[1]["overall"]["visited_nodes"],
        -item[1]["overall"]["evidence_tokens"],
        -(item[1]["nodes"] + item[1]["edges"]),
        -item[0],
    ))[0]
    run("cross_session_merge_off", best_fanout, False)
    summary["selection"] = {"best_fanout": best_fanout, "seed": config.random_seed}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    store.close()
    print(output)


if __name__ == "__main__":
    main()
