#!/usr/bin/env python3
"""Run V5.5 query-only Harness ablations over an immutable V5.4 authority DB."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from graphmem.config import config_hash, load_config
from graphmem.domain import QueryBudget, dataclass_dict
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval import aggregate_metrics, load_dev_questions, load_gold_turns, navigation_metrics
from graphmem.retrieval import GraphDiagnosticProbe, GraphNavigator, HarnessProfile
from graphmem.storage import SQLiteGraphStore


def _gold_turn_ids(store, question):
    keys = {(item.session_id, item.turn_index) for item in question.gold_turns}
    return {turn.turn_id for turn in store.turns(question.memory_id)
            if (turn.session_id, turn.turn_index) in keys}


def _oracle_rows(store, question, budget, probe, navigation):
    gold_ids = _gold_turn_ids(store, question)
    nodes = []
    for node in store.nodes(question.memory_id):
        if node.node_type.value in {"routing_card", "virtual_region"}:
            continue
        for group_id in node.all_evidence_group_ids:
            group = store.evidence_group(group_id)
            if group and any(member.turn_id in gold_ids for member in group.members):
                nodes.append(node.node_id); break
    oracle = probe.run(question.memory_id, question.query, budget, mode="relation_only",
                       oracle_seed_ids=tuple(sorted(nodes)[:8]))
    candidate_ids = {row.turn_id for row in navigation.candidate_scores}
    turns = {item.turn_id: item for item in store.turns(question.memory_id)}
    gold_tokens = sum(max(1, len(turns[item].raw_text.split())) for item in gold_ids if item in turns)
    return {
        "oracle_seed_turn_all_hit": gold_ids <= set(oracle.candidate_turn_ids),
        "oracle_seed_turn_recall": len(gold_ids & set(oracle.candidate_turn_ids)) / max(1, len(gold_ids)),
        "oracle_packer_possible": gold_ids <= candidate_ids and len(gold_ids) <= budget.max_evidence_turns and gold_tokens <= budget.max_evidence_tokens,
        "gold_terminal_hydratable": bool(nodes),
    }


def _bootstrap_delta(h0, current, *, samples=2000, seed=42):
    baseline = {row["question_id"]: float(row["turn_all_hit"]) for row in h0}
    paired = [(baseline[row["question_id"]], float(row["turn_all_hit"])) for row in current]
    if not paired:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    values = [right - left for left, right in paired]; rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return {"point": sum(values) / len(values), "ci_low": draws[int(samples * .025)],
            "ci_high": draws[int(samples * .975)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profiles", default="h0,h1,h2,h3,h4,h5,h6")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root / f"v5_5_retrieval200_{stamp}"; root.mkdir(parents=True)
    store = SQLiteGraphStore(args.source_db, read_only=True)
    config = load_config(args.config)
    questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    if args.max_questions:
        questions = questions[:args.max_questions]
    embedding = QwenEmbeddingIndex(store, config, record_usage=False) if args.embedding else None
    all_metrics = []; navigation_rows = []; oracle_rows = []; summaries = {}; by_profile = {}
    for label in (item.strip() for item in args.profiles.split(",") if item.strip()):
        profile = HarnessProfile(label)
        budget = config.query_budget if profile in {HarnessProfile.H0_N5, HarnessProfile.H1_TELEMETRY} else replace(
            config.query_budget, max_evidence_turns=32)
        navigator = GraphNavigator(store, dense_search=embedding.search if embedding else None,
                                   harness_profile=profile)
        probe = GraphDiagnosticProbe(store, dense_search=embedding.search if embedding else None)
        rows = []
        for index, question in enumerate(questions, 1):
            result = navigator.navigate(question.memory_id, question.query, budget)
            metric = navigation_metrics(question, result, store)
            metric.update({"configuration": label, "budget_max_evidence_turns": budget.max_evidence_turns})
            metric.update(_oracle_rows(store, question, budget, probe, result))
            rows.append(metric); all_metrics.append(metric)
            navigation_rows.append({"configuration": label, "question_id": question.question_id,
                                    **dataclass_dict(result)})
            if index % 25 == 0:
                print(f"[{label}] navigated {index}/{len(questions)}", flush=True)
        by_profile[label] = rows
        summaries[label] = {"navigation": aggregate_metrics(rows), "oracle": {
            key: sum(float(row[key]) for row in rows) / max(1, len(rows))
            for key in ("oracle_seed_turn_all_hit", "oracle_seed_turn_recall", "oracle_packer_possible", "gold_terminal_hydratable")
        }, "config_hash": config_hash(config), "budget": dataclass_dict(budget)}
    paired = ({label: _bootstrap_delta(by_profile["h0"], rows)
               for label, rows in by_profile.items() if label != "h0"}
              if "h0" in by_profile else {})
    (root / "summary.json").write_text(json.dumps({"profiles": summaries, "paired_turn_all_hit_vs_h0": paired}, indent=2) + "\n")
    for name, rows in (("metrics.jsonl", all_metrics), ("navigation_results.jsonl", navigation_rows)):
        with (root / name).open("w") as handle:
            for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
    columns = sorted({key for row in all_metrics for key in row})
    with (root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(all_metrics)
    try:
        import pandas as pd
        pd.DataFrame(all_metrics).to_parquet(root / "metrics.parquet", index=False)
    except ImportError:
        pass
    with (root / "error_cases.jsonl").open("w") as handle:
        for row in all_metrics:
            if not row["turn_all_hit"]: handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_hash = hashlib.sha256(args.source_db.read_bytes()).hexdigest()
    manifest = {"created_at": stamp, "source_db": str(args.source_db), "source_db_sha256": source_hash,
                "source_read_only": True, "questions": len(questions),
                "unique_question_ids": len({row.question_id for row in questions}), "profiles": list(by_profile),
                "embedding_enabled": bool(embedding), "generative_llm_calls": 0,
                "input_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (args.config, args.lme, args.locomo, args.gold)}}
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "ablation_report.html").write_text("<!doctype html><meta charset='utf-8'><title>V5.5 retrieval</title><pre>" +
                                                html.escape(json.dumps({"profiles": summaries, "paired": paired}, indent=2)) + "</pre>")
    store.close(); print(root)


if __name__ == "__main__":
    main()
