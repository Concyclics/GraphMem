#!/usr/bin/env python3
"""Measure whether hard mandatory-first ranking hides better raw evidence.

The algebra may bind dozens of weak facts and mark every provenance turn
mandatory.  The production rank currently treats that boolean as an infinite
score before lexical, dense, graph, owner and temporal signals.  This gate keeps
the candidate reservoir and all component scores fixed, then replaces only that
infinite priority with finite bonuses.  Gold labels are used solely here, after
retrieval, to choose an operating point; they never enter GraphNavigator.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / (
        "artifacts/development_sets/"
        "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/"
                        "hnsw_qwen_typed_dev200_graph_bounded_frontier/"
                        "report_graph.sqlite")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--bonuses", type=float, nargs="+", default=(0, .5, 1, 2, 4))
    parser.add_argument("--routes", type=int, nargs="+", default=(3, 5, 8))
    parser.add_argument("--ks", type=int, nargs="+", default=(8, 16, 32, 48, 64))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_13/mandatory_fusion_dev200")
    return parser.parse_args()


def ratio(a: int | float, b: int | float) -> float:
    return float(a) / float(b) if b else 0.0


def set_metrics(gold: set[tuple[str, int]], predicted: set[tuple[str, int]]) -> dict[str, float]:
    hits = len(gold & predicted)
    precision = ratio(hits, len(predicted))
    recall = ratio(hits, len(gold))
    return {
        "all_hit": float(gold <= predicted), "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
    }


def bootstrap_delta(rows: list[dict[str, Any]], left: str, right: str,
                    key: str, samples: int = 4000) -> list[float]:
    rng = random.Random(42)
    values = []
    for _ in range(samples):
        selected = [rows[rng.randrange(len(rows))] for _row in rows]
        values.append(statistics.fmean(
            row[right][key] - row[left][key] for row in selected))
    values.sort()
    return [values[int(.025 * samples)], values[int(.975 * samples)]]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bonuses = tuple(dict.fromkeys(args.bonuses))
    routes = tuple(dict.fromkeys(args.routes))
    ks = tuple(dict.fromkeys(sorted(args.ks)))
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    if args.limit:
        questions = questions[:args.limit]
    config = load_config(args.config)
    budget = replace(config.query_budget, max_evidence_turns=max(ks),
                     max_evidence_tokens=10000)
    store = SQLiteGraphStore(args.db, read_only=True)
    navigator = GraphNavigator(
        store, harness_profile=HarnessProfile.H11_UNIFIED_IR,
        hierarchical_routing=True, hierarchy_operator_aware=True,
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=.3, expansion_beam=2,
        obligation_aware_packing=True, span_pack_window=96,
        native_seed_fusion=True, read_pool_size=4)
    arms = ("hard", *(f"soft_{bonus:g}" for bonus in bonuses),
            *(f"route_{route}" for route in routes))
    rows: list[dict[str, Any]] = []
    turn_cache: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for index, question in enumerate(questions, 1):
        result = navigator.navigate(question.memory_id, question.query, budget)
        turn_map = turn_cache.setdefault(
            question.memory_id,
            {turn.turn_id: turn for turn in store.turns(question.memory_id)})
        gold = {(item.session_id, item.turn_index) for item in question.gold_turns}
        rankings = {"hard": result.candidate_scores}
        for bonus in bonuses:
            rankings[f"soft_{bonus:g}"] = tuple(sorted(
                result.candidate_scores,
                key=lambda item: (-(item.fused_score + bonus * int(item.mandatory)),
                                  item.turn_id)))
        for route in routes:
            routed_sessions = set(navigator.route_sessions(
                tuple(turn_map.values()), question.query, route))
            filtered = tuple(
                item for item in result.candidate_scores
                if item.mandatory or item.session_id in routed_sessions)
            # Production keeps the original pool if routing finds nothing.
            rankings[f"route_{route}"] = filtered or result.candidate_scores
        row: dict[str, Any] = {
            "question_id": question.question_id, "memory_id": question.memory_id,
            "stratum": question.stratum, "query": question.query,
            "gold_refs": sorted(gold),
        }
        candidate_gold = []
        for candidate in result.candidate_scores:
            turn = turn_map.get(candidate.turn_id)
            if turn is None or (turn.session_id, turn.turn_index) not in gold:
                continue
            candidate_gold.append({
                "turn_id": candidate.turn_id, "mandatory": candidate.mandatory,
                "fused_score": candidate.fused_score,
                "components": asdict(candidate),
            })
        row["candidate_gold"] = candidate_gold
        for arm, ranking in rankings.items():
            payload: dict[str, float] = {}
            refs = tuple(dict.fromkeys(
                (turn_map[item.turn_id].session_id, turn_map[item.turn_id].turn_index)
                for item in ranking if item.turn_id in turn_map))
            for k in ks:
                metrics = set_metrics(gold, set(refs[:k]))
                payload.update({f"{name}@{k}": value for name, value in metrics.items()})
            payload["first_gold_rank"] = float(next(
                (rank for rank, ref in enumerate(refs, 1) if ref in gold), 0))
            payload["last_gold_rank"] = float(max(
                (rank for rank, ref in enumerate(refs, 1) if ref in gold), default=0))
            row[arm] = payload
        rows.append(row)
        if index % 25 == 0:
            print(f"ranked {index}/{len(questions)}", flush=True)
    store.close()

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {"questions": len(items), "arms": {}}
        fields = tuple(rows[0]["hard"]) if rows else ()
        for arm in arms:
            output["arms"][arm] = {
                field: statistics.fmean(row[arm][field] for row in items)
                for field in fields}
        for arm in arms[1:]:
            output.setdefault("deltas_vs_hard", {})[arm] = {
                field: output["arms"][arm][field] - output["arms"]["hard"][field]
                for field in fields}
            output.setdefault("ci95_vs_hard", {})[arm] = {
                field: bootstrap_delta(items, "hard", arm, field)
                for field in fields if field.startswith(("recall@", "all_hit@"))}
        return output

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    summary = {
        "schema_version": "graphmem-v5.13-mandatory-fusion-v1",
        "method": "fixed_candidate_reservoir_finite_mandatory_bonus",
        "bonuses": list(bonuses), "routes": list(routes), "ks": list(ks),
        "overall": summarize(rows),
        "strata": {name: summarize(group) for name, group in sorted(groups.items())},
        "mandatory_gold_rate": ratio(
            sum(item["mandatory"] for row in rows for item in row["candidate_gold"]),
            sum(len(row["candidate_gold"]) for row in rows)),
        "elapsed_sec": time.perf_counter() - started,
    }
    with (args.output / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Hard mandatory-first → finite mandatory bonus", "",
        f"Candidate gold 中 mandatory 占比：{summary['mandatory_gold_rate']:.2%}", "",
        "| Arm | R@16 | P@16 | All@16 | R@32 | P@32 | All@32 | R@48 | P@48 | All@48 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in summary["overall"]["arms"].items():
        lines.append(
            f"| {arm} | {values['recall@16']:.2%} | {values['precision@16']:.2%} | "
            f"{values['all_hit@16']:.2%} | {values['recall@32']:.2%} | "
            f"{values['precision@32']:.2%} | {values['all_hit@32']:.2%} | "
            f"{values['recall@48']:.2%} | {values['precision@48']:.2%} | "
            f"{values['all_hit@48']:.2%} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
