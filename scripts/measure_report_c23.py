#!/usr/bin/env python3
"""Measure C2/C3 routing scope, evidence all-hit, latency and false-complete."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.domain import QueryBudget  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.retrieval import GraphNavigator  # noqa: E402
from graphmem.retrieval.navigator import HarnessProfile  # noqa: E402
from graphmem.retrieval.operators import CLOSED_FORM_KINDS  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


ARMS = {
    "flat@32": {"hierarchical": False, "adaptive": False, "turns": 32},
    "fixed@32": {"hierarchical": True, "adaptive": False, "turns": 32},
    "adaptive@16": {"hierarchical": True, "adaptive": True, "turns": 16},
    "adaptive@32": {"hierarchical": True, "adaptive": True, "turns": 32},
    "adaptive@64": {"hierarchical": True, "adaptive": True, "turns": 64},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path,
                        default=Path("../artifacts/v5_8/lme_gold/graphmem.sqlite"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/report/v5_9/c23"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/v5/v5_9_report.json"))
    parser.add_argument("--lme", type=Path,
                        default=Path("../artifacts/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path,
                        default=Path("../artifacts/data/locomo10_graphmem.json"))
    parser.add_argument("--gold", type=Path,
                        default=Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--embedding", action="store_true",
                        help="index/use the configured local embedding service")
    parser.add_argument("--allow-subset", action="store_true",
                        help="allow diagnostic subset inputs instead of enforcing 500+1540")
    parser.add_argument("--arm", action="append", choices=tuple(ARMS))
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), p, method="linear"))


def route_sessions(view, node_ids) -> set[str]:
    sessions = set()
    for node_id in node_ids:
        if node_id not in view.nodes:
            continue
        node = view.nodes[node_id]
        session_id = node.attributes.get("session_id")
        if session_id:
            sessions.add(str(session_id))
        elif not view.hierarchy_children(node_id):
            # A terminal Session Card may expose only the compact descendant
            # list.  Never read this field from a selected root/parent: doing so
            # would turn a narrow route into artificial 100% scope recall.
            sessions.update(map(str, node.attributes.get("session_ids", ()) or ()))
    return sessions


def score(store, records, spec, search) -> dict:
    navigator = GraphNavigator(
        store, dense_search=search, harness_profile=HarnessProfile.H10_AST,
        hierarchical_routing=bool(spec["hierarchical"]),
        hierarchy_operator_aware=bool(spec["adaptive"]),
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=0.3, expansion_beam=2,
        read_pool_size=4,
    )
    budget = QueryBudget(max_evidence_turns=int(spec["turns"]))
    all_hit = recall_num = recall_den = 0
    certificate_complete = false_complete = prepack_pack_gap = 0
    route_num = route_den = 0
    route_selected: list[int] = []
    route_candidates: list[int] = []
    latencies: list[float] = []
    evidence_tokens: list[int] = []
    stages: dict[str, list[float]] = defaultdict(list)
    strata: dict[str, Counter] = defaultdict(Counter)
    per_question = []
    current = None
    turns = {}
    view = None
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        question = record.question
        if question.memory_id != current:
            current = question.memory_id
            turns = {row.turn_id: row for row in store.turns(current)}
            view = navigator.runtime.view(current)
        result = navigator.navigate(question.memory_id, question.query, budget)
        hit = {(turns[item].session_id, turns[item].turn_index)
               for item in result.retrieved_turn_ids if item in turns}
        gold = {(row.session_id, row.turn_index) for row in question.gold_turns}
        is_all_hit = int(gold <= hit)
        all_hit += is_all_hit
        recall_num += len(hit & gold)
        recall_den += len(gold)
        prepack_complete = bool(result.certificate and result.certificate.complete)
        postpack_complete = bool(
            result.certificate and result.certificate.post_pack_complete)
        bypass_eligible = bool(
            result.algebra and result.algebra.answer_kind in CLOSED_FORM_KINDS)
        # False-complete is a safety metric for certified deterministic bypass,
        # not for ordinary Lookup.  Lookup can have one sufficient witness while
        # turn-level gold intentionally annotates two redundant supports, and it
        # is never sent down the closed-form answer path.
        complete = bool(postpack_complete and bypass_eligible)
        certificate_complete += int(complete)
        false_complete += int(complete and not is_all_hit)
        prepack_pack_gap += int(prepack_complete and not postpack_complete)
        seeding = dict(result.trace.get("seeding", {}))
        hierarchy = dict(seeding.get("hierarchical_route", {}))
        selected_ids = tuple(seeding.get("hierarchical_selected_node_ids", ()))
        terminal_ids = tuple(seeding.get("hierarchical_terminal_node_ids", ()))
        # Session Cards are meaningful route decisions even when the physical
        # plan subsequently opens their Scene children.  Looking only at final
        # terminal Fact/Scene nodes discarded those selected sessions and
        # systematically undercounted hierarchy recall.
        route_ids = (tuple(dict.fromkeys((*selected_ids, *terminal_ids)))
                     or result.seed_node_ids)
        selected_sessions = route_sessions(view, route_ids) if view else set()
        gold_sessions = {row.session_id for row in question.gold_turns}
        route_num += len(selected_sessions & gold_sessions)
        route_den += len(gold_sessions)
        route_selected.append(int(hierarchy.get("selected_nodes", len(selected_ids))))
        route_candidates.append(int(hierarchy.get("candidate_count", len(selected_ids))))
        total = float(result.stage_latency_ms.get("total", 0.0))
        latencies.append(total)
        evidence_tokens.append(int(result.evidence_tokens))
        for key, value in result.stage_latency_ms.items():
            stages[key].append(float(value))
        counter = strata[record.stratum]
        counter.update({"n": 1, "all_hit": is_all_hit,
                        "recall_num": len(hit & gold), "recall_den": len(gold),
                        "false_complete": int(complete and not is_all_hit)})
        per_question.append({
            "question_id": question.question_id,
            "memory_id": question.memory_id,
            "stratum": record.stratum,
            "all_hit": bool(is_all_hit),
            "turn_recall": len(hit & gold) / max(1, len(gold)),
            "certificate_complete": complete,
            "postpack_complete": postpack_complete,
            "bypass_eligible": bypass_eligible,
            "prepack_complete": prepack_complete,
            "false_complete": bool(complete and not is_all_hit),
            "route_gold_session_recall": (
                len(selected_sessions & gold_sessions) / max(1, len(gold_sessions))),
            "route_selected_nodes": route_selected[-1],
            "route_candidate_nodes": route_candidates[-1],
            "evidence_tokens": result.evidence_tokens,
            "latency_ms": total,
        })
        if index % 100 == 0:
            print(f"  {index}/{len(records)} ({time.perf_counter()-started:.1f}s)",
                  flush=True)
    n = len(records)
    return {
        "n": n,
        "all_hit": all_hit / max(1, n),
        "turn_recall": recall_num / max(1, recall_den),
        "certificate_complete_rate": certificate_complete / max(1, n),
        "false_complete_rate": false_complete / max(1, n),
        "false_complete_given_certified": false_complete / max(1, certificate_complete),
        "prepack_to_postpack_gap_rate": prepack_pack_gap / max(1, n),
        "route_gold_session_recall": route_num / max(1, route_den),
        "route_selected_nodes_mean": statistics.fmean(route_selected),
        "route_candidate_nodes_mean": statistics.fmean(route_candidates),
        "evidence_tokens_mean": statistics.fmean(evidence_tokens),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": statistics.fmean(latencies),
        },
        "stage_mean_ms": {key: statistics.fmean(values)
                          for key, values in sorted(stages.items())},
        "strata": {key: dict(value) for key, value in strata.items()},
        "per_question": per_question,
    }


def setup_font() -> None:
    path = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if path.exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = ["FandolHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def plot(summary: dict[str, dict], output: Path) -> None:
    setup_font()
    compare = ["flat@32", "fixed@32", "adaptive@32"]
    labels = ["平面路由", "固定 Beam", "Operator-aware"]
    colors = ["#5F6B76", "#2378D7", "#18A999"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.8))
    x = np.arange(len(compare))
    route_recall = [summary[key]["route_gold_session_recall"] * 100 for key in compare]
    selected = [summary[key]["route_selected_nodes_mean"] for key in compare]
    axes[0].bar(x - 0.18, route_recall, width=0.36, color=colors, alpha=0.95)
    twin = axes[0].twinx()
    twin.bar(x + 0.18, selected, width=0.36, color=colors, alpha=0.35)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Gold Session 路由召回率（%）")
    twin.set_ylabel("平均选择节点数")
    axes[0].set_title("(a) 路由范围与召回")

    all_hit = [summary[key]["all_hit"] * 100 for key in compare]
    false_complete = [summary[key]["false_complete_rate"] * 100 for key in compare]
    axes[1].bar(x - 0.18, all_hit, width=0.36, color=colors, label="all-hit")
    axes[1].bar(x + 0.18, false_complete, width=0.36,
                color="#D84A4A", alpha=0.70, label="false-complete")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("问题占比（%）")
    axes[1].set_title("(b) 证据完备性与错误闭包")
    axes[1].legend(frameon=False)

    curve = ["adaptive@16", "adaptive@32", "adaptive@64"]
    tokens = [summary[key]["evidence_tokens_mean"] for key in curve]
    hits = [summary[key]["all_hit"] * 100 for key in curve]
    grouped: dict[tuple[float, float], list[str]] = {}
    for key, token, hit in zip(curve, tokens, hits):
        grouped.setdefault((round(token, 6), round(hit, 6)), []).append(
            key.split("@")[1])
    points = sorted(grouped)
    axes[2].plot([row[0] for row in points], [row[1] for row in points],
                 color="#18A999", marker="o", linewidth=2)
    for (token, hit), budgets in grouped.items():
        axes[2].annotate("/".join(budgets) + " turns", (token, hit),
                         xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[2].set_xlabel("平均 Evidence Token")
    axes[2].set_ylabel("all-hit（%）")
    axes[2].set_title("(c) Token--完备率曲线")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"eval_c23.{suffix}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    store = SQLiteGraphStore(args.db, read_only=not args.embedding)
    present = {str(row["memory_id"]) for row in
               store._read("SELECT memory_id FROM conversations")}
    records = [row for row in load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold),
        expect_lme=None if args.allow_subset else 500,
        expect_locomo=None if args.allow_subset else 1540)
        if row.question.gold_turns and row.question.memory_id in present]
    records.sort(key=lambda row: (row.question.memory_id, row.question.question_id))
    if args.limit:
        records = records[:args.limit]
    search = None
    if args.embedding:
        index = QwenEmbeddingIndex(store, config, record_usage=False)
        indexed = sum(index.index_memory(memory_id)
                      for memory_id in sorted({row.question.memory_id for row in records}))
        print(f"indexed {indexed} source turns", flush=True)
        search = index.search
    names = args.arm or list(ARMS)
    results = {}
    for name in names:
        print(f"=== {name} ===", flush=True)
        results[name] = score(store, records, ARMS[name], search)
        row = results[name]
        print(f"all_hit={row['all_hit']:.4f} route={row['route_gold_session_recall']:.4f} "
              f"p95={row['latency_ms']['p95']:.2f}ms "
              f"false_complete={row['false_complete_rate']:.4f}", flush=True)
    store.close()
    payload = {
        "experiment": "report_c23",
        "db": str(args.db),
        "config": str(args.config),
        "config_hash": config_hash(config),
        "embedding_enabled": args.embedding,
        "question_limit": args.limit,
        "arms": results,
    }
    (args.output / "c23_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [{"arm": name, **{key: value for key, value in row.items()
                              if key not in {"per_question", "strata", "latency_ms",
                                             "stage_mean_ms"}},
             **{f"latency_{key}_ms": value for key, value in row["latency_ms"].items()}}
            for name, row in results.items()]
    pd.DataFrame(rows).to_csv(args.output / "c23_summary.csv", index=False)
    if set(ARMS) <= set(results):
        plot(results, args.output)


if __name__ == "__main__":
    main()
