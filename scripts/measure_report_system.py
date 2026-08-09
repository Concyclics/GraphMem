#!/usr/bin/env python3
"""Concurrent read and atomic full-vs-affected-path publication benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

from graphmem.build import GraphBuildPipeline  # noqa: E402
from graphmem.build.incremental import (  # noqa: E402
    plan_affected_paths,
    publish_affected_path,
    recompile_route_ancestors,
)
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.domain import (  # noqa: E402
    Conversation,
    QueryBudget,
    Session,
    SourceTurn,
    stable_id,
)
from graphmem.retrieval import GraphNavigator  # noqa: E402
from graphmem.retrieval.navigator import HarnessProfile  # noqa: E402
from graphmem.runtime import GraphReadView, SQLiteSnapshotRuntime  # noqa: E402
from graphmem.serving import ProcessShardedNavigator  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/report/v5_9/system"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/v5/v5_9_report.json"))
    parser.add_argument("--sessions", type=int, default=256)
    parser.add_argument("--turns-per-session", type=int, default=4)
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--requests", type=int, default=256,
                        help="minimum requests per concurrency point")
    parser.add_argument("--update-repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8,
                        help="persistent process shards in the multi-user query plane")
    parser.add_argument("--process-start-method", default="spawn")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse output/system_graph.sqlite if already complete")
    return parser.parse_args()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_synthetic(store: SQLiteGraphStore, sessions_n: int,
                     turns_per_session: int) -> None:
    memory_id = "report-system"
    sessions = []
    turns = []
    for session_index in range(sessions_n):
        session_id = f"session:{session_index:05d}"
        timestamp = f"2025-{session_index % 12 + 1:02d}-{session_index % 27 + 1:02d}"
        sessions.append(Session(
            session_id, memory_id, session_index, timestamp,
            sha(f"session-{session_index}")))
        topic = session_index % max(8, sessions_n // 16)
        for turn_index in range(turns_per_session):
            text = (f"User_{session_index % 32} discussed topic_{topic} "
                    f"anchor_{session_index} state_{turn_index} project update")
            turns.append(SourceTurn(
                stable_id("turn", memory_id, session_id, turn_index),
                memory_id, session_id, turn_index,
                f"User_{session_index % 32}", "Agent",
                "user" if turn_index % 2 == 0 else "assistant",
                timestamp, text, sha(text)))
    store.ingest_conversation(
        Conversation(memory_id, "synthetic-system", memory_id,
                     sha(f"{sessions_n}:{turns_per_session}")),
        sessions, turns)


def report_config(path: Path):
    base = load_config(path)
    return replace(
        base,
        scenes=replace(base.scenes, llm_semantic_extraction=False,
                       llm_hierarchy_compression=False),
        coarsen=replace(base.coarsen, entity_merge=False),
        edges=replace(base.edges, refine_mode="none"),
    )


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values), p, method="linear")) if values else 0.0


def query_workload(sessions_n: int, count: int) -> list[str]:
    return [
        f"What project update happened to User_{index % 32} "
        f"at anchor_{index % sessions_n} in topic_{index % max(8, sessions_n // 16)}?"
        for index in range(count)
    ]


def run_concurrency(navigator: GraphNavigator, queries: list[str], concurrency: int,
                    requests: int) -> dict:
    workload = [queries[index % len(queries)] for index in range(requests)]
    budget = QueryBudget(max_evidence_turns=16, max_evidence_tokens=2400)
    latencies = []
    stage_rows = []
    errors = []

    def one(query: str):
        started = time.perf_counter()
        result = navigator.navigate("report-system", query, budget)
        return (time.perf_counter() - started) * 1000, result.stage_latency_ms

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, query) for query in workload]
        for future in as_completed(futures):
            try:
                latency, stages = future.result()
                latencies.append(latency)
                stage_rows.append(stages)
            except BaseException as exc:  # preserve an error rate in the artifact
                errors.append(f"{type(exc).__name__}: {exc}")
    wall = time.perf_counter() - wall_started
    stage_names = sorted({key for row in stage_rows for key in row if key != "total"})
    return {
        "concurrency": concurrency,
        "requests": requests,
        "completed": len(latencies),
        "errors": len(errors),
        "error_samples": errors[:3],
        "wall_seconds": wall,
        "qps": len(latencies) / max(wall, 1e-9),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": statistics.fmean(latencies) if latencies else 0.0,
        },
        "stage_mean_ms": {
            key: statistics.fmean(float(row.get(key, 0.0)) for row in stage_rows)
            for key in stage_names
        },
    }


def run_process_concurrency(
    navigator: ProcessShardedNavigator,
    queries: list[str],
    concurrency: int,
    requests: int,
) -> dict:
    """Drive a fixed process-sharded service with closed-loop clients."""
    workload = [queries[index % len(queries)] for index in range(requests)]
    budget = QueryBudget(max_evidence_turns=16, max_evidence_tokens=2400)
    latencies = []
    stage_rows = []
    errors = []

    def one(query: str):
        started = time.perf_counter()
        result = navigator.submit("report-system", query, budget).result()
        return (time.perf_counter() - started) * 1000, result.stage_latency_ms

    wall_started = time.perf_counter()
    # These are clients, not query workers.  Each blocks on the persistent
    # process service, so latency includes IPC and worker-queue time.
    with ThreadPoolExecutor(max_workers=concurrency) as clients:
        futures = [clients.submit(one, query) for query in workload]
        for future in as_completed(futures):
            try:
                latency, stages = future.result()
                latencies.append(latency)
                stage_rows.append(stages)
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    wall = time.perf_counter() - wall_started
    stage_names = sorted({key for row in stage_rows for key in row if key != "total"})
    return {
        "concurrency": concurrency,
        "requests": requests,
        "completed": len(latencies),
        "errors": len(errors),
        "error_samples": errors[:3],
        "wall_seconds": wall,
        "qps": len(latencies) / max(wall, 1e-9),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": statistics.fmean(latencies) if latencies else 0.0,
        },
        "stage_mean_ms": {
            key: statistics.fmean(float(row.get(key, 0.0)) for row in stage_rows)
            for key in stage_names
        },
    }


def changed_route_nodes(view: GraphReadView, version: int, iteration: int):
    plan = plan_affected_paths(
        view, memory_id="report-system", source_version=version,
        changed_session_ids=("session:00000",))
    leaf = view.nodes[plan.session_card_ids[0]]
    replacement = replace(
        leaf, summary=f"incremental cobalt state iteration_{iteration} anchor_0")
    nodes = recompile_route_ancestors(
        view, plan, (replacement,), summary_words=320)
    return plan, nodes


def benchmark_updates(store: SQLiteGraphStore, repeats: int) -> dict:
    rows = []
    for mode in ("full_snapshot", "affected_path"):
        for iteration in range(repeats):
            version = store.graph_version("report-system")
            view = GraphReadView(store.nodes("report-system"), store.edges("report-system"))
            plan, route_nodes = changed_route_nodes(view, version, iteration)
            by_id = dict(view.nodes)
            by_id.update({node.node_id: node for node in route_nodes})
            started = time.perf_counter()
            if mode == "full_snapshot":
                new_version = store.replace_graph(
                    "report-system", tuple(by_id.values()), tuple(view.edges.values()),
                    store.evidence_groups("report-system"))
                touched = len(by_id) + len(view.edges) + len(store.evidence_groups("report-system"))
                recomputed = len(by_id)
            else:
                result = publish_affected_path(
                    store, view, plan, replacement_nodes=(route_nodes[0],),
                    summary_words=320)
                new_version = result.graph_version
                touched = result.touched_rows
                recomputed = plan.recomputed_nodes
            commit_ms = (time.perf_counter() - started) * 1000
            visible_started = time.perf_counter()
            visible_version = store.graph_version("report-system")
            authority_visible_ms = (time.perf_counter() - visible_started) * 1000
            cold_runtime = SQLiteSnapshotRuntime(store, max_cached_views=2)
            view_started = time.perf_counter()
            new_view = cold_runtime.view("report-system")
            first_view_ready_ms = (time.perf_counter() - view_started) * 1000
            rows.append({
                "mode": mode,
                "iteration": iteration,
                "source_version": version,
                "new_version": new_version,
                "visible_version": visible_version,
                "commit_ms": commit_ms,
                "authority_visible_probe_ms": authority_visible_ms,
                "first_new_read_view_ms": first_view_ready_ms,
                "touched_rows": touched,
                "recomputed_nodes": recomputed,
                "graph_nodes": len(new_view.nodes),
                "graph_edges": len(new_view.edges),
                "recompute_fraction": recomputed / max(1, len(new_view.nodes)),
            })
    return {"rows": rows}


def failure_visibility_probe(store: SQLiteGraphStore) -> dict:
    runtime = SQLiteSnapshotRuntime(store, max_cached_views=2)
    old = runtime.view("report-system")
    old_version = store.graph_version("report-system")
    errors = []
    observed = []
    stop = False

    def reader():
        while not stop:
            try:
                view = runtime.view("report-system")
                observed.append((store.graph_version("report-system"), len(view.nodes),
                                 len(view.edges)))
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(reader) for _ in range(8)]
        time.sleep(0.02)
        version = store.graph_version("report-system")
        current = GraphReadView(store.nodes("report-system"), store.edges("report-system"))
        plan, nodes = changed_route_nodes(current, version, 999)
        publish_affected_path(store, current, plan,
                              replacement_nodes=(nodes[0],))
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if observed and any(row[0] == version + 1 for row in observed):
                break
            time.sleep(0.001)
        stop = True
        for future in futures:
            future.result()
    shapes = {(nodes_n, edges_n) for _version, nodes_n, edges_n in observed}
    return {
        "old_version": old_version,
        "new_version": store.graph_version("report-system"),
        "old_view_remained_immutable": len(old.nodes) > 0,
        "reader_operations": len(observed),
        "reader_errors": len(errors),
        "error_samples": errors[:3],
        "observed_graph_shapes": sorted(map(list, shapes)),
    }


def setup_font() -> None:
    path = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if path.exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = ["FandolHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def plot(read_rows: list[dict], update_rows: list[dict], output: Path) -> None:
    setup_font()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.8))
    for mode, label, color in (
            ("flat", "单进程·平面", "#5F6B76"),
            ("hierarchical", "单进程·层次", "#18A999"),
            ("process_hierarchical", "进程分片·层次", "#2962A3")):
        part = sorted((row for row in read_rows if row["mode"] == mode),
                      key=lambda row: row["concurrency"])
        x = [row["concurrency"] for row in part]
        axes[0].plot(x, [row["qps"] for row in part], marker="o",
                     color=color, label=label)
        axes[1].plot(x, [row["latency_ms"]["p95"] for row in part], marker="o",
                     color=color, label=label)
    axes[0].set(xlabel="并发客户端", ylabel="吞吐（QPS）", title="(a) 并发吞吐")
    axes[1].set(xlabel="并发客户端", ylabel="p95 延迟（ms）", title="(b) 尾延迟")
    axes[0].legend(frameon=False); axes[1].legend(frameon=False)

    modes = ["full_snapshot", "affected_path"]
    labels = ["全量 Snapshot", "Affected-path"]
    commit = [statistics.median(row["commit_ms"] for row in update_rows
                                if row["mode"] == mode) for mode in modes]
    touched = [statistics.median(row["touched_rows"] for row in update_rows
                                 if row["mode"] == mode) for mode in modes]
    x = np.arange(2)
    axes[2].bar(x - 0.18, commit, width=0.36,
                color=["#D84A4A", "#18A999"])
    twin = axes[2].twinx()
    twin.bar(x + 0.18, touched, width=0.36,
             color=["#D84A4A", "#18A999"], alpha=0.32)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("发布事务延迟（ms）")
    twin.set_ylabel("写事务触及行数")
    twin.set_yscale("log")
    axes[2].set_title("(c) 增量发布")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"eval_system.{suffix}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    db = args.output / "system_graph.sqlite"
    if db.exists() and not args.reuse:
        raise SystemExit(f"{db} already exists; use --reuse or choose a new --output")
    store = SQLiteGraphStore(db)
    config = report_config(args.config)
    if store.graph_version("report-system") == 0:
        ingest_synthetic(store, args.sessions, args.turns_per_session)
        started = time.perf_counter()
        manifest = GraphBuildPipeline(store, dataset_hash="report-system-v1").build(
            "report-system", config)
        build_seconds = time.perf_counter() - started
        build_summary = {
            "seconds": build_seconds,
            "nodes": manifest.node_count,
            "edges": manifest.edge_count,
            "diagnostics": manifest.build_diagnostics,
            "usage": manifest.build_token_usage,
        }
    else:
        build_summary = {"reused": True, "nodes": len(store.nodes("report-system")),
                         "edges": len(store.edges("report-system"))}

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    flat = GraphNavigator(
        store, harness_profile=HarnessProfile.H10_AST,
        hierarchical_routing=False, read_pool_size=8,
        snapshot_cache_bytes=512 * 1024 * 1024)
    hierarchical = GraphNavigator(
        store, harness_profile=HarnessProfile.H10_AST,
        hierarchical_routing=True, hierarchy_operator_aware=True,
        read_pool_size=8, snapshot_cache_bytes=512 * 1024 * 1024)
    queries = query_workload(args.sessions, max(args.requests, args.sessions))
    # Warm all immutable indexes and tokenizer paths outside the timed region.
    warm_budget = QueryBudget(max_evidence_turns=16, max_evidence_tokens=2400)
    for navigator in (flat, hierarchical):
        for query in queries[:16]:
            navigator.navigate("report-system", query, warm_budget)
    rss_hot = process.memory_info().rss

    process_pool = ProcessShardedNavigator(
        db, workers=args.workers,
        navigator_options={
            "harness_profile": HarnessProfile.H10_AST,
            "hierarchical_routing": True,
            "hierarchy_operator_aware": True,
            "snapshot_cache_bytes": 512 * 1024 * 1024,
        },
        start_method=args.process_start_method,
    )
    worker_snapshots = process_pool.warm(
        "report-system", queries[:16], warm_budget)
    worker_cache_bytes = sum(row.estimated_bytes for row in worker_snapshots)

    read_rows = []
    try:
        for concurrency in (int(item) for item in args.concurrency.split(",") if item):
            requests = max(args.requests, concurrency * 4)
            for mode, navigator in (("flat", flat), ("hierarchical", hierarchical)):
                print(f"{mode} concurrency={concurrency} requests={requests}", flush=True)
                row = run_concurrency(navigator, queries, concurrency, requests)
                row["mode"] = mode
                read_rows.append(row)
                print(f"  qps={row['qps']:.2f} p95={row['latency_ms']['p95']:.2f}ms "
                      f"errors={row['errors']}", flush=True)
            print(f"process_hierarchical concurrency={concurrency} "
                  f"requests={requests}", flush=True)
            row = run_process_concurrency(
                process_pool, queries, concurrency, requests)
            row["mode"] = "process_hierarchical"
            row["workers"] = args.workers
            read_rows.append(row)
            print(f"  qps={row['qps']:.2f} p95={row['latency_ms']['p95']:.2f}ms "
                  f"errors={row['errors']}", flush=True)
    finally:
        process_pool.close()

    updates = benchmark_updates(store, args.update_repeats)
    availability = failure_visibility_probe(store)
    payload = {
        "experiment": "report_system",
        "config": str(args.config),
        "effective_config_hash": config_hash(config),
        "build": build_summary,
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "memory_total_gib": psutil.virtual_memory().total / 1024 ** 3,
            "python": sys.version,
        },
        "memory": {
            "rss_before_mib": rss_before / 1024 ** 2,
            "rss_hot_mib": rss_hot / 1024 ** 2,
            "hot_delta_mib": (rss_hot - rss_before) / 1024 ** 2,
            "flat_cache": flat.runtime.cache_stats(),
            "hierarchical_cache": hierarchical.runtime.cache_stats(),
            "process_worker_count": args.workers,
            "process_worker_snapshot_bytes": worker_cache_bytes,
            "process_worker_snapshots": [asdict(row) for row in worker_snapshots],
        },
        "reads": read_rows,
        "updates": updates,
        "availability": availability,
    }
    (args.output / "system_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{
        "mode": row["mode"], "concurrency": row["concurrency"],
        "qps": row["qps"], "p50_ms": row["latency_ms"]["p50"],
        "p95_ms": row["latency_ms"]["p95"], "p99_ms": row["latency_ms"]["p99"],
        "errors": row["errors"],
    } for row in read_rows]).to_csv(args.output / "system_reads.csv", index=False)
    pd.DataFrame(updates["rows"]).to_csv(
        args.output / "system_updates.csv", index=False)
    plot(read_rows, updates["rows"], args.output)
    store.close()


if __name__ == "__main__":
    main()
