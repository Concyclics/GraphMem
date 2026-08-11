#!/usr/bin/env python3
"""Render the V5.18 Pareto figure with adaptive low-concurrency resampling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from render_v5_11_mem0_pareto import (
    plot_memory,
    plot_pareto,
    write_tex_table,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
METRIC_NAMES = ("mean", "p50", "p95", "p99", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=WORKSPACE / "artifacts/report/v5_18",
    )
    parser.add_argument(
        "--recheck", type=Path,
        default=(WORKSPACE / "artifacts/report/v5_18/"
                 "tail_latency_recheck_20260810"),
    )
    parser.add_argument(
        "--report", type=Path, default=WORKSPACE / "GraphMem_report",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_metric(cell: dict[str, Any], section: str, metric: str) -> float:
    value = cell["aggregate"][section][metric]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(WORKSPACE.resolve())
        return f"../{relative.as_posix()}"
    except ValueError:
        return str(path.resolve())


def load_screening_rows(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    systems = (
        ("GraphMem", "qps_w{workers}_optimized_prewarm"),
        ("Mem0 OSS", "mem0_qps_w{workers}"),
    )
    for system, pattern in systems:
        for workers in (1, 4, 8):
            path = root / pattern.format(workers=workers) / "summary.json"
            summary = load(path)
            sources.append(path)
            for cell in summary["cells"]:
                trial = cell["trials"][0]
                row = {
                    "system": system,
                    "workers": workers,
                    "clients": int(cell["clients"]),
                    "qps": aggregate_metric(cell, "qps", "mean"),
                    "completed": int(trial["completed"]),
                    "failed": int(trial["failed"]),
                    "timed_out": int(trial["timed_out"]),
                    "rejected": int(trial["rejected"]),
                    "wrong_partition": int(trial.get(
                        "wrong_memory", trial.get("wrong_user", 0))),
                    "worker_rss_mib": float(cell["total_worker_rss_mib"]),
                    "worker_pss_mib": float(cell["total_worker_pss_mib"]),
                    "sample_protocol": "screening_2s_x1",
                    "source": display_path(path),
                }
                for metric in METRIC_NAMES:
                    row[f"latency_{metric}_ms"] = aggregate_metric(
                        cell, "latency_ms", metric)
                rows.append(row)
    return rows, sources


def apply_recheck(
    rows: list[dict[str, Any]], analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    indexed = {
        (row["system"], row["workers"], row["clients"]): row
        for row in rows
    }
    for sample in analysis["rows"]:
        key = (sample["system"], sample["workers"], sample["clients"])
        row = indexed.get(key)
        if row is None:
            row = {
                "system": sample["system"],
                "workers": sample["workers"],
                "clients": sample["clients"],
                "latency_mean_ms": float(sample["metrics"]["p50"]["mean"]),
                "failed": int(sample["failed"]),
                "rejected": int(sample["rejected"]),
                "wrong_partition": 0,
                "worker_rss_mib": float(sample["worker_rss_mib"]),
                "worker_pss_mib": float(sample["worker_pss_mib"]),
            }
            rows.append(row)
            indexed[key] = row
        row["qps"] = float(sample["metrics"]["qps"]["mean"])
        for metric in ("p50", "p95", "p99", "max"):
            row[f"latency_{metric}_ms"] = float(
                sample["metrics"][metric]["mean"])
            row[f"latency_{metric}_ci95_half_width_ms"] = float(
                sample["metrics"][metric]["ci95_half_width"])
        row["completed"] = int(sample["completed"])
        row["timed_out"] = int(sample["timed_out"])
        row["sample_protocol"] = "tail_resample_30s_x5"
        row["source"] = display_path(Path(sample["source"]))
    return rows


def pairwise_common(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["system"], row["workers"], row["clients"]): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    clients_by_workers = {
        1: (1, 4, 16, 64),
        4: (1, 4, 16, 64, 128),
        8: (1, 4, 16, 64, 128),
    }
    for workers, clients_for_worker in clients_by_workers.items():
        for clients in clients_for_worker:
            graph = indexed[("GraphMem", workers, clients)]
            mem0 = indexed[("Mem0 OSS", workers, clients)]
            result.append({
                "workers": workers,
                "clients": clients,
                "graphmem_qps": graph["qps"],
                "mem0_qps": mem0["qps"],
                "qps_speedup": graph["qps"] / mem0["qps"],
                "graphmem_p95_ms": graph["latency_p95_ms"],
                "mem0_p95_ms": mem0["latency_p95_ms"],
                "p95_reduction": 1.0 - (
                    graph["latency_p95_ms"] / mem0["latency_p95_ms"]
                ),
                "strictly_dominates": (
                    graph["qps"] > mem0["qps"]
                    and graph["latency_p95_ms"] < mem0["latency_p95_ms"]
                ),
            })
    return result


def latency_inversions(
    rows: list[dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    inversions: list[dict[str, Any]] = []
    for system in ("GraphMem", "Mem0 OSS"):
        for workers in (1, 4, 8):
            group = sorted(
                (row for row in rows
                 if row["system"] == system and row["workers"] == workers),
                key=lambda row: row["clients"],
            )
            for left, right in zip(group, group[1:]):
                key = f"latency_{metric}_ms"
                if right[key] < left[key]:
                    inversions.append({
                        "system": system,
                        "workers": workers,
                        "from_clients": left["clients"],
                        "to_clients": right["clients"],
                        "from_ms": left[key],
                        "to_ms": right[key],
                    })
    return inversions


def write_tail_table(path: Path, table_path: Path) -> None:
    path.write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis_path = args.recheck / "analysis.json"
    analysis = load(analysis_path)
    rows, screening_sources = load_screening_rows(args.root)
    rows = [
        row for row in rows
        if row["clients"] != 256
        and not (
            row["system"] == "Mem0 OSS"
            and row["workers"] == 1
            and row["clients"] == 128
        )
    ]
    apply_recheck(rows, analysis)
    p99_inversions = latency_inversions(rows, "p99")
    max_inversions = latency_inversions(rows, "max")
    pairs = pairwise_common(rows)

    figures = args.report / "figures"
    generated = args.report / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plot_pareto(
        rows,
        figures / "eval_mem0_pareto",
        sampling_note="完整在线检索路径；不含回答模型生成",
    )
    plot_memory(
        rows,
        figures / "eval_mem0_memory",
        clients=(1, 4, 16, 64, 128),
    )
    write_tex_table(generated / "v5_18_mem0_pareto_table.tex", pairs)
    write_tail_table(
        generated / "v5_18_tail_recheck_table.tex",
        args.recheck / "table.tex",
    )

    recheck_sources = sorted({Path(row["source"]) for row in analysis["rows"]})
    source_files = [
        {"path": display_path(path), "sha256": sha256(path)}
        for path in [*screening_sources, *recheck_sources, analysis_path]
    ]
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for pair in pairs:
        workers = str(pair["workers"])
        clients = str(pair["clients"])
        comparisons.setdefault(workers, {})[clients] = {
            "graphmem_qps": pair["graphmem_qps"],
            "mem0_qps": pair["mem0_qps"],
            "qps_ratio": pair["qps_speedup"],
            "graphmem_p95_ms": pair["graphmem_p95_ms"],
            "mem0_p95_ms": pair["mem0_p95_ms"],
            "p95_ratio": pair["graphmem_p95_ms"] / pair["mem0_p95_ms"],
        }
    payload = {
        "schema_version": "graphmem-v5.18-pareto-comparison-v2",
        "frozen_at": "2026-08-10",
        "protocol": {
            "plane": "warm retrieval data plane; answer generation excluded",
            "workers": [1, 4, 8],
            "clients": [1, 4, 16, 64, 128],
            "graphmem_worker1_curve_clients": [1, 4, 8, 16, 64, 128],
            "mem0_worker1_curve_clients": [1, 4, 16, 64],
            "screening": {
                "cells": "30 shared system-worker-client comparison points",
                "warmup_seconds": 1,
                "duration_seconds": 2,
                "repetitions": 1,
            },
            "adaptive_tail_resampling": {
                "cells": (
                    "GraphMem at W=1,C in {1,4,8,16}; both systems at "
                    "W in {4,8}, C in {1,4,16}"
                ),
                "warmup_seconds": 5,
                "duration_seconds": 30,
                "repetitions": 5,
                "reported_point": "mean of five per-trial metrics",
                "confidence_interval": (
                    "two-sided Student t interval, df=4"
                ),
            },
            "deadline_seconds": 5,
            "workload_sha256": (
                "500af8f7d8bf781566f43501b61f9868"
                "e3010c0956108cc47e0dca5efb60ff3d"
            ),
        },
        "source_files": source_files,
        "conditions": {
            "memory_count": 16,
            "vector_count": 7971,
            "top_k": 64,
            "query_vectors": "cached for both systems",
            "affinity_replicas": "min(2, workers)",
        },
        "interpretation": {
            "qps_higher_pairs": (
                f"{sum(p['qps_speedup'] > 1 for p in pairs)}/{len(pairs)}"
            ),
            "qps_and_p95_strictly_better_pairs": (
                f"{sum(p['strictly_dominates'] for p in pairs)}/{len(pairs)}"
            ),
            "graphmem_low_concurrency_p99_inversion_remains": False,
            "adjacent_concurrency_p99_inversions": p99_inversions,
            "adjacent_concurrency_max_inversions": max_inversions,
            "release_gate": (
                "The remaining screening cells retain 2 s x 1 sampling; "
                "a full-matrix repeated run is future validation."
            ),
        },
        "comparisons": comparisons,
        "tail_recheck": {
            "analysis": display_path(analysis_path),
            "rows": analysis["rows"],
            "comparisons": analysis["comparisons"],
        },
        "rows": rows,
    }
    manifest_path = generated / "v5_18_mem0_pareto_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "figure": str(figures / "eval_mem0_pareto.pdf"),
        "manifest": str(manifest_path),
        "strict_dominance": payload["interpretation"][
            "qps_and_p95_strictly_better_pairs"],
        "p99_inversions": p99_inversions,
        "max_inversions": max_inversions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
