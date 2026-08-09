#!/usr/bin/env python3
"""Render the same-workload GraphMem/Mem0 V5.18 Pareto comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=WORKSPACE / "artifacts/report/v5_18")
    parser.add_argument("--output", type=Path,
                        default=WORKSPACE / "artifacts/report/v5_18/figures")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def value(cell: dict, section: str, metric: str | None = None) -> float:
    row = cell["aggregate"][section]
    return float(row["mean"] if metric is None else row[metric]["mean"])


def main() -> None:
    args = parse_args()
    systems: dict[str, dict[int, dict]] = {"GraphMem": {}, "Mem0": {}}
    for workers in (1, 4, 8):
        systems["GraphMem"][workers] = load(
            args.root / f"qps_w{workers}_optimized_prewarm/summary.json")
        systems["Mem0"][workers] = load(
            args.root / f"mem0_qps_w{workers}/summary.json")

    colors = {"GraphMem": "#2563eb", "Mem0": "#e11d48"}
    markers = {"GraphMem": "o", "Mem0": "s"}
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), sharex="col")
    for column, workers in enumerate((1, 4, 8)):
        for system in ("GraphMem", "Mem0"):
            cells = systems[system][workers]["cells"]
            clients = [int(row["clients"]) for row in cells]
            qps = [value(row, "qps") for row in cells]
            p95 = [value(row, "latency_ms", "p95") for row in cells]
            axes[0, column].plot(
                clients, qps, color=colors[system], marker=markers[system],
                linewidth=2.0, markersize=4.5, label=system)
            axes[1, column].plot(
                clients, p95, color=colors[system], marker=markers[system],
                linewidth=2.0, markersize=4.5, label=system)
        for row in axes[:, column]:
            row.set_xscale("log", base=2)
            row.set_xticks((1, 4, 16, 64, 256))
            row.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            row.grid(True, which="major", linewidth=0.5, alpha=0.25)
        axes[0, column].set_title(f"{workers} worker{'s' if workers > 1 else ''}")
        axes[1, column].set_yscale("log")
        graph_pss = systems["GraphMem"][workers]["cells"][4]["total_worker_pss_mib"]
        mem0_pss = systems["Mem0"][workers]["cells"][4]["total_worker_pss_mib"]
        axes[1, column].text(
            .98, .05, f"PSS @ C128\n{graph_pss/1024:.2f} / {mem0_pss/1024:.2f} GiB",
            transform=axes[1, column].transAxes, ha="right", va="bottom", fontsize=8)
        axes[1, column].set_xlabel("Concurrent users")
    axes[0, 0].set_ylabel("Throughput (QPS) ↑")
    axes[1, 0].set_ylabel("End-to-end p95 (ms, log) ↓")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, 1.01))
    figure.suptitle(
        "GraphMem V5.18 vs Mem0: warm retrieval data-plane Pareto",
        y=1.045, fontsize=13, fontweight="bold")
    figure.text(
        .5, .005,
        "Same 16-memory / 7,971-vector workload; top-k/evidence=64; cached query vectors; "
        "GraphMem affinity prewarm",
        ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, .035, 1, .98))
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(args.output / f"graphmem_mem0_dense_pareto.{suffix}",
                       dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)

    comparisons = {}
    for workers in (1, 4, 8):
        graph_cells = {int(row["clients"]): row
                       for row in systems["GraphMem"][workers]["cells"]}
        mem0_cells = {int(row["clients"]): row
                      for row in systems["Mem0"][workers]["cells"]}
        comparisons[str(workers)] = {}
        for clients in sorted(graph_cells):
            graph = graph_cells[clients]
            mem0 = mem0_cells[clients]
            comparisons[str(workers)][str(clients)] = {
                "graphmem_qps": value(graph, "qps"),
                "mem0_qps": value(mem0, "qps"),
                "qps_ratio": value(graph, "qps") / value(mem0, "qps"),
                "graphmem_p95_ms": value(graph, "latency_ms", "p95"),
                "mem0_p95_ms": value(mem0, "latency_ms", "p95"),
                "p95_ratio": (value(graph, "latency_ms", "p95")
                              / value(mem0, "latency_ms", "p95")),
                "graphmem_pss_mib": float(graph["total_worker_pss_mib"]),
                "mem0_pss_mib": float(mem0["total_worker_pss_mib"]),
            }
    (args.output / "comparison_summary.json").write_text(
        json.dumps({
            "schema_version": "graphmem-v5.18-pareto-comparison-v1",
            "conditions": {
                "memory_count": 16,
                "vector_count": 7971,
                "top_k": 64,
                "query_vectors": "cached for both systems",
                "graphmem_prewarm": "all memories on rendezvous affinity replicas",
            },
            "comparisons": comparisons,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "workers": [1, 4, 8]}, indent=2))


if __name__ == "__main__":
    main()
