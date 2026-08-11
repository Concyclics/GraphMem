#!/usr/bin/env python3
"""Summarize the V5.18 low-concurrency tail-latency sample recheck."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
T_CRITICAL_95_DF4 = 2.776
METRICS = ("qps", "p50", "p95", "p99", "max")
SYSTEM_INPUTS = {
    "GraphMem": "graphmem_w{workers}_30s5",
    "Mem0 OSS": "mem0_w{workers}_30s5{suffix}",
}
SAMPLE_CELLS = {
    "GraphMem": {1: (1, 4, 8, 16), 4: (1, 4, 16), 8: (1, 4, 16)},
    "Mem0 OSS": {4: (1, 4, 16), 8: (1, 4, 16)},
}


def input_dir_for(system: str, workers: int, clients: int) -> str:
    if system == "GraphMem" and workers == 1:
        return "graphmem_w1_c1_4_8_16_30s5_v2"
    if clients == 16:
        prefix = "graphmem" if system == "GraphMem" else "mem0"
        return f"{prefix}_w{workers}_c16_30s5"
    suffix = "_v2" if system == "Mem0 OSS" and workers == 4 else ""
    return SYSTEM_INPUTS[system].format(workers=workers, suffix=suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(WORKSPACE / "artifacts/report/v5_18/"
                 "tail_latency_recheck_20260810"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=WORKSPACE / "artifacts/report/v5_18",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * quantile) - 1),
    )
    return ordered[index]


def trial_values(trials: list[dict[str, Any]], metric: str) -> list[float]:
    if metric == "qps":
        return [float(row["qps"]) for row in trials]
    return [float(row["latency_ms"][metric]) for row in trials]


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        T_CRITICAL_95_DF4 * standard_deviation / math.sqrt(len(values))
        if len(values) == 5 else 0.0
    )
    return {
        "mean": mean,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "standard_deviation": standard_deviation,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_half_width": half_width,
    }


def request_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        trace = trial.get("request_trace")
        if not trace:
            continue
        rows.extend(
            json.loads(line)
            for line in Path(trace).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    latencies = [float(row["latency_ms"]) for row in completed]
    slow = [row for row in completed if float(row["service_ms"]) >= 100.0]
    dominant_stages: Counter[str] = Counter()
    slow_memories: Counter[str] = Counter()
    stage_over_100_ms: Counter[str] = Counter()
    for row in slow:
        stages = {
            key: float(value)
            for key, value in row["stage_latency_ms"].items()
            if key != "total"
        }
        dominant_stages[max(stages, key=stages.get)] += 1
        slow_memories[str(row["memory_id"])] += 1
        for stage, value in stages.items():
            if value >= 100.0:
                stage_over_100_ms[stage] += 1
    return {
        "rows": len(rows),
        "completed": len(completed),
        "slow_service_ge_100_ms": len(slow),
        "slow_service_rate": len(slow) / max(1, len(completed)),
        "pooled_latency_ms": {
            "p50": nearest_rank(latencies, 0.50),
            "p95": nearest_rank(latencies, 0.95),
            "p99": nearest_rank(latencies, 0.99),
            "max": max(latencies, default=0.0),
        },
        "dominant_stage_for_slow_requests": dict(dominant_stages),
        "stages_ge_100_ms": dict(stage_over_100_ms),
        "slow_memories": dict(slow_memories),
    }


def cell_by_clients(summary: dict[str, Any], clients: int) -> dict[str, Any]:
    return next(row for row in summary["cells"] if int(row["clients"]) == clients)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# V5.18 low-concurrency tail-latency sample recheck",
        "",
        "Protocol: warm retrieval data plane, cached Query vectors, "
        "5 s warmup, 30 s measurement, five independent trials per point, "
        "5 s deadline; request-level stage trace enabled for GraphMem.",
        "",
        "| system | workers | clients | successful requests | QPS mean | p50 mean | "
        "p95 mean | p99 mean ± 95% CI | max mean ± 95% CI |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['system']} | {row['workers']} | {row['clients']} | "
            f"{row['completed']} | "
            f"{metrics['qps']['mean']:.2f} | {metrics['p50']['mean']:.2f} ms | "
            f"{metrics['p95']['mean']:.2f} ms | "
            f"{metrics['p99']['mean']:.2f} ± "
            f"{metrics['p99']['ci95_half_width']:.2f} ms | "
            f"{metrics['max']['mean']:.2f} ± "
            f"{metrics['max']['ci95_half_width']:.2f} ms |"
        )
    lines.extend(["", "## Inversion check", ""])
    for system, system_comparisons in result["comparisons"].items():
        for workers, comparison in system_comparisons.items():
            lines.append(
                f"- {system}, {workers} workers: C=1 versus C=4 mean p99 "
                f"{comparison['c1_p99_mean_ms']:.2f}/"
                f"{comparison['c4_p99_mean_ms']:.2f} ms; mean max "
                f"{comparison['c1_max_mean_ms']:.2f}/"
                f"{comparison['c4_max_mean_ms']:.2f} ms; "
                f"p99 inversion remains="
                f"{comparison['p99_inversion_remains']}."
            )
            lines.append(
                f"  Adjacent p99 monotonicity C="
                f"{'/'.join(map(str, comparison['client_sequence']))}: "
                f"{comparison['p99_monotonic']}; values="
                f"{comparison['p99_sequence_ms']}."
            )
    lines.extend(["", "## Tail attribution", ""])
    for row in result["rows"]:
        if row["trace"] is None:
            continue
        trace = row["trace"]
        lines.append(
            f"- {row['system']}, W={row['workers']}, C={row['clients']}: "
            f"service >=100 ms in {trace['slow_service_ge_100_ms']}/"
            f"{trace['completed']} requests ({trace['slow_service_rate'] * 100:.3f}%); "
            f"dominant stages={trace['dominant_stage_for_slow_requests']}."
        )
    lines.append("")
    return "\n".join(lines)


def render_latex(result: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated by analyze_v5_18_tail_recheck.py",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "系统 & $W,C$ & 成功请求 & QPS & p99（ms） & max（ms） \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(result["rows"]):
        metrics = row["metrics"]
        lines.append(
            f"{row['system']} & {row['workers']},{row['clients']} & "
            f"{row['completed']:,} & {metrics['qps']['mean']:.2f} & "
            f"{metrics['p99']['mean']:.1f} $\\pm$ "
            f"{metrics['p99']['ci95_half_width']:.1f} & "
            f"{metrics['max']['mean']:.1f} $\\pm$ "
            f"{metrics['max']['ci95_half_width']:.1f} \\\\"
        )
        if index + 1 < len(result["rows"]) and (
            result["rows"][index + 1]["system"] != row["system"]
        ):
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    comparisons: dict[str, Any] = {}
    for system in SYSTEM_INPUTS:
        system_comparisons: dict[str, Any] = {}
        for workers, client_sequence in SAMPLE_CELLS[system].items():
            baseline_dir = (
                f"qps_w{workers}_optimized_prewarm"
                if system == "GraphMem"
                else f"mem0_qps_w{workers}"
            )
            baseline = load(args.baseline / baseline_dir / "summary.json")
            by_clients: dict[int, dict[str, Any]] = {}
            for clients in client_sequence:
                input_dir = input_dir_for(system, workers, clients)
                summary = load(args.input / input_dir / "summary.json")
                cell = cell_by_clients(summary, clients)
                trials = cell["trials"]
                metrics = {
                    metric: summarize(trial_values(trials, metric))
                    for metric in METRICS
                }
                baseline_cell = (
                    cell_by_clients(baseline, clients)
                    if clients != 8 else None
                )
                row = {
                    "system": system,
                    "workers": workers,
                    "clients": clients,
                    "duration_seconds": summary["duration_sec"],
                    "warmup_seconds": summary["warmup_sec"],
                    "repetitions": summary["repetitions"],
                    "completed": sum(int(trial["completed"]) for trial in trials),
                    "timed_out": sum(int(trial["timed_out"]) for trial in trials),
                    "failed": sum(int(trial["failed"]) for trial in trials),
                    "rejected": sum(int(trial["rejected"]) for trial in trials),
                    "worker_rss_mib": float(cell["total_worker_rss_mib"]),
                    "worker_pss_mib": float(cell["total_worker_pss_mib"]),
                    "metrics": metrics,
                    "trace": (
                        trace_summary(request_rows(trials))
                        if system == "GraphMem" else None
                    ),
                    "source": str((args.input / input_dir / "summary.json").resolve()),
                    "single_run_baseline": (
                        {
                            "p99_ms": float(baseline_cell["aggregate"]
                                            ["latency_ms"]["p99"]["mean"]),
                            "max_ms": float(baseline_cell["aggregate"]
                                           ["latency_ms"]["max"]["mean"]),
                        }
                        if baseline_cell is not None else None
                    ),
                }
                rows.append(row)
                by_clients[clients] = row
            system_comparisons[str(workers)] = {
                "c1_p99_mean_ms": by_clients[1]["metrics"]["p99"]["mean"],
                "c4_p99_mean_ms": by_clients[4]["metrics"]["p99"]["mean"],
                "c1_max_mean_ms": by_clients[1]["metrics"]["max"]["mean"],
                "c4_max_mean_ms": by_clients[4]["metrics"]["max"]["mean"],
                "p99_inversion_remains": (
                    by_clients[1]["metrics"]["p99"]["mean"]
                    > by_clients[4]["metrics"]["p99"]["mean"]
                ),
                "max_inversion_remains": (
                    by_clients[1]["metrics"]["max"]["mean"]
                    > by_clients[4]["metrics"]["max"]["mean"]
                ),
                "client_sequence": list(client_sequence),
                "p99_sequence_ms": [
                    by_clients[clients]["metrics"]["p99"]["mean"]
                    for clients in client_sequence
                ],
                "max_sequence_ms": [
                    by_clients[clients]["metrics"]["max"]["mean"]
                    for clients in client_sequence
                ],
                "p99_monotonic": all(
                    by_clients[left]["metrics"]["p99"]["mean"]
                    <= by_clients[right]["metrics"]["p99"]["mean"]
                    for left, right in zip(client_sequence, client_sequence[1:])
                ),
                "max_monotonic": all(
                    by_clients[left]["metrics"]["max"]["mean"]
                    <= by_clients[right]["metrics"]["max"]["mean"]
                    for left, right in zip(client_sequence, client_sequence[1:])
                ),
            }
        comparisons[system] = system_comparisons
    result = {
        "schema_version": "graphmem-v5.18-tail-recheck-v1",
        "protocol": {
            "plane": "warm retrieval data plane",
            "systems": list(SYSTEM_INPUTS),
            "workers": [1, 4, 8],
            "clients_by_system_worker": SAMPLE_CELLS,
            "duration_seconds": 30,
            "warmup_seconds": 5,
            "repetitions": 5,
            "deadline_seconds": 5,
            "confidence_interval": (
                "two-sided Student t interval over five per-trial metrics, df=4"
            ),
        },
        "rows": rows,
        "comparisons": comparisons,
    }
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "analysis.md").write_text(
        render_markdown(result), encoding="utf-8")
    (output / "table.tex").write_text(
        render_latex(result), encoding="utf-8")
    print(json.dumps({
        "analysis": str(output / "analysis.json"),
        "markdown": str(output / "analysis.md"),
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
