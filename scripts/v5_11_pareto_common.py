"""Shared deterministic workload and measurement helpers for V5.11 Pareto runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable


CLIENT_LEVELS = (1, 4, 16, 64, 128, 256)
WORKER_LEVELS = (1, 4, 8)


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("list must contain positive integers")
    return values


def parse_cpu_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value < 0 for value in values):
        raise ValueError("CPU list must contain nonnegative integers")
    return values


def percentile(values: Iterable[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    return rows[min(len(rows) - 1, max(0, math.ceil(len(rows) * q) - 1))]


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def process_memory_mib(pid: int) -> dict[str, float]:
    result = {"rss": 0.0, "pss": 0.0, "private_dirty": 0.0,
              "shared_clean": 0.0}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["rss"] = int(line.split()[1]) / 1024
                break
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            mapped = {"Pss": "pss", "Private_Dirty": "private_dirty",
                      "Shared_Clean": "shared_clean"}.get(key)
            if mapped:
                result[mapped] = int(value.split()[0]) / 1024
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass
    return result


def load_workload(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "graphmem-mem0-pareto-workload-v1":
        raise ValueError(f"unsupported workload manifest: {path}")
    by_memory: dict[str, list[dict[str, str]]] = {}
    for row in payload["queries"]:
        by_memory.setdefault(str(row["memory_id"]), []).append({
            "question_id": str(row["question_id"]),
            "query": str(row["query"]),
            "benchmark": str(row["benchmark"]),
            "stratum": str(row["stratum"]),
        })
    if sorted(by_memory) != payload["memory_ids"]:
        raise ValueError("manifest memory_ids do not match query rows")
    return payload, by_memory


def popularity_weights(memory_ids: list[str], alpha: float) -> list[float]:
    return [1.0 / ((index + 1) ** alpha) for index in range(len(memory_ids))]


def affinity_shards(memory_id: str, workers: int, replicas: int) -> tuple[int, ...]:
    ranked = sorted(range(workers), key=lambda shard: (
        -int.from_bytes(hashlib.blake2b(
            f"{memory_id}:{shard}".encode("utf-8"), digest_size=8).digest(),
                        "big"),
        shard,
    ))
    return tuple(ranked[:replicas])


def default_cpu_ids(workers: int) -> tuple[int, ...]:
    try:
        available = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError as error:  # pragma: no cover - Linux benchmark path
        raise RuntimeError("CPU affinity is required for this benchmark") from error
    if len(available) < workers:
        raise RuntimeError(f"only {len(available)} CPUs visible for {workers} workers")
    return available[:workers]


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("cannot summarize zero trials")
    keys = ("qps", "completed", "failed", "timed_out", "rejected")
    aggregate: dict[str, Any] = {"trials": len(trials)}
    for key in keys:
        values = [float(row[key]) for row in trials]
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
    for section in ("latency_ms", "service_ms", "queue_ms"):
        aggregate[section] = {}
        for metric in ("mean", "p50", "p95", "p99", "max"):
            values = [float(row[section][metric]) for row in trials]
            aggregate[section][metric] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
    return aggregate
