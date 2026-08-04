#!/usr/bin/env python3
"""Freeze a read-only V4.1 navigation baseline for the V5 development set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.manifests import create_run_manifest, file_sha256  # noqa: E402
from graphmem.domain import stable_id  # noqa: E402
from graphmem_demo.v36 import prompt_hash as v36_prompt_hash  # noqa: E402
from graphmem_demo.v41.schema import V41_POLICY_VERSION  # noqa: E402


VARIANT = "hierarchical_hybrid_graph_v4_1_query"


def read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def stratum(benchmark: str, row: dict[str, Any]) -> str:
    if benchmark == "longmemeval":
        return "lme_multi_session" if row["question_type"] == "multi-session" else "lme_temporal"
    return "locomo_multihop" if int(row["locomo_category"]) == 1 else "locomo_temporal"


def run_files(run_root: Path) -> tuple[Path, Path, Path]:
    variant = run_root / VARIANT if run_root.name != VARIANT else run_root
    return (
        variant / "retrieval_results.jsonl",
        variant / "question_stats.jsonl",
        variant / "llm_calls.jsonl",
    )


def audit_benchmark(
    benchmark: str, cases: list[dict[str, Any]], run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    ids = {str(row["question_id"]) for row in cases}
    retrieval_path, stats_path, calls_path = run_files(run_root)
    retrieval = {str(row["question_id"]): row for row in read_jsonl(retrieval_path) if str(row["question_id"]) in ids}
    stats = {str(row["question_id"]): row for row in read_jsonl(stats_path) if str(row["question_id"]) in ids}
    if set(retrieval) != ids or set(stats) != ids:
        raise ValueError(
            f"{benchmark} baseline incomplete: retrieval={len(retrieval)}, stats={len(stats)}, expected={len(ids)}"
        )
    calls = [row for row in read_jsonl(calls_path) if str(row.get("question_id")) in ids]
    unique_calls: dict[str, dict[str, Any]] = {}
    for row in calls:
        unique_calls.setdefault(str(row.get("call_id") or stable_call_key(row)), row)
    if any(int(row.get("reasoning_tokens") or 0) for row in unique_calls.values()):
        raise ValueError(f"{benchmark} provider calls contain reasoning tokens")

    rows: list[dict[str, Any]] = []
    by_case = {str(row["question_id"]): row for row in cases}
    for qid in sorted(ids):
        ret, stat, case = retrieval[qid], stats[qid], by_case[qid]
        recall = float(ret.get("answer_session_recall", ret.get("retrieved_answer_session_recall", 0.0)) or 0.0)
        rows.append({
            "benchmark": benchmark, "question_id": qid,
            "stratum": stratum(benchmark, case),
            "session_any_hit": recall > 0.0, "session_all_hit": recall == 1.0,
            "session_recall": recall,
            "retrieved_session_count": len(ret.get("retrieved_session_ids") or []),
            "packed_rough_tokens": int(ret.get("packed_rough_tokens") or 0),
            "retrieval_latency_sec": float(stat.get("retrieval_latency_sec") or ret.get("latency_sec") or 0.0),
            "query_tokens": int(stat.get("answer_total_tokens") or 0),
            "reasoning_tokens": int(stat.get("reasoning_tokens") or 0),
            "leaf_count": int(stat.get("leaf_count") or 0),
            "summary_count": int(stat.get("summary_count") or 0),
            "edge_count": int(stat.get("edge_count") or ret.get("edge_count") or 0),
            "retrieval_fingerprint": hashlib.sha256(json.dumps({
                "routing_card_ids": ret.get("routing_card_ids") or [],
                "fact_node_ids": ret.get("fact_node_ids") or [],
                "leaf_node_ids": ret.get("leaf_node_ids") or [],
                "evidence_leaf_ids": ret.get("evidence_leaf_ids") or [],
                "session_ids": ret.get("retrieved_session_ids") or [],
                "packed_rough_tokens": ret.get("packed_rough_tokens") or 0,
            }, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest(),
        })

    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    summary_groups = {}
    for name, items in sorted(groups.items()):
        summary_groups[name] = {
            "questions": len(items),
            "session_any_hit": sum(row["session_any_hit"] for row in items) / len(items),
            "session_all_hit": sum(row["session_all_hit"] for row in items) / len(items),
            "session_recall": statistics.fmean(row["session_recall"] for row in items),
            "query_tokens": distribution([row["query_tokens"] for row in items]),
            "retrieval_latency_sec": distribution([row["retrieval_latency_sec"] for row in items]),
        }
    call_stage: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_calls.values():
        call_stage[str(row.get("stage") or "unknown")].append(row)
    token_usage = {
        stage: {
            "calls": len(items),
            "cached_input_tokens": sum(int(row.get("prompt_cache_hit_tokens") or 0) for row in items),
            "uncached_input_tokens": sum(int(row.get("prompt_cache_miss_tokens") or row.get("prompt_tokens") or 0) for row in items),
            "output_tokens": sum(int(row.get("completion_tokens") or 0) for row in items),
            "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in items),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in items),
        }
        for stage, items in sorted(call_stage.items())
    }
    variant_root = retrieval_path.parent
    frozen_files = [retrieval_path, stats_path, calls_path]
    frozen_files.extend(
        path for path in (
            variant_root / "nodes.jsonl", variant_root / "edges.jsonl",
            variant_root / "index_diagnostics.jsonl", variant_root / "embedding_calls.jsonl",
        ) if path.exists()
    )
    summary = {
        "benchmark": benchmark, "questions": len(rows), "strata": summary_groups,
        "overall": {
            "session_any_hit": sum(row["session_any_hit"] for row in rows) / len(rows),
            "session_all_hit": sum(row["session_all_hit"] for row in rows) / len(rows),
            "session_recall": statistics.fmean(row["session_recall"] for row in rows),
            "query_tokens": distribution([row["query_tokens"] for row in rows]),
            "retrieval_latency_sec": distribution([row["retrieval_latency_sec"] for row in rows]),
            "leaf_count": distribution([row["leaf_count"] for row in rows]),
            "summary_count": distribution([row["summary_count"] for row in rows]),
            "edge_count": distribution([row["edge_count"] for row in rows]),
        },
        "token_usage_by_stage": token_usage,
        "source_files": {
            str(path): file_sha256(path)
            for path in frozen_files
        },
    }
    return rows, summary, list(unique_calls.values())


def stable_call_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lme-data", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--lme-run", type=Path, required=True)
    parser.add_argument("--locomo-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v5/gate_a.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    lme_rows, lme_summary, lme_calls = audit_benchmark("longmemeval", read_json(args.lme_data), args.lme_run)
    locomo_rows, locomo_summary, locomo_calls = audit_benchmark("locomo", read_json(args.locomo_data), args.locomo_run)
    if any(row["reasoning_tokens"] for row in [*lme_rows, *locomo_rows]):
        raise ValueError("baseline contains non-zero reasoning tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "question_metrics.jsonl", [*lme_rows, *locomo_rows])
    write_csv(args.output_dir / "metrics.csv", [*lme_rows, *locomo_rows])
    errors = [row | {"failure_class": "legacy_session_miss"}
              for row in [*lme_rows, *locomo_rows] if not row["session_all_hit"]]
    write_jsonl(args.output_dir / "error_cases.jsonl", errors)
    write_json(args.output_dir / "baseline_metrics.json", {
        "schema_version": "graphmem-v5-gate-a-audit-v1",
        "longmemeval": lme_summary, "locomo": locomo_summary,
    })
    manifest = create_run_manifest(
        repo=ROOT, dataset_paths=[args.lme_data, args.locomo_data], config=config,
        prompt_hashes={"v36_build": v36_prompt_hash(), "v41_policy": V41_POLICY_VERSION},
    )
    payload = asdict(manifest)
    payload["audit_inputs"] = {
        "config": str(args.config), "config_file_sha256": file_sha256(args.config),
        "lme_run": str(args.lme_run), "locomo_run": str(args.locomo_run),
    }
    payload["unique_provider_calls"] = len({
        str(row.get("call_id") or stable_call_key(row)) for row in [*lme_calls, *locomo_calls]
    })
    payload["frozen_graph_artifacts"] = []
    for name, summary in (("longmemeval", lme_summary), ("locomo", locomo_summary)):
        source_hashes = summary["source_files"]
        payload["frozen_graph_artifacts"].append({
            "benchmark": name,
            "artifact_id": stable_id("graph-artifact", name, sorted(source_hashes.values())),
            "source_files": source_hashes,
        })
    write_json(args.output_dir / "run_manifest.json", payload)
    token_rows = []
    for name, summary in (("longmemeval", lme_summary), ("locomo", locomo_summary)):
        for stage, usage in summary["token_usage_by_stage"].items():
            token_rows.append({"benchmark": name, "stage": stage, **usage})
    write_csv(args.output_dir / "token_usage.csv", token_rows)
    print(json.dumps({
        "questions": len(lme_rows) + len(locomo_rows),
        "lme_session_all_hit": lme_summary["overall"]["session_all_hit"],
        "locomo_session_all_hit": locomo_summary["overall"]["session_all_hit"],
    }, indent=2))


if __name__ == "__main__":
    main()
