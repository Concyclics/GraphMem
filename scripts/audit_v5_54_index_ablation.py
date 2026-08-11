#!/usr/bin/env python3
"""Audit the V5.54 hierarchy x relation-traversal factorial experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping


ARMS = {
    "seed_only": (False, False),
    "hierarchy_only": (True, False),
    "flat_graph": (False, True),
    "full": (True, True),
}
BUDGETS = (32, 64)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def by_id(
        data: list[Mapping[str, Any]],
        key: str = "question_id",
) -> dict[str, Mapping[str, Any]]:
    result = {str(row[key]): row for row in data}
    if len(result) != len(data):
        raise ValueError(f"duplicate {key} in audit input")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority_graph_checksum(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        for row in connection.execute(
                "SELECT memory_id,node_xor,node_sum,node_count,edge_xor,edge_sum,"
                "edge_count,algorithm FROM graph_checksum_state ORDER BY memory_id"):
            digest.update(json.dumps(row, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    finally:
        connection.close()


def prediction_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(row.get("prediction", "")).encode("utf-8")).hexdigest()


def nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def usage_statistics(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values), "mean": sum(values) / max(1, len(values)),
        "p50": nearest_rank(values, 0.50), "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99), "max": max(values),
    }


def prepare_audit(root: Path, expected: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "graphmem-v5.54-index-ablation-audit-v1",
        "root": str(root), "expected_questions": expected,
        "arms": {}, "pairwise_prompt_identity": {},
    }
    authority_source = None
    ids_by_arm: dict[tuple[int, str], set[str]] = {}
    prepared_by_arm: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = {}
    failures: list[str] = []
    for budget in BUDGETS:
        for arm, (hierarchy, traversal) in ARMS.items():
            arm_root = root / f"turn{budget}" / arm / "prepare"
            manifest_path = arm_root / "prepare_manifest.json"
            prepared_path = arm_root / "prepared_answers.jsonl"
            retrieval_path = arm_root / "retrieval.jsonl"
            manifest = read(manifest_path)
            prepared = rows(prepared_path)
            retrieval = rows(retrieval_path)
            prepared_map = by_id(prepared, "question_id")
            retrieval_map = by_id(retrieval, "dev_question_id")
            question_ids = set(prepared_map)
            ids_by_arm[(budget, arm)] = question_ids
            prepared_by_arm[(budget, arm)] = prepared_map
            signal_traversals = sum(
                sum(int(value) for value in dict(
                    row.get("traversed_relation_signals") or {}).values())
                for row in retrieval)
            hierarchy_route_nonzero = sum(
                float(row.get("latency_hierarchical_route_ms") or 0) > 0
                for row in retrieval)
            checks = {
                "questions": len(prepared) == expected == len(retrieval),
                "ids_match": question_ids == set(retrieval_map),
                "answer_policy_v5_54": manifest.get("answer_policy") == "v5_54",
                "answer_calls_zero": manifest.get("answer_calls") == 0,
                "turn_budget": manifest.get("max_evidence_turns") == budget,
                "token_budget": manifest.get("max_evidence_tokens") == 12_000,
                "hierarchical_routing": manifest.get(
                    "hierarchical_routing") is hierarchy,
                "graph_traversal": manifest.get("graph_traversal") is traversal,
                "packed_turn_cap": all(
                    int(row.get("packed_turns") or 0) <= budget
                    for row in retrieval),
                "disabled_traversal_unused": traversal or signal_traversals == 0,
                "disabled_hierarchy_unused": hierarchy or hierarchy_route_nonzero == 0,
            }
            for key, passed in checks.items():
                if not passed:
                    failures.append(f"turn{budget}/{arm}: {key}")
            source = str(manifest.get("source_db") or "")
            authority_source = authority_source or source
            if source != authority_source:
                failures.append(f"turn{budget}/{arm}: source_db differs")
            payload["arms"][f"turn{budget}/{arm}"] = {
                "checks": checks,
                "manifest": str(manifest_path),
                "runtime_config": manifest.get("runtime_config"),
                "runtime_config_hash": manifest.get("runtime_config_hash"),
                "prepared_sha256": sha256(prepared_path),
                "retrieval_sha256": sha256(retrieval_path),
                "relation_signal_traversals": signal_traversals,
                "hierarchy_route_nonzero_questions": hierarchy_route_nonzero,
                "mean_packed_turns": sum(int(row.get("packed_turns") or 0)
                                         for row in retrieval) / max(1, len(retrieval)),
            }

    reference_ids = ids_by_arm[(64, "full")]
    for key, ids in ids_by_arm.items():
        if ids != reference_ids:
            failures.append(f"turn{key[0]}/{key[1]}: question IDs differ")
    for budget in BUDGETS:
        baseline = prepared_by_arm[(budget, "seed_only")]
        for arm in ("hierarchy_only", "flat_graph", "full"):
            candidate = prepared_by_arm[(budget, arm)]
            common = sorted(set(baseline) & set(candidate))
            same_prompt = sum(
                baseline[item].get("prompt_payload_hash")
                == candidate[item].get("prompt_payload_hash") for item in common)
            same_evidence = sum(
                baseline[item].get("evidence_turn_ids")
                == candidate[item].get("evidence_turn_ids") for item in common)
            payload["pairwise_prompt_identity"][
                f"turn{budget}/seed_only->{arm}"] = {
                    "questions": len(common),
                    "same_prompt": same_prompt,
                    "same_evidence_and_order": same_evidence,
                    "changed_prompt": len(common) - same_prompt,
                }
    payload["authority_source_db"] = authority_source
    payload["authority_graph_checksum"] = authority_graph_checksum(
        Path(str(authority_source)))
    payload["failures"] = failures
    payload["passed"] = not failures
    output = root / "prepare_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("; ".join(failures[:12]))
    return payload


def final_audit(root: Path, expected: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "graphmem-v5.54-index-ablation-final-audit-v1",
        "root": str(root), "expected_questions": expected, "arms": {},
    }
    failures: list[str] = []
    for budget in BUDGETS:
        for arm in ARMS:
            label = f"turn{budget}/{arm}"
            prepare_root = root / f"turn{budget}" / arm / "prepare"
            answer_root = root / f"turn{budget}" / arm / "answer"
            prepared = by_id(rows(prepare_root / "prepared_answers.jsonl"))
            answers = by_id(rows(answer_root / "answers.jsonl"))
            usage = by_id(rows(answer_root / "answer_usage.jsonl"))
            manifest = read(answer_root / "run_manifest.json")
            expected_ids = set(prepared)
            prompt_hash_ok = all(
                str(prepared[item].get("prompt_payload_hash") or "")
                == str(answers.get(item, {}).get("prompt_payload_hash") or "")
                == str(usage.get(item, {}).get("prompt_payload_hash") or "")
                for item in expected_ids)
            additive = all(
                int(row.get("total_tokens") or 0)
                == int(row.get("api_prompt_tokens") or 0)
                + int(row.get("completion_tokens") or 0)
                for row in usage.values())
            token_values = {
                "prompt": [int(row.get("api_prompt_tokens") or 0)
                           for row in usage.values()],
                "completion": [int(row.get("completion_tokens") or 0)
                               for row in usage.values()],
                "total": [int(row.get("total_tokens") or 0)
                           for row in usage.values()],
            }
            recomputed = {key: usage_statistics(values)
                          for key, values in token_values.items()}
            stats_ok = True
            for key, stats in recomputed.items():
                declared = dict(manifest.get("api_tokens", {}).get(key) or {})
                stats_ok &= all(
                    abs(float(declared.get(field, float("nan"))) - float(value)) < 1e-8
                    for field, value in stats.items())
            sums = {key: sum(values) for key, values in token_values.items()}
            declared_sums = dict(manifest.get("api_usage_sums") or {})
            sums_ok = all(int(declared_sums.get(key, -1)) == value
                          for key, value in sums.items())
            checks = {
                "questions": len(prepared) == len(answers) == len(usage) == expected,
                "ids_match": set(answers) == set(usage) == expected_ids,
                "prompt_hashes_match": prompt_hash_ok,
                "answer_model": manifest.get("answer_model")
                == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                "max_output_tokens": manifest.get("max_output_tokens") == 2000,
                "usage_additive": additive,
                "usage_sums_match": sums_ok,
                "nearest_rank_statistics_match": stats_ok,
                "manifest_completed": manifest.get("completed_questions") == expected,
                "manifest_prompt_audit": manifest.get(
                    "prompt_identity_audit", {}).get("prompt_hash_mismatches") == 0,
            }
            judges: dict[str, Any] = {}
            for benchmark, count in (("longmemeval", 500), ("locomo", 1540)):
                verdict_path = answer_root / f"judge_{benchmark}" / "paired_verdicts.jsonl"
                verdicts = by_id(rows(verdict_path))
                benchmark_answers = {
                    item: row for item, row in answers.items()
                    if row.get("benchmark") == benchmark}
                aligned = all(
                    verdicts[item].get("prediction_sha256")
                    == prediction_sha256(benchmark_answers[item])
                    for item in benchmark_answers if item in verdicts)
                judge_checks = {
                    "questions": len(verdicts) == len(benchmark_answers) == count,
                    "ids_match": set(verdicts) == set(benchmark_answers),
                    "prediction_hashes_match": aligned,
                    "judge_model": all(row.get("judge_model") == "gpt-5.6-luna"
                                       for row in verdicts.values()),
                    "judge_prompt_pinned": all(
                        bool(row.get("judge_prompt_sha256"))
                        and bool(row.get("judge_prompt_commit"))
                        for row in verdicts.values()),
                }
                checks.update({f"judge_{benchmark}_{key}": value
                               for key, value in judge_checks.items()})
                judges[benchmark] = {
                    "checks": judge_checks, "verdicts_sha256": sha256(verdict_path),
                    "correct": sum(bool(row.get("correct"))
                                   for row in verdicts.values()),
                    "judge_prompt_sha256": sorted({str(row.get("judge_prompt_sha256"))
                                                     for row in verdicts.values()}),
                    "resume_dedup_audits": [
                        read(path) for path in sorted(
                            (answer_root / f"judge_{benchmark}").glob(
                                "*dedup_audit*.json"))],
                }
            for key, passed in checks.items():
                if not passed:
                    failures.append(f"{label}: {key}")
            payload["arms"][label] = {
                "checks": checks, "answers_sha256": sha256(answer_root / "answers.jsonl"),
                "usage_sha256": sha256(answer_root / "answer_usage.jsonl"),
                "api_usage_sums": sums, "api_tokens": recomputed, "judges": judges,
            }
    payload["failures"] = failures
    payload["passed"] = not failures
    output = root / "final_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("; ".join(failures[:12]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--expected", type=int, default=2040)
    final = sub.add_parser("final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--expected", type=int, default=2040)
    args = parser.parse_args()
    payload = (prepare_audit(args.root, args.expected)
               if args.command == "prepare" else final_audit(args.root, args.expected))
    print(json.dumps({"passed": payload["passed"],
                      "arms": len(payload["arms"]),
                      "output": str(args.root / (
                          "prepare_audit.json" if args.command == "prepare"
                          else "final_audit.json"))}, indent=2))


if __name__ == "__main__":
    main()
