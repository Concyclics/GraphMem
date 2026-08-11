#!/usr/bin/env python3
"""Summarize the full V5.54 hierarchy x relation-traversal ablation."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = ("seed_only", "hierarchy_only", "flat_graph", "full")
BUDGETS = (32, 64)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def by_id(data: Sequence[Mapping[str, Any]], key: str = "question_id") -> dict[str, Mapping[str, Any]]:
    result = {str(row[key]): row for row in data}
    if len(result) != len(data):
        raise ValueError(f"duplicate {key}")
    return result


def group_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    benchmark = str(row.get("benchmark") or "")
    stratum = str(row.get("stratum") or "")
    groups = ["overall"]
    if benchmark == "longmemeval":
        groups.append("longmemeval")
        if stratum == "lme_multi_session":
            groups += ["lme_multi_session", "structural", "hard869"]
        elif stratum == "lme_temporal_reasoning":
            groups += ["lme_temporal", "temporal", "hard869"]
    elif benchmark == "locomo":
        groups.append("locomo")
        if stratum == "locomo_cat1":
            groups += ["locomo_multihop", "structural", "hard869"]
        elif stratum == "locomo_cat2":
            groups += ["locomo_temporal", "temporal", "hard869"]
    return tuple(groups)


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = min(gains, losses)
    return min(1.0, 2.0 * sum(
        math.comb(discordant, value) for value in range(tail + 1))
        / (2 ** discordant))


def cluster_bootstrap(values: Mapping[str, float], clusters: Mapping[str, str],
                      *, iterations: int = 10_000) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for question_id, value in values.items():
        grouped[str(clusters[question_id])].append(float(value))
    keys = sorted(grouped)
    if not keys:
        return (0.0, 0.0)
    rng = random.Random(42)
    samples = []
    for _ in range(iterations):
        picked = [rng.choice(keys) for _ in keys]
        flattened = [value for key in picked for value in grouped[key]]
        samples.append(sum(flattened) / max(1, len(flattened)))
    samples.sort()
    return samples[249], samples[9749]


def paired(left: Mapping[str, bool], right: Mapping[str, bool],
           question_ids: Sequence[str], clusters: Mapping[str, str]) -> dict[str, Any]:
    ids = [item for item in question_ids if item in left and item in right]
    gains = sum(not left[item] and right[item] for item in ids)
    losses = sum(left[item] and not right[item] for item in ids)
    values = {item: int(right[item]) - int(left[item]) for item in ids}
    ci = cluster_bootstrap(values, clusters)
    return {
        "questions": len(ids), "gains": gains, "losses": losses,
        "delta": sum(values.values()) / max(1, len(values)),
        "mcnemar_exact_p": exact_mcnemar(gains, losses),
        "memory_cluster_bootstrap_95ci": list(ci),
    }


def mean(data: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in data if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def retrieval_metrics(data: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = [row for row in data if row.get("has_turn_gold")]
    signals = Counter()
    for row in data:
        signals.update({str(key): int(value) for key, value in dict(
            row.get("traversed_relation_signals") or {}).items()})
    return {
        key: mean(annotated if key.startswith("turn_") else data, key)
        for key in (
            "turn_recall", "turn_precision", "turn_f1", "turn_all_hit",
            "candidate_turn_recall", "candidate_turn_precision",
            "visited_nodes", "visited_edges", "latency_total_ms",
            "latency_hierarchical_route_ms", "packed_turns", "prompt_tokens",
            "api_prompt_tokens", "api_total_tokens")
    } | {"traversed_relation_signals": dict(sorted(signals.items())),
         "annotated_questions": len(annotated)}


def summarize_arm(root: Path) -> dict[str, Any]:
    prepare_root = root / "prepare"
    answer_root = root / "answer"
    answers = rows(answer_root / "answers.jsonl")
    usage = rows(answer_root / "answer_usage.jsonl")
    prepared = rows(prepare_root / "prepared_answers.jsonl")
    retrieval = rows(prepare_root / "retrieval.jsonl")
    verdict_rows = (rows(answer_root / "judge_longmemeval" / "paired_verdicts.jsonl")
                    + rows(answer_root / "judge_locomo" / "paired_verdicts.jsonl"))
    answer_map = by_id(answers)
    usage_map = by_id(usage) if usage else {}
    prepared_map = by_id(prepared)
    retrieval_map = by_id(retrieval, "dev_question_id")
    verdicts = {str(row["question_id"]): bool(row.get("correct"))
                for row in verdict_rows}
    expected = set(prepared_map)
    if answers and set(answer_map) != expected:
        raise ValueError(f"answer IDs differ under {root}")
    if usage and set(usage_map) != expected:
        raise ValueError(f"answer usage IDs differ under {root}")
    if set(retrieval_map) != expected:
        raise ValueError(f"retrieval IDs differ under {root}")
    if verdict_rows and set(verdicts) != expected:
        raise ValueError(f"judge IDs differ under {root}")
    clusters = {item: str(row.get("memory_id"))
                for item, row in prepared_map.items()}
    ids_by_group: dict[str, list[str]] = defaultdict(list)
    for item, row in retrieval_map.items():
        for group in group_names(row):
            ids_by_group[group].append(item)
    accuracy = {}
    if verdicts:
        for group, ids in sorted(ids_by_group.items()):
            correct = sum(verdicts[item] for item in ids)
            accuracy[group] = {"correct": correct, "questions": len(ids),
                               "accuracy": correct / max(1, len(ids))}
    combined_retrieval = {
        item: {
            **dict(row),
            **({"api_prompt_tokens": usage_map[item].get("api_prompt_tokens"),
                "api_total_tokens": usage_map[item].get("total_tokens")}
               if item in usage_map else {}),
        } for item, row in retrieval_map.items()}
    metrics = {group: retrieval_metrics([combined_retrieval[item] for item in ids])
               for group, ids in sorted(ids_by_group.items())}
    return {
        "accuracy": accuracy, "retrieval": metrics,
        "prepare_manifest": read(prepare_root / "prepare_manifest.json"),
        "answer_manifest": read(answer_root / "run_manifest.json"),
        "artifacts": {"prepare": str(prepare_root), "answer": str(answer_root)},
        "_verdicts": verdicts, "_groups": dict(ids_by_group),
        "_clusters": clusters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload: dict[str, Any] = {
        "schema_version": "graphmem-v5.54-index-structure-ablation-v1",
        "root": str(args.root), "protocol": {
            "questions": 2040, "factorial": "hierarchy x relation_traversal",
            "turn_budgets": list(BUDGETS), "evidence_tokens": 12000,
            "answer_model": "Qwen3-30B", "judge_model": "gpt-5.6-luna",
            "answer_policy": "v5_54", "candidate_answer_injection": False,
        }, "budgets": {},
        "prepare_audit": read(args.root / "prepare_audit.json"),
        "final_audit": read(args.root / "final_audit.json"),
    }
    internal: dict[int, dict[str, dict[str, Any]]] = {}
    for budget in BUDGETS:
        internal[budget] = {}
        payload["budgets"][str(budget)] = {"arms": {}, "comparisons": {}}
        for arm in ARMS:
            result = summarize_arm(args.root / f"turn{budget}" / arm)
            internal[budget][arm] = result
            payload["budgets"][str(budget)]["arms"][arm] = {
                key: value for key, value in result.items()
                if not key.startswith("_")}
        reference = internal[budget]["full"]
        clusters = reference["_clusters"]
        all_groups = sorted(reference["_groups"])
        comparisons = {}
        for left, right in (
                ("seed_only", "hierarchy_only"),
                ("seed_only", "flat_graph"),
                ("hierarchy_only", "full"),
                ("flat_graph", "full"),
                ("seed_only", "full")):
            comparisons[f"{left}->{right}"] = {
                group: paired(
                    internal[budget][left]["_verdicts"],
                    internal[budget][right]["_verdicts"],
                    reference["_groups"][group], clusters)
                for group in all_groups
            } if reference["_verdicts"] else {}
        interaction = {}
        if reference["_verdicts"]:
            for group in all_groups:
                ids = reference["_groups"][group]
                values = {
                    item: (int(internal[budget]["full"]["_verdicts"][item])
                           - int(internal[budget]["hierarchy_only"]["_verdicts"][item])
                           - int(internal[budget]["flat_graph"]["_verdicts"][item])
                           + int(internal[budget]["seed_only"]["_verdicts"][item]))
                    for item in ids}
                interaction[group] = {
                    "questions": len(ids),
                    "difference_in_differences": sum(values.values()) / max(1, len(ids)),
                    "memory_cluster_bootstrap_95ci": list(
                        cluster_bootstrap(values, clusters)),
                }
        payload["budgets"][str(budget)]["comparisons"] = comparisons
        payload["budgets"][str(budget)]["interaction"] = interaction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        str(budget): {arm: payload["budgets"][str(budget)]["arms"][arm]
                      ["accuracy"].get("overall") for arm in ARMS}
        for budget in BUDGETS}, indent=2))


if __name__ == "__main__":
    main()
