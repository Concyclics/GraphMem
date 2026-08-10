#!/usr/bin/env python3
"""Summarize the V5.20 graph-structure and evidence-layout ablation."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ("seed_only", "flat_graph", "hierarchical", "topology_layout")
COMPARISONS = (
    ("seed_only", "flat_graph"),
    ("flat_graph", "hierarchical"),
    ("hierarchical", "topology_layout"),
    ("seed_only", "topology_layout"),
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def stratum(row: dict) -> str:
    benchmark = str(row.get("benchmark"))
    raw = str(row.get("stratum") or row.get("question_type") or "")
    if benchmark == "longmemeval":
        return "lme_temporal" if "temporal" in raw else "lme_multi_session"
    return ("locomo_temporal" if "temporal" in raw or raw.endswith("cat2")
            else "locomo_multihop")


def group_names(label: str) -> tuple[str, ...]:
    return (label, "temporal" if "temporal" in label else "structural", "overall")


def paired(left: dict[str, bool], right: dict[str, bool]) -> dict:
    ids = sorted(set(left) & set(right))
    gains = sum(not left[item] and right[item] for item in ids)
    losses = sum(left[item] and not right[item] for item in ids)
    discordant = gains + losses
    tail = min(gains, losses)
    p_value = (min(1.0, 2.0 * sum(math.comb(discordant, k)
                                  for k in range(tail + 1)) / (2 ** discordant))
               if discordant else 1.0)
    deltas = [int(right[item]) - int(left[item]) for item in ids]
    rng = random.Random(42)
    boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas)
                  for _ in range(10_000)) if deltas else [0.0]
    return {
        "questions": len(ids), "gains": gains, "losses": losses,
        "delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "mcnemar_exact_p": p_value,
        "paired_bootstrap_95ci": [boot[249], boot[9749]],
    }


def mean_metrics(data: list[dict]) -> dict:
    keys = (
        "turn_recall", "turn_precision", "turn_f1", "turn_all_hit",
        "candidate_turn_recall", "candidate_turn_precision",
        "candidate_average_precision", "candidate_ndcg_at_8",
        "graph_reachable_turn_recall", "graph_reachable_turn_precision",
        "visited_nodes", "visited_edges", "latency_total_ms", "packed_turns",
        "answer_total_tokens", "evidence_chain_count", "evidence_chain_turns",
        "evidence_graph_group_count", "evidence_graph_turns",
        "evidence_auxiliary_turns",
    )
    payload = {}
    for key in keys:
        values = [float(row[key]) for row in data if row.get(key) is not None]
        payload[key] = sum(values) / len(values) if values else None
    signals = Counter()
    for row in data:
        signals.update({str(key): int(value) for key, value in
                        dict(row.get("traversed_relation_signals", {})).items()})
    payload["traversed_relation_signals"] = dict(sorted(signals.items()))
    return payload


def summarize_arm(root: Path) -> tuple[dict, dict[str, bool], dict[str, dict[str, bool]]]:
    answer_root = root / "answer"
    answers = rows(answer_root / "answers.jsonl")
    answer_by_id = {str(row["question_id"]): row for row in answers}
    evaluations = (rows(answer_root / "judge_lme" / "auto_eval.jsonl")
                   + rows(answer_root / "judge_locomo" / "auto_eval.jsonl"))
    verdicts: dict[str, bool] = {}
    verdicts_by_group: dict[str, dict[str, bool]] = defaultdict(dict)
    accuracy_buckets: dict[str, list[bool]] = defaultdict(list)
    for row in evaluations:
        question_id = str(row["question_id"])
        if question_id not in answer_by_id:
            continue
        correct = bool(row["correct"])
        verdicts[question_id] = correct
        label = stratum(answer_by_id[question_id])
        for group in group_names(label):
            verdicts_by_group[group][question_id] = correct
            accuracy_buckets[group].append(correct)
    retrieval = rows(answer_root / "retrieval.jsonl")
    retrieval_buckets: dict[str, list[dict]] = defaultdict(list)
    for row in retrieval:
        for group in group_names(stratum(row)):
            retrieval_buckets[group].append(row)
    manifest = read(answer_root / "run_manifest.json")
    payload = {
        "accuracy": {group: {
            "correct": sum(values), "total": len(values),
            "accuracy": sum(values) / len(values) if values else None,
        } for group, values in sorted(accuracy_buckets.items())},
        "retrieval": {group: mean_metrics(values)
                      for group, values in sorted(retrieval_buckets.items())},
        "manifest": manifest,
        "artifacts": {"root": str(root), "answer": str(answer_root)},
    }
    return payload, verdicts, dict(verdicts_by_group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "graphmem-v5.20-graph-structure-ablation-v1",
        "root": str(args.root), "arms": {}, "comparisons": {},
        "protocol": {
            "questions": 200, "evidence_turns": 64,
            "evidence_tokens": 12000, "answer_output_tokens": 2000,
            "candidate_answer_injection": False,
            "answer_model": "Qwen3-30B", "judge_model": "gpt-5.6-luna",
        },
    }
    verdicts = {}; grouped = {}
    for arm in ARMS:
        arm_payload, verdicts[arm], grouped[arm] = summarize_arm(args.root / arm)
        payload["arms"][arm] = arm_payload
    all_groups = ("lme_multi_session", "lme_temporal", "locomo_multihop",
                  "locomo_temporal", "structural", "temporal", "overall")
    for left, right in COMPARISONS:
        key = f"{left}->{right}"
        payload["comparisons"][key] = {
            "overall": paired(verdicts[left], verdicts[right]),
            "by_group": {
                group: paired(grouped[left].get(group, {}),
                              grouped[right].get(group, {}))
                for group in all_groups
            },
        }
    expected = {
        "seed_only": (False, False, "adaptive"),
        "flat_graph": (True, False, "adaptive"),
        "hierarchical": (True, True, "adaptive"),
        "topology_layout": (True, True, "topological"),
    }
    audit = {}
    for arm, (traversal, hierarchy, order) in expected.items():
        manifest = payload["arms"][arm]["manifest"]
        audit[arm] = {
            "questions_200": manifest.get("questions") == 200,
            "turn_budget_64": manifest.get("budget", {}).get(
                "max_evidence_turns") == 64,
            "graph_traversal": manifest.get("graph_traversal") == traversal,
            "hierarchical_routing": manifest.get("hierarchical_routing") == hierarchy,
            "evidence_order": manifest.get("evidence_order") == order,
            "candidate_answer_injection_off": not manifest.get(
                "candidate_answer_injection", False),
            "luna_verdicts_200": len(verdicts[arm]) == 200,
        }
    payload["audit"] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({arm: payload["arms"][arm]["accuracy"].get("overall")
                      for arm in ARMS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
