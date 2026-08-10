#!/usr/bin/env python3
"""Summarize the six-arm V5.19 relation-signal ablation."""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ("full", "no_scene", "no_entity_family", "no_temporal",
        "no_lexical", "semantic_only")


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def stratum(answer: dict) -> str:
    benchmark = str(answer.get("benchmark"))
    raw = str(answer.get("stratum") or answer.get("question_type") or "")
    if benchmark == "longmemeval":
        return "lme_temporal" if "temporal" in raw else "lme_multi_session"
    return "locomo_temporal" if ("temporal" in raw or raw.endswith("cat2")) \
        else "locomo_multihop"


def groups(label: str) -> tuple[str, ...]:
    return (label,
            "temporal" if "temporal" in label else "structural",
            "overall")


def accuracy_payload(answers: list[dict], evaluations: list[dict]) -> dict:
    answer_by_id = {str(row["question_id"]): row for row in answers}
    buckets: dict[str, list[bool]] = defaultdict(list)
    verdicts = {}
    verdicts_by_group: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in evaluations:
        question_id = str(row["question_id"])
        if question_id not in answer_by_id:
            continue
        correct = bool(row["correct"])
        verdicts[question_id] = correct
        label = stratum(answer_by_id[question_id])
        for group in groups(label):
            buckets[group].append(correct)
            verdicts_by_group[group][question_id] = correct
    return {
        "by_group": {key: {"correct": sum(values), "total": len(values),
                           "accuracy": sum(values) / len(values) if values else None}
                     for key, values in sorted(buckets.items())},
        "verdicts": verdicts,
        "verdicts_by_group": dict(verdicts_by_group),
    }


def paired(full: dict[str, bool], arm: dict[str, bool]) -> dict:
    ids = sorted(set(full) & set(arm))
    new_only = sum(not full[item] and arm[item] for item in ids)
    full_only = sum(full[item] and not arm[item] for item in ids)
    discordant = new_only + full_only
    tail = min(new_only, full_only)
    p_value = (min(1.0, 2.0 * sum(math.comb(discordant, k)
                                  for k in range(tail + 1)) / (2 ** discordant))
               if discordant else 1.0)
    deltas = [int(arm[item]) - int(full[item]) for item in ids]
    rng = random.Random(42)
    boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas)
                  for _ in range(10_000)) if deltas else [0.0]
    return {
        "questions": len(ids), "new_only": new_only, "full_only": full_only,
        "delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "mcnemar_exact_p": p_value,
        "paired_bootstrap_95ci": [boot[249], boot[9749]],
    }


def mean_metrics(retrieval: list[dict]) -> dict:
    keys = ("turn_recall", "turn_precision", "turn_f1", "turn_all_hit",
            "candidate_turn_recall", "candidate_turn_precision",
            "visited_nodes", "visited_edges", "latency_total_ms",
            "packed_turns", "answer_total_tokens")
    result = {}
    for key in keys:
        values = [float(row[key]) for row in retrieval if row.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    traversed = Counter()
    for row in retrieval:
        traversed.update({str(key): int(value) for key, value in
                          dict(row.get("traversed_relation_signals", {})).items()})
    result["traversed_relation_signals"] = dict(sorted(traversed.items()))
    return result


def grouped_retrieval_metrics(retrieval: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in retrieval:
        label = stratum(row)
        for group in groups(label):
            buckets[group].append(row)
    return {group: mean_metrics(rows) for group, rows in sorted(buckets.items())}


def graph_signals(path: Path) -> dict:
    counts = Counter(); multi = edges = 0
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        for (source,) in connection.execute(
                "SELECT source FROM graph_edges WHERE source LIKE '%relation_mask:%'"):
            mask = str(source).split("relation_mask:", 1)[1].split("|", 1)[0]
            signals = tuple(filter(None, mask.split(",")))
            edges += 1
            multi += len(signals) > 1
            counts.update(signals)
    return {"relation_mask_edges": edges, "multi_attribute_edges": multi,
            "edge_signal_counts": dict(sorted(counts.items()))}


def build_candidates(report: dict) -> dict:
    masks = Counter(); sources = Counter(); atomic_signals = Counter()
    scalar_keys = (
        "coarse_candidate_pairs", "gated_child_pairs",
        "atomic_relation_candidates_generated",
        "atomic_relation_pairs_proposed", "relation_mask_pairs",
        "accepted_pairs",
    )
    totals = Counter()
    enabled = set()
    rows = list(report.get("rows", ()))
    for row in rows:
        diagnostic = dict(row.get("relation_candidate_diagnostics", {}))
        enabled.update(map(str, diagnostic.get("enabled_relation_signals", ())))
        for key in scalar_keys:
            totals[key] += int(diagnostic.get(key, 0))
        masks.update({str(key): int(value) for key, value in
                      dict(diagnostic.get("relation_mask_counts", {})).items()})
        sources.update({str(key): int(value) for key, value in
                        dict(diagnostic.get(
                            "atomic_candidate_source_counts", {})).items()})
        atomic_signals.update({str(key): int(value) for key, value in
                               dict(diagnostic.get(
                                   "atomic_candidate_signal_counts", {})).items()})
    return {
        "memories": len(rows), "enabled_relation_signals": sorted(enabled),
        **dict(totals),
        "relation_mask_counts": dict(sorted(masks.items())),
        "atomic_candidate_source_counts": dict(sorted(sources.items())),
        "atomic_candidate_signal_counts": dict(sorted(atomic_signals.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"schema_version": "graphmem-v5.19-attribute-ablation-v1",
               "root": str(args.root), "arms": {}}
    verdicts = {}
    verdicts_by_group = {}
    for arm in ARMS:
        root = args.root / arm
        answers = rows(root / "answer" / "answers.jsonl")
        evaluations = (rows(root / "answer" / "judge_lme" / "auto_eval.jsonl")
                       + rows(root / "answer" / "judge_locomo" / "auto_eval.jsonl"))
        accuracy = accuracy_payload(answers, evaluations)
        verdicts[arm] = accuracy.pop("verdicts")
        verdicts_by_group[arm] = accuracy.pop("verdicts_by_group")
        graph_db = root / "graph" / "graphmem.sqlite"
        retrieval_rows = rows(root / "answer" / "retrieval.jsonl")
        build_report = read(root / "build_report.json")
        payload["arms"][arm] = {
            "accuracy": accuracy,
            "retrieval": mean_metrics(retrieval_rows),
            "retrieval_by_group": grouped_retrieval_metrics(retrieval_rows),
            "graph": graph_signals(graph_db) if graph_db.exists() else None,
            "build": build_report.get("summary"),
            "candidate_generation": build_candidates(build_report),
            "answer_manifest": read(root / "answer" / "run_manifest.json"),
            "artifacts": {"root": str(root), "graph_db": str(graph_db)},
        }
    full = verdicts.get("full", {})
    for arm in ARMS:
        payload["arms"][arm]["paired_vs_full"] = paired(
            full, verdicts.get(arm, {}))
        payload["arms"][arm]["paired_vs_full_by_group"] = {
            group: paired(
                verdicts_by_group.get("full", {}).get(group, {}),
                verdicts_by_group.get(arm, {}).get(group, {}))
            for group in (*(
                "lme_multi_session", "lme_temporal", "locomo_multihop",
                "locomo_temporal", "structural", "temporal"), "overall")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({arm: payload["arms"][arm]["accuracy"]
                      for arm in ARMS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
