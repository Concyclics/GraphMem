#!/usr/bin/env python3
"""Audit full-corpus predicate-family state relation materialization."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.storage import SQLiteGraphStore  # noqa: E402


EXPECTED_SIGNALS = frozenset({
    "scene_similar", "shared_entity", "state_compatible"})


def metadata(source: str) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    signals: tuple[str, ...] = ()
    witnesses: dict[str, tuple[str, ...]] = {}
    for component in source.split("|"):
        if component.startswith("relation_mask:"):
            signals = tuple(filter(None, component.split(":", 1)[1].split(",")))
        elif component.startswith("relation_witness:"):
            try:
                row = json.loads(component.split(":", 1)[1])
            except (TypeError, ValueError):
                continue
            if isinstance(row, dict):
                witnesses = {
                    str(key): tuple(map(str, values))
                    for key, values in row.items() if isinstance(values, list)}
    return signals, witnesses


def snapshot(store: SQLiteGraphStore) -> dict:
    totals: Counter[str] = Counter()
    signal_edges: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    polarity_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    state_degree: Counter[tuple[str, str]] = Counter()
    rows = []
    for memory_id in store.memory_ids():
        local: Counter[str] = Counter()
        seen: set[tuple[str, str, str]] = set()
        for edge in store.edges(memory_id):
            local["edges"] += 1
            endpoint = tuple(sorted((edge.src_id, edge.dst_id))) + (
                str(edge.relation),)
            if endpoint in seen:
                local["duplicate_endpoint_relations"] += 1
            seen.add(endpoint)
            signals, witnesses = metadata(edge.source)
            if not signals:
                continue
            local["relation_mask_edges"] += 1
            if len(signals) > 1:
                local["multi_attribute_edges"] += 1
            leaked = set(signals) - EXPECTED_SIGNALS
            local["disabled_signal_leaks"] += len(leaked)
            for signal in signals:
                signal_edges[signal] += 1
            if "state_compatible" not in signals:
                continue
            local["state_edges"] += 1
            state_degree[(memory_id, edge.src_id)] += 1
            state_degree[(memory_id, edge.dst_id)] += 1
            state_witnesses = witnesses.get("state_compatible", ())
            if not state_witnesses:
                local["state_edges_missing_witness"] += 1
            if len(state_witnesses) > 1:
                local["multi_witness_state_edges"] += 1
            for witness in state_witnesses:
                fields = witness.split("\x1f")
                if len(fields) == 4 and all(fields):
                    local["family_state_witnesses"] += 1
                    family_counts[fields[1]] += 1
                    polarity_counts[fields[2]] += 1
                    modality_counts[fields[3]] += 1
                else:
                    local["legacy_or_invalid_state_witnesses"] += 1
        totals.update(local)
        version, checksum = store.graph_identity(memory_id)
        rows.append({
            "memory_id": memory_id,
            "graph_version": version,
            "graph_checksum": checksum,
            **dict(local),
        })
    degree_values = sorted(state_degree.values())
    return {
        "memories": len(rows),
        "totals": dict(sorted(totals.items())),
        "relation_signal_edges": dict(sorted(signal_edges.items())),
        "predicate_family_witnesses_top100": family_counts.most_common(100),
        "polarity_witnesses": dict(sorted(polarity_counts.items())),
        "modality_witnesses": dict(sorted(modality_counts.items())),
        "state_endpoint_degree": {
            "count": len(degree_values),
            "mean": (sum(degree_values) / len(degree_values)
                     if degree_values else 0),
            "p95": (degree_values[max(0, (95 * len(degree_values) + 99) // 100 - 1)]
                    if degree_values else 0),
            "max": max(degree_values, default=0),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-db", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_store = SQLiteGraphStore(args.baseline_db, read_only=True)
    candidate_store = SQLiteGraphStore(args.candidate_db, read_only=True)
    try:
        baseline = snapshot(baseline_store)
        candidate = snapshot(candidate_store)
    finally:
        baseline_store.close()
        candidate_store.close()
    baseline_checksums = {
        row["memory_id"]: row["graph_checksum"] for row in baseline["rows"]}
    changed = sum(
        row["graph_checksum"] != baseline_checksums.get(row["memory_id"])
        for row in candidate["rows"])
    payload = {
        "schema_version": "graphmem-v5.33-predicate-family-state-audit-v1",
        "baseline_db": str(args.baseline_db),
        "candidate_db": str(args.candidate_db),
        "graph_checksums_changed": changed,
        "baseline": baseline,
        "candidate": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "graph_checksums_changed": changed,
        "baseline": {key: value for key, value in baseline.items() if key != "rows"},
        "candidate": {key: value for key, value in candidate.items() if key != "rows"},
    }, indent=2))


if __name__ == "__main__":
    main()
