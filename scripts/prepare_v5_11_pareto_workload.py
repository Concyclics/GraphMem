#!/usr/bin/env python3
"""Freeze the exact mixed LongMemEval/LoCoMo retrieval workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / (
        "artifacts/development_sets/"
        "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/v5_10/full_benchmark_20260809/graph/report_graph.sqlite")
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    memory_ids = sorted({row.memory_id for row in questions})
    digest = hashlib.sha256()
    turn_counts: dict[str, int] = {}
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        for memory_id in memory_ids:
            rows = con.execute(
                """SELECT turn_id, session_id, turn_index, role, speaker,
                          COALESCE(timestamp, ''), raw_text
                   FROM source_turns WHERE memory_id=?
                   ORDER BY session_id, turn_index, turn_id""",
                (memory_id,),
            ).fetchall()
            if not rows:
                raise SystemExit(f"source DB has no turns for {memory_id}")
            turn_counts[memory_id] = len(rows)
            for row in rows:
                digest.update(memory_id.encode("utf-8"))
                for value in row:
                    digest.update(b"\0")
                    digest.update(str(value).encode("utf-8"))
                digest.update(b"\n")
    finally:
        con.close()
    payload = {
        "schema_version": "graphmem-mem0-pareto-workload-v1",
        "protocol": {
            "plane": "retrieval data plane",
            "query_mode": "closed-loop; one outstanding request per logical user",
            "top_k": 32,
            "zipf_alpha": 1.1,
            "clients": [1, 4, 16, 64, 128, 256],
            "workers": [1, 4, 8],
            "affinity_replicas": "min(2, workers)",
        },
        "source": {
            "db": str(args.db.resolve()),
            "db_size_bytes": args.db.stat().st_size,
            "turn_payload_sha256": digest.hexdigest(),
            "lme": str(args.lme.resolve()),
            "locomo": str(args.locomo.resolve()),
            "gold": str(args.gold.resolve()),
        },
        "memory_ids": memory_ids,
        "memory_count": len(memory_ids),
        "turn_count": sum(turn_counts.values()),
        "turn_counts": turn_counts,
        "queries": [{
            "question_id": row.question_id,
            "memory_id": row.memory_id,
            "benchmark": row.benchmark,
            "stratum": row.stratum,
            "query": row.query,
        } for row in questions],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "queries": len(questions),
        "memories": len(memory_ids),
        "turns": sum(turn_counts.values()),
        "turn_payload_sha256": digest.hexdigest(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
