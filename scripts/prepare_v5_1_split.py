#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from graphmem.eval import conversation_holdout_split, load_dev_questions, load_gold_turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lme", type=Path, required=True); parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = conversation_holdout_split(load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold)))
    payload = {"protocol": "graphmem-v5.1-conversation-holdout", "seed": 42,
               "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (args.lme, args.locomo, args.gold)},
               "splits": {name: [{"question_id": row.question_id, "memory_id": row.memory_id,
                                    "benchmark": row.benchmark, "stratum": row.stratum} for row in rows]
                          for name, rows in splits.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__": main()
