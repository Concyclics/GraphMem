#!/usr/bin/env python3
"""Does an LLM-compiled predicate vocabulary form real families -- safely?

Runs alongside the free embedding-clustering sweep.  Reports both halves of the
trade, because either alone is misleading:

* **did families form** -- distinct predicates, singleton rate, members per
  collection.  Without this the pass is pointless.
* **did it over-merge** -- families rejected for straddling modality or
  polarity, largest family, and a sample of accepted merges to read by eye.
  Collapsing "visited Kyoto" into "wants to visit Kyoto" would raise aggregation
  accuracy while corrupting every modality-sensitive answer, so a gain reported
  without this number cannot be trusted.

Reads predicates from an existing graph; writes nothing to it.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build.vocabulary import PredicateVocabulary, _axis  # noqa: E402
from graphmem.config import load_config  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-db", type=Path, required=True,
                        help="writable sidecar for the vocabulary call ledger")
    parser.add_argument("--memories", type=int, default=6)
    parser.add_argument("--max-family-size", type=int, default=24)
    args = parser.parse_args()

    config = load_config(args.config)
    store = SQLiteGraphStore(args.source_db, read_only=True)
    cache = SQLiteGraphStore(args.cache_db)
    vocabulary = PredicateVocabulary(cache, config, "v5.6-vocabulary",
                                     max_family_size=args.max_family_size)

    memory_ids = [row[0] for row in store._read(
        "SELECT memory_id FROM conversations ORDER BY memory_id")][:args.memories]
    rows = []
    for memory_id in memory_ids:
        facts = [node for node in store.nodes(memory_id)
                 if node.node_type == NodeType.CANONICAL_FACT]
        chains = collections.Counter()
        raw_predicates = []
        for node in facts:
            predicate = str(node.attributes.get("predicate", ""))
            raw_predicates.append(predicate)
            chains[(str(node.attributes.get("owner_id", "")), predicate,
                    str(node.attributes.get("scope", "")),
                    str(node.attributes.get("collection_key", "")))] += 1
        started = time.perf_counter()
        result = vocabulary.compile(memory_id, raw_predicates)
        seconds = time.perf_counter() - started

        after = collections.Counter()
        for node in facts:
            predicate = result.mapping.get(str(node.attributes.get("predicate", "")),
                                           str(node.attributes.get("predicate", "")))
            after[(str(node.attributes.get("owner_id", "")), predicate,
                   str(node.attributes.get("scope", "")),
                   str(node.attributes.get("collection_key", "")))] += 1

        def shape(counter):
            sizes = sorted(counter.values())
            return {"collections": len(sizes), "mean": statistics.mean(sizes),
                    "singleton_rate": sum(1 for s in sizes if s == 1) / len(sizes),
                    "ge3": sum(1 for s in sizes if s >= 3), "max": max(sizes)}

        merges = collections.defaultdict(list)
        for label, canonical in result.mapping.items():
            if label != canonical:
                merges[canonical].append(label)
        # Any accepted merge whose members disagree on modality/polarity is a
        # containment failure, not a stylistic one.
        unsafe = [f"{canonical}: {members}" for canonical, members in merges.items()
                  if len({_axis(item) for item in members + [canonical]}) > 1]
        rows.append({
            "memory_id": memory_id, "facts": len(facts),
            "predicates_before": len(set(raw_predicates)),
            "predicates_after": result.families,
            "merged_labels": result.merged_labels,
            "largest_family": result.largest_family,
            "rejected_families": list(result.rejected_families)[:8],
            "rejected_count": len(result.rejected_families),
            "unsafe_accepted": unsafe[:5], "unsafe_count": len(unsafe),
            "collections_before": shape(chains), "collections_after": shape(after),
            "tokens": result.tokens, "cached": result.cached, "seconds": round(seconds, 1),
            "finish_reason": result.finish_reason,
            "sample_merges": {k: v for k, v in list(merges.items())[:6]},
        })
        print(f"{memory_id}: predicates {rows[-1]['predicates_before']} -> "
              f"{rows[-1]['predicates_after']}, singleton "
              f"{rows[-1]['collections_before']['singleton_rate']:.3f} -> "
              f"{rows[-1]['collections_after']['singleton_rate']:.3f}, "
              f"rejected={rows[-1]['rejected_count']} unsafe={rows[-1]['unsafe_count']} "
              f"tokens={result.tokens:,} finish={result.finish_reason}", flush=True)

    summary = {
        "memories": len(rows),
        "tokens_per_memory": statistics.mean(row["tokens"] for row in rows),
        "predicates_before": statistics.mean(row["predicates_before"] for row in rows),
        "predicates_after": statistics.mean(row["predicates_after"] for row in rows),
        "singleton_before": statistics.mean(row["collections_before"]["singleton_rate"] for row in rows),
        "singleton_after": statistics.mean(row["collections_after"]["singleton_rate"] for row in rows),
        "members_mean_before": statistics.mean(row["collections_before"]["mean"] for row in rows),
        "members_mean_after": statistics.mean(row["collections_after"]["mean"] for row in rows),
        "collections_ge3_before": sum(row["collections_before"]["ge3"] for row in rows),
        "collections_ge3_after": sum(row["collections_after"]["ge3"] for row in rows),
        "largest_family": max(row["largest_family"] for row in rows),
        "rejected_total": sum(row["rejected_count"] for row in rows),
        "unsafe_accepted_total": sum(row["unsafe_count"] for row in rows),
        # A truncated answer parses as "no families" and is indistinguishable
        # from "nothing should merge", so it must be counted, not averaged in.
        "truncated_memories": sum(1 for row in rows if row["finish_reason"] == "length"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, indent=2))
    store.close(); cache.close()


if __name__ == "__main__":
    main()
