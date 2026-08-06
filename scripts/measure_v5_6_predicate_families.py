#!/usr/bin/env python3
"""Sweep predicate-clustering settings and measure what they buy.

Extraction emits ~433 distinct predicates per memory for ~498 facts, so the
collection chain key is nearly a primary key and 96% of collections are
singletons: there is no aggregation unit.  ``PredicateCanonicalizer`` is meant
to fix that but is inert by construction -- it only merges *mutual* nearest
neighbours, only inside an (owner, scope, value_type, polarity) slot (which
leaves 51% of predicates ineligible), and only above 0.92 similarity.

Rebuilding a graph does **not** re-extract: the semantic cache key covers only
``semantic_*`` fields, so changing ``edges.*`` reuses every extraction and every
embedding.  A sweep therefore costs no generation tokens.

Reported per arm: predicate families, collection member distribution, and edge
counts by relation -- the last because a coarser predicate vocabulary should
also make ``shared_entity`` / ``collection_co_member`` / ``state_next`` denser.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller  # noqa: E402
from graphmem.build.canonicalize import PredicateCanonicalizer  # noqa: E402
from graphmem.config import EdgeConfig, load_config  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.projection import ProjectionConfig, build_manifests  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402

ARMS: dict[str, dict] = {
    "E0_frozen":        {"predicate_cluster_scope": "slot",  "predicate_cluster_mode": "mutual_pair",   "predicate_embedding_threshold": 0.92},
    "E1_thresh_80":     {"predicate_cluster_scope": "slot",  "predicate_cluster_mode": "mutual_pair",   "predicate_embedding_threshold": 0.80},
    "E2_owner_slot":    {"predicate_cluster_scope": "owner", "predicate_cluster_mode": "mutual_pair",   "predicate_embedding_threshold": 0.92},
    "E3_agglomerative": {"predicate_cluster_scope": "slot",  "predicate_cluster_mode": "agglomerative", "predicate_embedding_threshold": 0.92},
    "E4_owner_agg_85":  {"predicate_cluster_scope": "owner", "predicate_cluster_mode": "agglomerative", "predicate_embedding_threshold": 0.85},
    "E5_owner_agg_80":  {"predicate_cluster_scope": "owner", "predicate_cluster_mode": "agglomerative", "predicate_embedding_threshold": 0.80},
    "E6_owner_agg_70":  {"predicate_cluster_scope": "owner", "predicate_cluster_mode": "agglomerative", "predicate_embedding_threshold": 0.70},
}


def measure(store: SQLiteGraphStore, memory_ids) -> dict:
    predicates_per_memory, sizes, relations = [], [], collections.Counter()
    facts_total = 0
    for memory_id in memory_ids:
        nodes = list(store.nodes(memory_id))
        facts = [node for node in nodes if node.node_type == NodeType.CANONICAL_FACT]
        facts_total += len(facts)
        predicates_per_memory.append(len({str(node.attributes.get("predicate", "")) for node in facts}))
        for edge in store.edges(memory_id):
            relations[str(edge.relation)] += 1
        _nodes, _edges, rows = build_manifests(
            memory_id, nodes, ProjectionConfig(collection_manifest=True))
        sizes.extend(row.member_count for row in rows)
    sizes.sort()
    return {
        "facts": facts_total,
        "predicates_per_memory": statistics.mean(predicates_per_memory) if predicates_per_memory else 0,
        "collections": len(sizes),
        "members_mean": statistics.mean(sizes) if sizes else 0,
        "members_p95": sizes[max(0, int(0.95 * len(sizes)) - 1)] if sizes else 0,
        "members_max": max(sizes) if sizes else 0,
        "singleton_rate": (sum(1 for s in sizes if s == 1) / len(sizes)) if sizes else 0,
        "collections_ge3": sum(1 for s in sizes if s >= 3),
        "edges_total": sum(relations.values()),
        "edges_by_relation": dict(sorted(relations.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True,
                        help="a graph database whose llm_cache still holds the extraction")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--memories", type=int, default=6)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()

    base = load_config(args.config)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = args.output_root / f"predicate_families_{stamp}"
    root.mkdir(parents=True)

    probe = SQLiteGraphStore(args.source_db, read_only=True)
    memory_ids = [row[0] for row in probe._read(
        "SELECT memory_id FROM conversations ORDER BY memory_id")][:args.memories]
    probe.close()
    results = {}
    for name in (item.strip() for item in args.arms.split(",") if item.strip()):
        overrides = ARMS[name]
        target = root / f"{name}.sqlite"
        shutil.copy2(args.source_db, target)
        config = replace(base, edges=replace(base.edges, **overrides))
        store = SQLiteGraphStore(target)
        started = time.perf_counter()
        for memory_id in memory_ids:
            distiller = QwenSemanticDistiller(store, config, "v5.6-predicate")
            GraphBuildPipeline(
                store, dataset_hash="v5.6-predicate", distiller=distiller,
                predicate_canonicalizer=PredicateCanonicalizer(store, config),
            ).build(memory_id, config)
        stats = measure(store, memory_ids)
        stats["seconds"] = round(time.perf_counter() - started, 1)
        stats["overrides"] = overrides
        # A rebuild that re-extracted would invalidate the whole comparison.
        stats["uncached_generation_calls"] = int(store._read_one(
            "SELECT count(*) FROM llm_calls WHERE cached=0 AND stage LIKE 'scene_semantic%'")[0])
        results[name] = stats
        print(f"{name}: predicates/mem={stats['predicates_per_memory']:.0f} "
              f"singleton={stats['singleton_rate']:.3f} members_mean={stats['members_mean']:.2f} "
              f"edges={stats['edges_total']:,} ({stats['seconds']}s)", flush=True)
        store.close()
        target.unlink(missing_ok=True)

    (root / "summary.json").write_text(json.dumps(
        {"memories": memory_ids, "arms": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in row.items() if k != "edges_by_relation"}
                      for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
