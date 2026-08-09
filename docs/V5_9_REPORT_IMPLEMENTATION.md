# V5.9 report implementation and experiment contract

This iteration keeps every frozen V5.8 B0--B5 profile unchanged. The report arm
is opt-in through `configs/v5/v5_9_report.json`.

## Implemented method path

- Recursive bounded semantic coarsening creates an arbitrary-depth routing tree.
  Routing provenance stores child references rather than copying every terminal
  EvidenceGroup into every ancestor.
- Parent-gated relation construction seeds bounded sibling candidates at every
  parent and expands a child cross-product only below a surviving parent pair.
  Candidate comparisons, accepted relations, refine candidates and hierarchy
  depth are persisted in the build manifest.
- H10 compiles QueryIR into a directional root-to-leaf physical route. The
  operator-aware arm widens set/closure queries while lookup queries retain a
  narrow beam. Global flattened routing candidates are disabled on this path;
  the adaptive plan may open at most four sparse Session portals and records
  their root-to-leaf ancestor corridor.
- The AST algebra is recursive rather than root-switched. Count, set, temporal,
  ordinal and nested operators preserve selected witnesses. A post-pack
  certificate rechecks bindings, collection/scope closure and packed evidence.
- Immutable read views precompile lexical/typed postings. The runtime uses a
  byte-bounded weighted LRU, single-flight cold compilation and an atomic SQLite
  read transaction for `(version, checksum, nodes, edges)`. Source turns,
  evidence-group maps and token costs are compiled/cached once per visible
  graph version rather than reloaded by every query.
- The SQLite authority has an opt-in WAL read pool with starvation fallback and
  an optimistic `expected_version` delta API. Affected-path publication
  recompiles the changed Session Card and route ancestors bottom-up, then
  commits their row delta in one transaction. Its composable graph checksum and
  local invariant checks are O(delta), avoiding a full graph deserialize/hash.
- `ProcessShardedNavigator` provides a persistent multi-process query plane.
  Each worker owns one query-only WAL connection and one versioned immutable
  snapshot cache, so CPU-bound Python ranking/packing scales beyond the GIL.

## Exact scope of incremental updates

The synchronous path currently covers updates to an existing partition: caller
supplies changed terminal/branch rows, and GraphMem recompiles/publishes its
routing ancestors. Unchanged rows retain their IDs and bytes. New-session
partition insertion, partition split/merge and global relation re-clustering are
not silently approximated; they remain explicit asynchronous maintenance work.
The report must not describe this implementation as a complete incremental LLM
fact extractor.

## Environment

```bash
conda env create -f environment-report.yml
conda activate graphmem-v58-report
pytest -q
```

The working run used Python 3.11 in
`/ssd3/chenhan/Spark_MemGraph_Dev/.conda-envs/graphmem-v58`. Each result JSON
also records its effective config hash and hardware/measurement scope.

## Reproduce report artifacts

```bash
python scripts/measure_report_c1_scaling.py
python scripts/measure_report_c23.py \
  --db artifacts/report/v5_9/c23_graph/report_graph.sqlite --limit 100
python scripts/measure_report_system.py --workers 8
python scripts/summarize_v5_9_full_benchmark.py
python scripts/render_report_results.py
```

Outputs are written below `artifacts/report/v5_9/`. The renderer copies only
generated figures/macros into `../GraphMem_report`; raw observations remain in
the implementation repository.

## Interpretation rules

- C1 All-pairs candidate counts are exact. Wall time above 2K is explicitly
  marked `projected_from_loop`; it must never be cited as a directly executed
  20K materialization.
- C1 Token is a relation-decision envelope under the fixed
  `2*96 input + 256 output` contract. Shared semantic fact extraction is excluded
  for every arm.
- C1 path retention is measured on controlled semantic components, not a QA
  answer score.
- C2/C3 uses the same frozen V5.8 non-routing facts/edges and turn-level gold,
  but replaces parent Routing Cards with the report recursive hierarchy. No
  extractor or answer-model call occurs during this recoarsening. A run without
  `--embedding` isolates index/routing/algebra behavior and is labelled accordingly.
- System update results separate authority commit latency from cold construction
  of the first new in-memory read view. QPS, p95/p99, memory and errors are always
  reported together. Process-shard memory is the aggregate immutable snapshot
  cache, not total Python runtime RSS.
- The full end-to-end result recoarsens all 510 frozen V5.8 P8 memories and uses
  the same local Qwen-30B answer/judge model and exact tokenizer for all 2,040
  questions. It does not rerun semantic extraction and does not enable dense
  retrieval; those scope labels must accompany the reported accuracy.
