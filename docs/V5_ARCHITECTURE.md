# GraphMem 5.0 architecture boundary

Gate A introduces contracts and auditability only. It deliberately leaves the
V4/V4.1 implementation and every frozen result unchanged. Gate B may implement
the stores, projector, read views, builder, and ablation runner behind these
contracts after review.

## Dependency direction

```text
graphmem.eval  (offline only; owns answer/gold labels)
      X
      X import forbidden
      X
domain <- config <- interfaces <- build / retrieval / runtime
   ^                                |
   +----------- legacy adapter -----+
```

Online build/query modules may import `graphmem.domain`, `graphmem.config`, and
`graphmem.interfaces`. They must never import `graphmem.eval`, benchmark answers,
gold sessions, gold turns, or question-type labels.

## Stable identity and cache invariant

Canonical JSON sorts mapping keys, uses compact separators, and encodes strings
as ASCII escapes before SHA-256. Domain IDs retain 128 bits and a type namespace.
Configuration hashes retain the full SHA-256 digest. A cache key is valid only
when it includes dataset, model, prompt, schema, and configuration hashes.

## Data placement invariant

- `SourceTurn` is the only online domain object allowed to contain raw text.
- Evidence groups reference immutable source spans.
- Graph nodes and edges carry compact summaries/typed properties plus an
  `evidence_group_id`.
- Gate B Neo4j projection must not contain raw conversation text, model requests,
  model responses, or embeddings. SQLite remains authoritative.

## Runtime invariant

`neo4j_direct`, `neo4j_cached`, and `sqlite_snapshot` must implement the same
`GraphRuntime` contract. A navigation proof is an ordered list of typed edges and
evidence-group references; every proof path must resolve back to source turns.

## Evaluation invariant

Gate A freezes V4.1 retrieval output, per-question fingerprints, token accounting,
latency, and graph statistics on exactly 200 unique questions. Gate B changes only
graph construction while holding navigator, seeds, evidence packing, budgets,
embedding model, and random seed fixed.
