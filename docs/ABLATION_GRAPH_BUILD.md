# Graph build — tunable parameters and what they do to index quality

Every number here is measured on the frozen V5.4 corpus (110 memories, 54,766
canonical facts, 55,323 evidence groups) or on the 200-question development set,
unless it is explicitly marked as an estimate.

---

## 0. The single most important fact: the cache boundary

The extraction cache key is **not** `config_hash`. `semantic.py:_call` builds a
narrow sub-hash from a specific field list plus a hash of the request payload.
That splits every build knob into two cost classes, and getting this wrong is
the difference between a 3-minute ablation arm and a 229,000-token-per-memory
rebuild.

| class | knobs | cost per arm | why |
|---|---|---|---|
| **Free** — reuses extraction and embeddings | `edges.*`, `coarsen.*`, `storage.*`, `query_budget.*` | **0 generation tokens**, minutes | not in the cache key |
| **Full rebuild** | every `models.semantic_*`, `models.llm_model`, `schema_version` | **~229K tokens/memory** | in the cache key |
| **Full rebuild (indirect)** | `scenes.min_turns`, `scenes.max_turns`, `scenes.topic_similarity_threshold` | **~229K tokens/memory** | changes segmentation → changes scene ids and payload → changes the payload hash |

`scenes.*` is the trap: it looks like a segmentation knob, but it re-cuts every
scene, so the payload differs and nothing is reused.

**Practical consequence.** Design ablations to sweep the free class exhaustively
and the rebuild class only on a held-out subset. The frozen artifact
`v5_1_graph_ablation_full_20260805T103058Z` accumulated `max(graph_version)=42`
precisely because arms varied `semantic_max_facts_per_scene`, and each variation
re-extracted every scene — which is why naively summing its ledger reports
854K tokens/memory against a true single-build cost of 229K.

---

## 1. Pipeline stages and where quality is won or lost

```
turns ──segment──▶ scenes ──extract(LLM)──▶ facts ──canonicalize──▶ predicates
                                                        │
                                              projection│(deterministic)
                                                        ▼
                                          nodes + edges + collections
```

Measured distribution of what the frozen build actually produces:

| relation | edges | node type | nodes |
|---|---:|---|---:|
| `scene_contains` | 114,446 | `evidence_group_ref` | 55,323 |
| `has_fact` | 48,296 | `canonical_fact` | 54,766 |
| `participates_in` | 39,789 | `scene` | 20,955 |
| `refines_to` | 26,629 | `canonical_entity` | 16,592 |
| `collection_co_member` | 10,115 | `routing_card` | 5,786 |
| `at_time` | 7,209 | `event_skeleton` | 4,357 |
| `state_next` | 3,087 | `state_head` | 2,712 |
| **`temporal_before`** | **150** | `time_anchor` | 2,471 |
| — | — | `collection_scope` | 1,856 |

**8 of 31 relation types and 9 of 19 node types are ever built.** The graph is
structurally a tree (`scene_contains` + `refines_to` + `has_fact`) with a thin
lateral layer. `temporal_before` is 150 edges corpus-wide — 1.4 per memory.

---

## 2. Segmentation — `scenes.*`

| knob | default | frozen | effect | cost |
|---|---|---|---|---|
| `min_turns` | 2 | 2 | floor on scene size | rebuild |
| `max_turns` | 8 | **4** | ceiling; drives scenes/memory and therefore call count | rebuild |
| `topic_similarity_threshold` | 0.55 | 0.55 | where a scene is cut | rebuild |
| `max_events_per_scene` | 3 | 3 | event skeletons per scene | rebuild |
| `llm_semantic_extraction` | false | **true** | master switch for the LLM extractor | rebuild |
| `llm_hierarchy_compression` | false | **false** | LLM summarisation of routing cards | rebuild |

**Measured**: 194 scenes and 494 turns per LME memory at `max_turns=4`, i.e.
2.5 turns per scene. Scenes partition turns exactly — duplication is 1.00×, so
segmentation is *not* a source of token amplification.

`max_turns` is the main lever on cost: halving scenes halves calls and therefore
halves the measured 316-token fixed overhead per call. It also enlarges the
context each fact is extracted from, which cuts both ways and has not been
measured against accuracy.

`llm_hierarchy_compression=false` is already correct — the frozen config uses the
deterministic `_compile_routing` instead, and the hierarchy stages cost ~71,620
tokens/memory when enabled.

---

## 3. Extraction — `models.semantic_*`

All of these are **rebuild-class**.

| knob | frozen | effect on index quality | measured |
|---|---|---|---|
| `semantic_extraction_mode` | `strict_pair` | `legacy_batch` uses a free-form prompt and a repair loop; `strict_*` uses a JSON schema | legacy's repair path cost 89,648 tok/memory in the artifact |
| `semantic_max_facts_per_scene` | **4** | facts per scene → total facts → collection sizes | halving it is the budget ladder's degradation step; costs ~0 coverage |
| `semantic_batch_scenes` | 2 (forced by `strict_pair`) | scenes per call → amortises the **316-token** fixed overhead | 2→4 saves ~18,500 tok/memory |
| `semantic_batch_output_tokens` | 1024 | output ceiling; truncation causes missing scenes → retries | retries are ~11 calls/memory |
| `semantic_turn_input_chars` | 2500 | truncates long turns head/tail | not ablated |
| `semantic_quote_evidence` | true | emits the exact-quote field `q` | **26.1% of output bytes**; dropping it costs ~3pp of turn coverage |
| `semantic_max_tokens_per_memory` | 0 (off) | hard per-memory ceiling, enforced by `BuildTokenLedger` | on: 206,224 vs 229,038 mean, 0/4 over ceiling, coverage unchanged |
| `semantic_budget_degrade_at` | 0.75 | where the fact cap halves | **free**: switching degradation off recovers no coverage |

### The dominant quality problem lives here

Extraction emits **433 distinct predicates per memory for 498 facts** — a
near-unique predicate per fact, in the shape of sentences like *"plans a monthly
family game night"* rather than relations like *"plays"*.

Everything downstream inherits this:

| symptom | measured |
|---|---|
| collections are singletons | **96.1%** (49,477 of 51,496); mean 1.06 members, p95 = 1 |
| collections with ≥3 members | **724** corpus-wide |
| `collection_complete` never fires | 0 `COLLECTION_MANIFEST` nodes existed at all |
| lexical operand→collection matching plateaus | recall 38.2%, all-hit 19.3% |
| aggregation accuracy | **51.9%**, worst category, 47% of all errors |

Coarsening the *projection* key does not repair it — the loosest key tried,
`(owner, scope)`, still leaves **67.4%** singletons. The aggregation unit is
absent from the extracted data, not mis-keyed downstream.

---

## 4. Predicate canonicalization — `edges.predicate_*`

**Free class**: rebuilding the graph re-runs canonicalization against cached
extraction and cached embeddings, at zero generation cost.

| knob | frozen | effect |
|---|---|---|
| `predicate_embedding_threshold` | 0.92 | cosine floor for merging two predicates |
| `predicate_cluster_scope` | `slot` | which predicates may merge at all |
| `predicate_cluster_mode` | `mutual_pair` | how merges compose |

Three restrictions compound, and **the threshold is the least important**:

| restriction | measured consequence |
|---|---|
| slot = `(owner, scope, value_type, polarity)` | only **23.9%** of slots hold ≥2 predicates; **51% of predicates are ineligible before the threshold is read** |
| mutual-nearest **pairs** only | a family of five similar predicates yields at most two merges |
| threshold 0.92 | very strict |

Widening the slot to `(owner, polarity)` raises eligible predicates from
**49.0% to 78.6%** and mean labels per slot from 1.49 to 3.87. `agglomerative`
takes the transitive closure of every above-threshold pair instead of pairing
mutual nearest neighbours.

Arms swept (`scripts/measure_v5_6_predicate_families.py`, zero generation
tokens, asserted): `E0_frozen`, `E1_thresh_80`, `E2_owner_slot`,
`E3_agglomerative`, `E4_owner_agg_85`, `E5_owner_agg_80`, `E6_owner_agg_70`.

**Risk to watch.** Over-merging destroys precision: if *"visited Kyoto"* and
*"wants to visit Kyoto"* collapse, modality is lost and a plan becomes a fact.
Report merged-family size distribution alongside any accuracy gain, and treat a
`members_max` in the hundreds as a failure, not a success.

---

## 5. Edge construction — `edges.*`

Free class.

| knob | frozen | effect on index quality |
|---|---|---|
| `graph_variant` | `g5` | `g5` is the lean projection; `g0-g4` build progressively more node types |
| `embedding_k` | 8 | candidate neighbours per node for similarity edges |
| `max_candidates_per_node` | 24 | cap on candidates considered |
| `max_degree_per_relation` | 12 | global degree cap |
| `relation_degree_caps` | per-relation | the real caps; `collection_co_member` 32, `has_fact` 128 |
| `low_threshold` / `high_threshold` | 0.45 / 0.78 | ambiguous band routed to the refiner |
| `refine_mode` | **`none`** | LLM edge refinement is **off**; turning it on is rebuild-class in cost though not in cache |
| `temporal_normalization` | **true** | required for `at_time` and any temporal ordering |
| `cross_session_portals` | false | `PORTAL` edges; gated off, degrades the scheduler |
| `predicate_embedding_threshold` | 0.92 | see §4 — also gates `shared_value`/`fact_value` |

**Why `temporal_before` is 150 edges.** `pipeline.py` requires
`left.end and right.start and left.end < right.start` — *both* intervals fully
resolved with explicit endpoints. Almost no extracted interval qualifies. The
fix is ordering by `TemporalKey.sort_key` with an `observed_at` fallback, which
is what the algebra already does.

This matters because the temporal stratum is **retrieval-insensitive**: packing
all gold turns changes its accuracy from 0.692 to 0.649 — the sign is reversed.
The evidence arrives and cannot be ordered. Temporal accuracy is an edge and
algebra problem, not a retrieval one.

**Expected coupling with §4**: `shared_entity`, `shared_value`,
`collection_co_member` and `state_next` are built on predicate/value equality,
so a coarser predicate vocabulary should make them denser. If predicate families
form and these edges do *not* get denser, the bottleneck is the edge builder
rather than the vocabulary — which is why the sweep reports edge counts by
relation per arm.

---

## 6. Coarsening — `coarsen.*`

Free class.

| knob | frozen | effect |
|---|---|---|
| `fanout` | 8 | children per routing card |
| `max_levels` | 3 | hierarchy depth |
| `summary_tokens` | 320 | routing-card summary length |
| `cross_session_merge` | **false** | whether L2 cards merge across sessions |

Routing cards are seeds for navigation, so this trades seed precision against
seed recall. **Not currently a priority**: the candidate pool already contains
every gold turn for **200/200** questions, so seeding recall has no headroom.

---

## 7. What to ablate, in order

Ranked by measured evidence of payoff, not by ease.

| # | arm | class | why |
|---|---|---|---|
| 1 | predicate clustering (§4) | **free** | the aggregation unit does not exist; 96.1% singletons; blocks 40% of questions |
| 2 | temporal edge construction (§5) | **free** | 150 edges corpus-wide; the temporal stratum is retrieval-insensitive |
| 3 | `semantic_max_facts_per_scene` 4 → 8 | rebuild | more facts per scene may populate collections directly |
| 4 | extraction schema constraining the predicate vocabulary | rebuild | the root fix if clustering cannot form families |
| 5 | `scenes.max_turns` 4 → 8 | rebuild | halves call count and the 316-token overhead; unmeasured accuracy effect |
| 6 | `coarsen.*` | free | no headroom — seeding is already at 100% |

### Metrics every arm must report

Quality of the index cannot be judged by node and edge counts alone; a denser
graph that scatters a collection is worse than a sparse one.

1. **collection member distribution** — mean, p95, max, singleton rate, count with ≥3 members
2. **predicates per memory** — the direct measure of vocabulary collapse
3. **edge counts by relation** — to catch a builder bottleneck (§5)
4. **collection recall / all-hit** against gold facts (`analyze_v5_6_collection_strategies.py`)
5. **judged answer accuracy** — `turn_all_hit` correlates at only ρ=0.230, so it must not decide an arm
6. **build tokens per memory** and `uncached_generation_calls` — an arm that silently re-extracted is not comparable

### Two traps

* **Free arms must prove they were free.** Assert `uncached_generation_calls == 0`.
  A knob that quietly falls into the rebuild class turns a comparison into two
  different corpora.
* **`turn_all_hit` must not rank arms.** It explains ~5% of the variance in
  judged correctness; the 7pp spread across h0–h9 is worth ~1.6pp of accuracy.
