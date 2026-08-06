# How mem0, LightMem and EverOS handle extraction and relations

Read against one question: why did GraphMem's collection index fail on aggregation
questions, and does any baseline solve the part we are stuck on?

## The failure being diagnosed

Measured over 149 gold-annotated aggregation questions ("how many model kits have
I worked on or bought"):

| | value |
|---|---:|
| A question's gold facts landing in **different** collection chains | **91.5%** |
| Judged accuracy, LME aggregation / non-aggregation | 0.649 / 0.775 |
| Judged accuracy, LoCoMo aggregation / non-aggregation | 0.681 / 0.830 |

The chain keys on the predicate (369.3 distinct per memory) while the question
keys on the object class. "How many model kits" has gold facts whose predicates
are just / start / finish; "how many movie festivals" has participate / attend /
volunteer / has / express. A count therefore has no set to range over, and the
answer stage faithfully counts whatever subset survived ranking.

## What each baseline does

| | Extraction product | Relations | Cross-call consistency |
|---|---|---|---|
| **mem0** | Flat natural-language fact strings | None in the default path; graph store is an optional layer | Not needed |
| **LightMem** | Flat fact sentences + `source_id` | None | Not needed |
| **EverOS** | Atomic facts + episodes | **Incremental embedding clusters** | **By geometry, not by the LLM** |

### mem0 — `mem0/configs/prompts.py:15`

`FACT_RETRIEVAL_PROMPT` asks for `{"facts": ["Looking for a restaurant in San
Francisco", ...]}`. No predicate, no value, no scope, no category. Retrieval is
vector search over those strings; `DEFAULT_UPDATE_MEMORY_PROMPT` then reconciles
new facts against retrieved old ones (ADD / UPDATE / DELETE / NONE).

There is no index key to get wrong because there is no index key.

### LightMem — `src/lightmem/memory/prompts.py:1`

Same shape, plus explicit topic segmentation (`--- Topic X ---`) and a
`source_id` per fact. Its LoCoMo prompt pushes in the **opposite** direction from
categorization — a "CRITICAL - Preserve All Specific Details" section demands
full names, complete location names, specific event names, numbers. One clause
gestures at aggregation ("If multiple related items mentioned → may infer general
pattern. Keep BOTH specific facts AND inferred insights as separate entries") but
it is left to the model per call.

Neither baseline offers anything to borrow for a collection key.

### EverOS — `src/everos/memory/strategies/trigger_profile_clustering.py`

This is the one that solves the part we are stuck on, and it does it by
**inverting the order of geometry and the LLM**.

```
EpisodeExtracted
  → embed(episode_text)
  → build a size-1 Cluster(centroid=vector, count=1, members=[entry_id])
  → load the owner's *existing persisted* clusters
  → cluster_by_geometry(new, existing, threshold=0.65, time_window_days=7)
        merge above threshold, otherwise keep as a new cluster
  → upsert back to SQLite
  → emit ProfileClusterUpdated
       → extract_user_profile: the LLM *names and summarizes* the cluster
```

Defaults live in `src/everos/config/settings.py:274` — `threshold=0.65`,
`time_window_days=7.0`.

The LLM never invents a grouping key. It is handed a group that geometry already
formed and asked only to describe it. Consistency across calls is structural:
every arriving item is compared against one persistent cluster set.

## Why this is exactly our defect

GraphMem asked the LLM for the grouping key directly — a `k` field per fact,
emitted by `scene_semantic`. Measured on 9 memories:

| | value |
|---|---:|
| Distinct categories / memory | 390.5 (worse than the predicate's 369.3) |
| Categories reused >= 3 times, as a share of facts | **6.3%** |
| `k` that is a leaked turn alias (`s1t0`) | 2.9% |
| `k` truncated at the 24-char schema cap | 43.8% |

2,012 distinct categories over 2,222 facts. The cause is structural, not
prompting: one memory takes ~129 independent `scene_semantic` calls, each seeing
2 scenes, **none able to see what the others chose**. Asking them to converge on
a shared vocabulary without a shared vocabulary is the same experiment that
already failed once as "LLM predicate vocabulary" (0.929 singleton rate).

A deterministic head-noun key avoids the consistency problem but hits a semantic
ceiling: it merges kit==kit and festival==festival, but not cake / baguette /
sourdough, nor fitbit / nebulizer / glucose meter.

| Route | Clusters/memory (gate <60) | Gold facts sharing a key (gate >=0.70) |
|---|---:|---:|
| V5.6 chain (predicate-keyed) | 373 | 0.085 |
| LLM `k` per fact | 390.5 | — (6.3% reuse) |
| Deterministic head noun | 373.3 | 0.532 (LME 0.482 / LoCoMo 0.591) |

Both fail. The LLM has the semantics but cannot be consistent; the deterministic
rule is consistent but has no semantics.

**EverOS's incremental centroid merge has both**: embeddings carry the semantics
that merges cake with sourdough, and comparing against a persistent cluster set
makes the assignment consistent by construction.

This is also distinct from GraphMem's earlier failed embedding experiment, which
clustered *predicate strings* — whole propositions — in one shot (0.922 singleton).
Clustering the fact's relation+object with an explicit merge threshold is a
different operation with a tunable knob.
