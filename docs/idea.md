# idea.md — Token-Efficient Hierarchical Memory Graph Demo Idea

## 0. Purpose of This Document

This document explains the research idea and first-demo direction for Codex. The immediate goal is **not** to build the final paper-quality system. The goal is to quickly implement a demo that measures how much **build-stage token cost** can be reduced when we combine:

1. **Evidence-preserving memory**: keep raw user-assistant dialogue pairs as leaf evidence.
2. **LLMLingua-based input compression**: compress long build prompts before sending them to DeepSeek V4.
3. **K-way hierarchical summarization**: summarize groups of dialogue pairs into multi-level memory nodes.
4. **Coarse-to-refine graph construction**: build graph edges first at coarse summary levels, then refine only related subtrees.
5. **Lightweight retrieval first**: use embedding/BM25/keyword/entity matching before optional LLM-based edge refinement or search.

The demo should focus on **measuring token cost reduction** and basic retrieval/QA feasibility. Accuracy can be measured on a small subset if time permits, but the first target is a reliable cost/latency/token accounting pipeline.

---

## 1. Problem We Want to Solve

Long-running LLM agents need persistent memory, but existing external text memory systems are expensive.

### Problem 1: Per-turn extraction is expensive

Many memory systems read every dialogue turn and use an LLM to extract facts, events, profiles, or memory units. This creates build-stage token cost that grows roughly linearly with the number of turns.

```text
conversation turns -> LLM extraction per turn -> memory units -> indexing
```

Even token-efficient systems such as Mem0 V3 additive remove UPDATE/DELETE maintenance but still perform LLM extraction. In our preliminary LongMemEval-S subset test, Mem0 V3 additive still consumed very high add-stage prompt tokens.

### Problem 2: Dynamic maintenance costs tokens

Traditional memory systems often do ADD / UPDATE / DELETE / MERGE operations. These operations require reading old memory and new memory together, which adds extra token cost beyond the original extraction.

Mem0's new ADD-only design reduces this maintenance cost, but it also leaves multi-version state resolution to retrieval and answering.

### Problem 3: Extracted memory can lose information

If the system stores only extracted facts, any detail not extracted at write time is lost for future questions. Examples:

- conditional plans,
- exceptions,
- uncertainty,
- implicit causes,
- temporary states,
- historical vs current values.

Therefore, the system should preserve raw evidence somewhere.

### Problem 4: Flat retrieval is weak for implicit and multi-hop reasoning

Embedding/BM25 retrieval works well when the query explicitly matches the stored text. It is weaker when the answer requires:

- connecting events across sessions,
- resolving contradiction or updates,
- following temporal chains,
- inferring causes from multiple fragments,
- summarizing user profile or long-term research direction.

### Problem 5: Full graph construction is too expensive

Graph memory can help multi-hop reasoning, but naïvely constructing edges among all memory units may require O(N^2) candidate comparisons. GraphRAG-style indexing may also require LLM-based entity and relationship extraction.

This is too expensive for online long-running agent memory.

### Problem 6: Large top-k retrieval is bad for small answer models

Some memory systems rely on large top-k retrieval, e.g. retrieving many compressed memories to increase recall. This may work for strong frontier models, but for smaller models such as Qwen3-4B, low-density retrieved context can distract the answer model and increase latency/token usage.

The goal should be **high-density retrieval**, not just high top-k recall.

---

## 2. Core Idea: Hierarchical Memory Graph (HMG / MLGM)

We build a **Multi-Level Graph Memory** over conversations.

The key idea is:

> Preserve raw dialogue as leaf evidence, compress/summarize dialogue hierarchically, build a small graph at the root level, and recursively refine edges only between children of connected parents.

This follows the spirit of multilevel graph partitioning algorithms such as METIS and KaHIP:

```text
coarsen -> process small graph -> uncoarsen/refine
```

For memory construction, this becomes:

```text
raw turns -> K-way summary tree -> root graph -> top-down edge refinement
```

---

## 3. Important Design Principle: Compression Is Not Truth

We want to use LLMLingua to compress inputs before DeepSeek V4 calls, but compressed text should **not** replace raw evidence.

Correct principle:

```text
Raw dialogue is the source of truth.
Compressed text is only used to reduce prompt tokens in build-stage LLM calls.
```

This means each leaf node stores raw text, while internal summaries may be produced from compressed inputs.

```text
LeafNode.raw_text = original user-assistant pair
LeafNode.compressed_text = optional LLMLingua-compressed version for build prompts
SummaryNode.summary = DeepSeek-generated summary from compressed or raw child text
```

If a query needs exact evidence, retrieval should still be able to return raw text.

---

## 4. Proposed Memory Layers

### 4.1 Leaf Evidence Layer

Each leaf node is one user-assistant pair or one short dialogue chunk.

```python
LeafNode = {
    "node_id": str,
    "node_type": "leaf",
    "session_id": str,
    "turn_id": int,
    "timestamp": str | None,
    "raw_text": str,
    "compressed_text": str | None,
    "raw_tokens": int,
    "compressed_tokens": int | None,
    "keywords": list[str],
    "entities": list[str],
    "embedding": list[float] | None,
    "metadata": dict,
}
```

For the first demo, keywords/entities can be simple and cheap:

- keywords: jieba / sklearn TF-IDF / BM25 tokens / simple noun phrases;
- entities: optional; can start with empty list or regex/simple NER;
- embeddings: use the existing local embedding model if available.

### 4.2 Summary Tree Layer

Group K adjacent leaf or summary nodes and summarize them into one parent.

```python
SummaryNode = {
    "node_id": str,
    "node_type": "summary",
    "level": int,
    "children": list[str],
    "covered_leaf_ids": list[str],
    "summary": str,
    "keywords": list[str],
    "entities": list[str],
    "time_range": tuple | None,
    "raw_input_tokens": int,
    "compressed_input_tokens": int,
    "llm_prompt_tokens": int,
    "llm_completion_tokens": int,
    "metadata": dict,
}
```

For level-1 summaries, children are leaf nodes. For higher levels, children are summary nodes.

### 4.3 Root Graph Layer

Each session eventually has one or a few root summary nodes. We first build edges only among these root nodes.

```python
Edge = {
    "edge_id": str,
    "src": str,
    "dst": str,
    "level": int,
    "edge_type": str,
    "score": float,
    "signals": dict,
    "refined_by_llm": bool,
    "llm_prompt_tokens": int,
    "llm_completion_tokens": int,
}
```

Initial edge types can be weak:

```text
related_to
same_topic
same_entity
possible_update
possible_conflict
temporal_neighbor
possible_causal_chain
```

### 4.4 Recursive Refined Graph Layer

Only if two parent summary nodes are connected do we check edges among their children.

Pseudo-rule:

```python
for edge(parent_a, parent_b) in graph_edges_at_level_l:
    for child_a in parent_a.children:
        for child_b in parent_b.children:
            score = lightweight_edge_score(child_a, child_b)
            if score >= high_threshold:
                add_weak_edge(child_a, child_b)
            elif low_threshold <= score < high_threshold and enable_llm_refine:
                add_llm_refined_edge(child_a, child_b)
            else:
                skip
```

This avoids checking all leaf pairs.

---

## 5. LLMLingua Integration

### 5.1 Why Use LLMLingua

The first demo should test whether compressing build prompts before DeepSeek V4 calls reduces token cost significantly.

LLMLingua can compress long prompts using a smaller model. We only need a wrapper with the following modes:

```text
compressor = none      # no compression
compressor = truncate  # simple fallback baseline
compressor = llmlingua # PromptCompressor
```

Optional future modes:

```text
compressor = longllmlingua
compressor = llmlingua2
```

### 5.2 Compression Targets

We should test several target ratios or token budgets:

```text
none
0.75x original tokens
0.50x original tokens
0.33x original tokens
0.25x original tokens
```

In code, expose both modes:

```bash
--compression-ratio 0.5
--target-token 512
```

If both are provided, `target-token` wins.

### 5.3 What to Compress

Compress only text that is sent to the LLM for build-stage summarization/refinement.

Compress:

- child raw texts before level-1 summarization;
- child summaries before higher-level summarization if long;
- edge-refinement context if needed.

Do not compress:

- stored raw evidence;
- final evaluation gold answers;
- logs needed for exact token accounting.

---

## 6. DeepSeek V4 API Integration

The demo should use DeepSeek V4 via an OpenAI-compatible client.

Recommended environment variables:

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

For cheaper/faster runs, allow:

```bash
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

Client wrapper should log API usage returned by the model:

```python
LLMResult = {
    "text": str,
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int,
    "latency_sec": float,
    "model": str,
}
```

Never log API keys.

---

## 7. First Demo Scope

The first demo should implement only the minimal pipeline needed to measure token savings.

### Must Have

1. Load a small LongMemEval-S / LoCoMo-like dialogue dataset.
2. Convert each user-assistant pair into a leaf node.
3. Optionally compress leaf text with LLMLingua before summarization.
4. Build a K-way summary tree using DeepSeek V4.
5. Build lightweight edges between root summary nodes.
6. Optionally refine a small number of ambiguous/high-value edges with DeepSeek V4.
7. Run simple retrieval for a few questions.
8. Log token usage, call count, latency, number of nodes, number of candidate edges, and final prompt length.

### Nice to Have

1. Simple answer generation and judge.
2. Compare retrieval variants.
3. Visualize tree/graph structure.
4. Export cost tables as CSV/JSON/Markdown.

### Not Needed in First Demo

1. Full BEAM evaluation.
2. Full GraphRAG baseline.
3. Full MemoryOS/Mem0/LightMem reproduction.
4. Complex entity extraction.
5. Production-grade graph database.
6. Perfect temporal reasoning.

---

## 8. Expected Demo Variants

The first demo should compare these variants:

| Variant | Description | Purpose |
|---|---|---|
| `flat_raw_rag` | Raw leaf retrieval, no summaries | Basic retrieval baseline |
| `kway_no_compress` | K-way tree, no LLMLingua | Measure hierarchy-only cost |
| `kway_llmlingua` | K-way tree with LLMLingua-compressed build prompts | Main token-saving variant |
| `hmg_light` | K-way tree + lightweight root/child edges | Measure graph overhead without LLM refinement |
| `hmg_refine_sample` | HMG + DeepSeek edge refinement on limited candidate edges | Estimate marginal value/cost of LLM edge refinement |

Primary comparison:

```text
kway_no_compress vs kway_llmlingua
```

Secondary comparison:

```text
hmg_light vs hmg_refine_sample
```

---

## 9. Metrics to Log

### 9.1 Build Metrics

```text
num_sessions
num_turn_pairs
num_leaf_nodes
num_summary_nodes
num_edges
num_candidate_edges_checked
num_llm_calls
build_prompt_tokens
build_completion_tokens
build_total_tokens
build_latency_sec
compression_origin_tokens
compression_compressed_tokens
compression_ratio_actual
```

### 9.2 Retrieval / Answer Metrics

```text
num_queries
retrieved_node_count
retrieved_leaf_count
retrieved_summary_count
retrieved_edge_count
answer_prompt_tokens
answer_completion_tokens
answer_latency_sec
judge_score or accuracy if available
```

### 9.3 Cost-Density Metrics

```text
build_tokens_per_turn_pair
build_tokens_per_session
answer_tokens_per_query
compression_savings_ratio
llm_calls_per_turn_pair
candidate_edges_checked_per_leaf
```

If answer evaluation is enabled:

```text
accuracy_per_1k_build_tokens
accuracy_per_1k_answer_tokens
```

---

## 10. Main Hypotheses for Demo

### H1: LLMLingua can significantly reduce build prompt tokens

Expected result:

```text
kway_llmlingua build_prompt_tokens << kway_no_compress build_prompt_tokens
```

### H2: K-way summarization reduces LLM call count compared with per-turn extraction

Expected result:

```text
num_llm_calls grows with number of merge nodes, not number of raw facts
```

### H3: Lightweight graph construction has low extra token cost

Expected result:

```text
hmg_light adds edges without adding LLM tokens, except optional refinement
```

### H4: Selective edge refinement has measurable marginal cost

Expected result:

```text
hmg_refine_sample should show exactly how many extra tokens LLM edge refinement costs
```

The first demo does not need to prove final accuracy. It should reveal whether the token-saving direction is promising.

---

## 11. Code Organization Suggestion

```text
hmg_demo/
  README.md
  idea.md
  plan01.md
  configs/
    demo.yaml
  data/
    sample.json
  hmg/
    __init__.py
    data_loader.py
    tokenizer.py
    compressor.py
    deepseek_client.py
    nodes.py
    tree_builder.py
    edge_builder.py
    retriever.py
    evaluator.py
    stats.py
  scripts/
    run_build_demo.py
    run_query_demo.py
    summarize_run.py
  runs/
    .gitkeep
```

---

## 12. References for Codex

- DeepSeek API Docs: https://api-docs.deepseek.com/
- Microsoft LLMLingua GitHub: https://github.com/microsoft/LLMLingua
- LLMLingua paper: https://arxiv.org/abs/2310.05736
- LongLLMLingua paper: https://arxiv.org/abs/2310.06839
- METIS: https://github.com/KarypisLab/METIS
- KaHIP: https://github.com/KaHIP/KaHIP
- GraphRAG paper: https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/
- HippoRAG NeurIPS page: https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html
- Existing proposal draft: `proposal_en.md`
