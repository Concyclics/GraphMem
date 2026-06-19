# plan01.md — First Demo Plan for DeepSeek V4 + LLMLingua HMG Prototype

## 0. Goal

Build a minimal demo to estimate how much build-stage token cost can be reduced by using:

1. LLMLingua prompt compression,
2. K-way hierarchical summarization,
3. lightweight coarse graph construction,
4. optional DeepSeek V4 edge refinement.

This demo is for quick engineering validation, not final benchmark reporting.

Primary output:

```text
A table comparing token cost, LLM calls, latency, graph size, and retrieval prompt length across demo variants.
```

Primary question:

```text
Can LLMLingua + K-way hierarchical memory construction reduce build-stage DeepSeek tokens by a large margin compared with uncompressed K-way construction and per-turn extraction-style baselines?
```

---

## 1. Demo Principles

### 1.1 Keep raw evidence

Raw user-assistant pairs must be stored unchanged.

```text
Raw evidence is never overwritten by compressed text or summaries.
```

### 1.2 Compress only LLM build prompts

LLMLingua compression should be applied only when preparing inputs for DeepSeek V4 calls:

- level-1 summarization,
- higher-level summarization,
- optional edge refinement.

Do not use compressed text as the only stored memory.

### 1.3 Log every token

Every DeepSeek call must log:

```text
prompt_tokens
completion_tokens
total_tokens
latency_sec
model
call_type
```

Every compression call should log:

```text
origin_tokens
compressed_tokens
actual_ratio
compress_latency_sec
compressor_name
```

---

## 2. Environment

### 2.1 Python packages

Create a minimal `requirements.txt`:

```text
openai>=1.0.0
llmlingua
pydantic
pyyaml
numpy
pandas
tqdm
rank-bm25
networkx
transformers
sentencepiece
```

Optional packages:

```text
scikit-learn
jieba
spacy
matplotlib
```

### 2.2 Environment variables

Use the DeepSeek OpenAI-compatible API.

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

Optional cheaper/faster run:

```bash
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

Never write API keys to logs.

---

## 3. Input Data

### 3.1 Supported input format

Start with a simple JSON or JSONL format.

```json
{
  "conversation_id": "conv_001",
  "sessions": [
    {
      "session_id": "s001",
      "turns": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
      ]
    }
  ],
  "questions": [
    {
      "question_id": "q001",
      "question": "...",
      "answer": "...",
      "type": "multi-session"
    }
  ]
}
```

### 3.2 Turn-pair conversion

Convert consecutive user-assistant pairs into leaf nodes:

```text
user turn + assistant turn -> LeafNode.raw_text
```

If the format has system messages, ignore them for the first demo unless needed.

---

## 4. Demo Variants

Implement these variants first.

| Variant | Compression | Tree | Graph | LLM Edge Refine | Purpose |
|---|---|---|---|---|---|
| `flat_raw_rag` | no | no | no | no | raw retrieval baseline |
| `kway_no_compress` | no | yes | no | no | hierarchy-only cost |
| `kway_llmlingua` | yes | yes | no | no | compression gain |
| `hmg_light` | yes | yes | yes | no | graph overhead without LLM edge cost |
| `hmg_refine_sample` | yes | yes | yes | yes, limited | estimate edge-refine marginal cost |

The most important comparison is:

```text
kway_no_compress vs kway_llmlingua
```

The second most important comparison is:

```text
hmg_light vs hmg_refine_sample
```

---

## 5. Configuration

Create `configs/demo.yaml`:

```yaml
run:
  name: demo01
  seed: 42
  output_dir: runs/demo01

data:
  path: data/sample.json
  max_conversations: 3
  max_sessions_per_conversation: 5
  max_turn_pairs_per_session: 50

llm:
  provider: deepseek
  base_url_env: DEEPSEEK_BASE_URL
  api_key_env: DEEPSEEK_API_KEY
  model_env: DEEPSEEK_MODEL
  default_model: deepseek-v4-pro
  temperature: 0.0
  max_retries: 3
  timeout_sec: 120

compression:
  enabled: true
  compressor: llmlingua
  target_token: null
  compression_ratio: 0.5
  fallback: truncate

tree:
  fanout_k: 4
  max_summary_tokens: 256
  max_levels: 10

edge:
  enabled: true
  high_threshold: 0.72
  low_threshold: 0.45
  max_refine_edges: 20
  refine_policy: ambiguous_only

retrieval:
  top_roots: 3
  beam_width: 4
  max_leaf_nodes: 10
  max_context_tokens: 5000
```

---

## 6. Implementation Tasks

### Task 1: DeepSeek client wrapper

File:

```text
hmg/deepseek_client.py
```

Responsibilities:

- Use OpenAI-compatible SDK.
- Read API config from env.
- Support `chat(messages, call_type, metadata)`.
- Return parsed text and usage stats.
- Retry on transient API errors.
- Never log API keys.

Expected return object:

```python
@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_sec: float
    model: str
    call_type: str
    raw_response: dict | None = None
```

Pseudo-code:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

response = client.chat.completions.create(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
    messages=messages,
    temperature=0.0,
)
```

### Task 2: Compression wrapper

File:

```text
hmg/compressor.py
```

Responsibilities:

- Implement identity compressor.
- Implement truncate compressor.
- Implement LLMLingua compressor.
- Return compressed text and compression stats.

Expected API:

```python
class BaseCompressor:
    def compress(self, text: str, instruction: str = "", question: str = "", target_token: int | None = None, ratio: float | None = None) -> CompressionResult:
        ...
```

Expected result:

```python
@dataclass
class CompressionResult:
    compressed_text: str
    origin_tokens: int
    compressed_tokens: int
    ratio: float
    latency_sec: float
    compressor: str
```

LLMLingua pseudo-code:

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor()
result = compressor.compress_prompt(
    text,
    instruction=instruction,
    question=question,
    target_token=target_token,
)
```

Important: if LLMLingua fails due to model download/GPU/runtime issues, fall back to truncate or identity and log the failure.

### Task 3: Data loader

File:

```text
hmg/data_loader.py
```

Responsibilities:

- Load JSON/JSONL.
- Normalize to conversations/sessions/turn pairs.
- Assign stable IDs.

Output:

```python
List[Conversation]
```

### Task 4: Node schemas

File:

```text
hmg/nodes.py
```

Define:

```python
LeafNode
SummaryNode
Edge
BuildStats
CompressionStats
```

Use Pydantic or dataclasses. JSON-serializable is required.

### Task 5: Tree builder

File:

```text
hmg/tree_builder.py
```

Responsibilities:

1. Build leaf nodes from turn pairs.
2. Compress child text if enabled.
3. Call DeepSeek to summarize K children into a parent node.
4. Repeat until one session root remains.
5. Save all nodes to JSONL.

Summarization prompt should be short and stable:

```text
You are building a compact memory summary for a long-running LLM agent.
Summarize the following child memory nodes.
Keep only information useful for future user-memory queries.
Preserve:
- key entities
- user preferences
- plans and decisions
- temporal anchors
- causes and outcomes
- contradictions or updates
Return JSON with fields:
summary, keywords, entities, events, state_hints, conflict_hints.
```

For demo robustness, parse JSON if possible; otherwise keep raw text summary and extract simple fields with fallback.

### Task 6: Edge builder

File:

```text
hmg/edge_builder.py
```

Responsibilities:

1. Build root-level candidate edges.
2. Score candidate pairs using lightweight signals.
3. Add weak edges above high threshold.
4. Drop below low threshold.
5. Optionally call DeepSeek for ambiguous edges.
6. Recursively expand children of connected parents.

Lightweight score can start simple:

```python
score = (
    0.5 * embedding_cosine
    + 0.2 * keyword_jaccard
    + 0.2 * entity_jaccard
    + 0.1 * temporal_score
)
```

If embeddings/entities are unavailable, use keyword/BM25 only.

LLM edge refinement prompt:

```text
Given two memory nodes, decide whether they should be connected in a long-term agent memory graph.
If connected, classify the relation as one of:
related_to, same_topic, same_entity, temporal_neighbor, update, conflict, causal, elaboration.
Return JSON:
{connected: bool, relation_type: str, confidence: float, rationale: str}
```

### Task 7: Retriever

File:

```text
hmg/retriever.py
```

First demo retrieval can be simple:

1. Score root/internal/leaf nodes by BM25 or embedding.
2. Start from top root/internal nodes.
3. Expand children and 1-hop graph neighbors.
4. Collect top leaf evidence and ancestor summaries.
5. Cap final context tokens.

Output:

```python
RetrievedContext = {
    "query": str,
    "summary_nodes": list[SummaryNode],
    "leaf_nodes": list[LeafNode],
    "edges": list[Edge],
    "context_text": str,
    "context_tokens": int,
}
```

### Task 8: Stats and reporting

File:

```text
hmg/stats.py
```

Collect all stats in one JSON file:

```json
{
  "variant": "kway_llmlingua",
  "num_conversations": 3,
  "num_sessions": 12,
  "num_leaf_nodes": 240,
  "num_summary_nodes": 81,
  "num_edges": 112,
  "num_candidate_edges_checked": 500,
  "num_llm_calls": 81,
  "build_prompt_tokens": 123456,
  "build_completion_tokens": 23456,
  "build_total_tokens": 146912,
  "compression_origin_tokens": 200000,
  "compression_compressed_tokens": 90000,
  "compression_ratio_actual": 0.45,
  "build_latency_sec": 123.4
}
```

Also export a Markdown table:

```text
runs/demo01/summary.md
```

---

## 7. Scripts

### 7.1 Build demo

```bash
python scripts/run_build_demo.py \
  --config configs/demo.yaml \
  --variant kway_llmlingua
```

Run all variants:

```bash
python scripts/run_build_demo.py --config configs/demo.yaml --variant flat_raw_rag
python scripts/run_build_demo.py --config configs/demo.yaml --variant kway_no_compress
python scripts/run_build_demo.py --config configs/demo.yaml --variant kway_llmlingua
python scripts/run_build_demo.py --config configs/demo.yaml --variant hmg_light
python scripts/run_build_demo.py --config configs/demo.yaml --variant hmg_refine_sample
```

### 7.2 Query demo

```bash
python scripts/run_query_demo.py \
  --run-dir runs/demo01/kway_llmlingua \
  --questions data/sample_questions.json
```

### 7.3 Summarize results

```bash
python scripts/summarize_run.py --run-root runs/demo01
```

Expected output:

```text
runs/demo01/summary.md
runs/demo01/summary.csv
runs/demo01/summary.json
```

---

## 8. Output Files

For each variant:

```text
runs/demo01/{variant}/
  nodes.jsonl
  edges.jsonl
  build_stats.json
  compression_stats.jsonl
  llm_calls.jsonl
  retrieval_results.jsonl        # optional
  answer_results.jsonl           # optional
```

Global summary:

```text
runs/demo01/summary.md
runs/demo01/summary.csv
runs/demo01/summary.json
```

---

## 9. Success Criteria for Demo01

The demo is successful if it produces the following table for at least 2 variants:

| Variant | Build Prompt Tokens | Build Completion Tokens | LLM Calls | Compression Ratio | Build Latency | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| kway_no_compress | ... | ... | ... | 1.00 | ... | ... | ... |
| kway_llmlingua | ... | ... | ... | ... | ... | ... | ... |
| hmg_light | ... | ... | ... | ... | ... | ... | ... |
| hmg_refine_sample | ... | ... | ... | ... | ... | ... | ... |

Minimum required conclusion:

```text
LLMLingua compression reduces build prompt tokens by X% under the same K-way tree setting.
```

Better conclusion:

```text
HMG-light adds graph structure with near-zero additional LLM token cost.
Selective LLM edge refinement costs Y extra tokens and Z extra calls.
```

Best conclusion:

```text
Under a small query subset, HMG retrieval reduces answer prompt tokens or improves accuracy compared with flat raw retrieval.
```

---

## 10. Guardrails and Engineering Notes

1. Do not hardcode API keys.
2. Do not overwrite raw evidence with compressed text.
3. Store every LLM call and usage count.
4. Make the demo resumable: if nodes already exist, skip completed steps unless `--force` is set.
5. Use small datasets first to avoid accidental API cost explosion.
6. Add `--dry-run` mode that estimates prompts and tokens without calling DeepSeek.
7. Add `--max-llm-calls` to cap cost.
8. Add `--max-refine-edges` to cap edge-refinement cost.
9. If LLMLingua fails, fallback to truncate and mark `compressor_fallback=true`.
10. Use deterministic prompts and `temperature=0.0`.

---

## 11. Suggested Development Order

### Step A: Build token accounting first

Before optimizing anything, ensure token usage is logged correctly.

Implement:

```text
deepseek_client.py
stats.py
simple tokenizer fallback
```

### Step B: Implement K-way tree without compression

Run:

```bash
python scripts/run_build_demo.py --variant kway_no_compress
```

Verify:

```text
nodes.jsonl exists
build_stats.json has LLM calls and tokens
```

### Step C: Add LLMLingua compression

Run:

```bash
python scripts/run_build_demo.py --variant kway_llmlingua
```

Compare:

```text
build_prompt_tokens
compression_origin_tokens
compression_compressed_tokens
```

### Step D: Add lightweight graph

Run:

```bash
python scripts/run_build_demo.py --variant hmg_light
```

Verify:

```text
edges.jsonl exists
num_candidate_edges_checked is logged
no extra LLM tokens except summarization
```

### Step E: Add limited LLM edge refinement

Run:

```bash
python scripts/run_build_demo.py --variant hmg_refine_sample --max-refine-edges 20
```

Verify:

```text
edge refinement tokens are separated from summary tokens
```

### Step F: Optional retrieval/QA

Run a small number of questions and log:

```text
retrieved node count
context tokens
answer tokens
judge score if available
```

---

## 12. What Not to Optimize Yet

Do not spend time on:

- perfect graph visualization,
- complete benchmark reproduction,
- exact Mem0/LightMem parity,
- GPU optimization for LLMLingua,
- complex NER,
- graph database integration,
- advanced agentic search.

The first demo should answer one question:

```text
Is the token-saving direction real enough to justify building the full system?
```

---

## 13. Reference Links

- DeepSeek API Docs: https://api-docs.deepseek.com/
- Microsoft LLMLingua GitHub: https://github.com/microsoft/LLMLingua
- LLMLingua paper: https://arxiv.org/abs/2310.05736
- LongLLMLingua paper: https://arxiv.org/abs/2310.06839
- Existing proposal draft: `proposal_en.md`
- Idea explanation: `idea.md`
