# Graph Harness Query IR — Presentation Notes

## One-line definition

**Graph Harness is the online execution layer over a question-independent persistent memory world. It compiles each incoming question into an evidence contract and assembles the smallest source-complete subgraph that can satisfy it.**

## 30-second version

The offline graph stores what can be known; the online Graph Harness decides what must be proven for this particular question. At question time, it compiles natural language into Query IR: target entities, relation, time and scope constraints, answer algebra, and required evidence roles. It then coordinates dense retrieval, BM25, exact and inverted lookup, and typed graph navigation. Instead of stopping at top-k, it stops only when entity, relation, scope, and provenance certificates pass. The result is a compact, auditable evidence pack for one final answer call.

## 90-second version

Graph Harness should be understood as the online control plane for GraphMem, not as the graph itself and not as another retriever. The memory world is built offline, once, without seeing any evaluation question. It contains lossless turns, compact routing cards, role frames, state chains, reliable typed relations, and persistent search indexes.

When a question arrives, Graph Harness runs online. First, a deterministic compiler converts the question into Query IR. That IR specifies what entity and relation are being asked about, who owns the fact, which temporal or collection scope applies, what kind of answer operation is required, and which evidence roles must be present.

Second, the harness orchestrates heterogeneous retrieval. Dense and BM25 provide semantic and lexical candidates; exact and inverted indexes enforce identity and structured constraints; graph relations recover missing roles such as a reply turn, a previous state, a collection member, or a temporal endpoint.

Third, it verifies the evidence. Four certificates must pass: the exact entity, the requested relation, the correct scope, and source provenance. If a role is missing, the harness expands only along an allowed typed edge, to a maximum depth, instead of performing open-ended graph diffusion. Once the evidence contract is complete, generic operators may compute a date difference, count, set update, or latest valid state, and one final LLM call verbalizes the certified result.

Deterministic Query IR, retrieval, traversal, and operators are local and use zero LLM tokens. A short planner is optional only when deterministic evidence recovery remains incomplete; both planner and final answer usage are counted in the online query budget.

## Four responsibilities to emphasize

1. **Compiler** — Converts an incoming question into targets, constraints, answer algebra, and required evidence roles.
2. **Orchestrator** — Coordinates dense, BM25/FTS, canonical exact match, inverted indexes, source projection, dialogue adjacency, and typed graph relations.
3. **Verifier** — Requires entity, relation, scope, and provenance certificates before structured computation or answer constraints can be trusted.
4. **Governor** — Enforces edge allowlists, graph depth, per-channel quotas, completeness-based stopping, evidence packing, and the token budget.

## Recommended wording for the offline / online boundary

> The HNMW is question-independent and built offline. Graph Harness is question-dependent and executed online for every request.

> Offline, we build a reusable memory world. Online, we compile the current question into a bounded evidence program over that world.

> The graph stores memory; the harness governs how memory is queried.

## Online execution sequence

1. Receive the natural-language question.
2. Deterministically compile Query IR.
3. Route to a small set of memory regions.
4. Generate candidates through independent dense, lexical, exact, structured, adjacency, and graph channels.
5. Fuse candidates with channel quotas so no single retriever dominates.
6. Compare present evidence with required roles.
7. Expand only the typed relation associated with each missing role, with depth and fan-out limits.
8. Verify entity, relation, scope, and provenance.
9. Apply a generic operator only when its operands are certified.
10. Pack the minimal complete evidence subgraph and make one final answer call.

## What Graph Harness is not

- It is not the offline graph-construction process.
- It is not an LLM agent freely roaming the graph.
- It is not a larger top-k retriever.
- It is not a benchmark-specific answer lookup or a list of one-off topic rules.
- It does not treat semantic similarity as proof of answer sufficiency.

## Example: temporal comparison

Question: “How many days earlier was Event A than Event B?”

The online compiler creates the contract `{Event A, time(A), Event B, time(B), source}`. Dense retrieval may initially find Event A and a related date, but the completeness checker sees that Event B is missing. The harness may then traverse only `same_event` or `temporal_endpoint`, verify both event identities and both source turns, and run date difference only after all roles pass. The key point is that retrieval is driven by a missing proof obligation, not by a desire to add more similar text.

## Example: dialogue ownership

Question: “What gift did Speaker B buy?”

The contract requires `{Speaker B, purchase relation, object, reply, source}`. If retrieval finds the question turn but not the answer, the harness follows `dialogue_pair` or `next_turn`. The owner/speaker certificate rejects a nearby preference stated by Speaker A. This is especially important for peer-to-peer dialogue, where the person who mentions an item is not always the owner of the fact.

## Token-accounting language

- **Offline build:** reported separately; amortized across future questions.
- **Online query:** optional planner input/output plus final answer input/output.
- **Zero-LLM-token online work:** deterministic Query IR compilation, FTS/BM25, dense ranking, exact/inverted lookup, graph traversal, certificates, and generic operators.
- Embedding and judge usage are excluded from the reported memory-backbone budget.

## Claims to phrase carefully

- Say: “The full GraphMem system improves LongMemEval by 18.80 percentage points over the evaluated Mem0 baseline.”
- Do not say: “Query IR alone causes the full 18.80-point gain.” A fixed-index module ablation is required for that causal claim.
- Say: “The frozen results motivate completeness-aware online navigation.”
- Do not imply that every historical run used a zero-planner path. The generalized architecture uses deterministic Query IR and permits one short, source-verified planner call when needed.

## Closing line

**HNMW makes the search space small and reusable; the online Graph Harness makes the selected evidence complete, valid, and auditable.**
