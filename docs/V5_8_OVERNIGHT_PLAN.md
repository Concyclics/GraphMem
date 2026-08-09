# V5.8 overnight run — working state

Autonomous until the full 510-memory rebuild has results.  Do not stop to ask;
fix bugs and keep going.

## Where the gap is (judged, `../artifacts/v5_6/full_answers/merged/report.json`)

LongMemEval 500 @ 0.712, needs +94 questions for 0.90:

| stratum | n | acc | questions short of 0.90 |
|---|---:|---:|---:|
| multi_session | 133 | 0.489 | 54.7 |
| temporal_reasoning | 133 | 0.714 | 24.7 |
| knowledge_update | 78 | 0.667 | 18.2 |
| single_session_preference | 30 | 0.867 | 1.0 |
| single_session_user | 70 | 0.914 | above |
| single_session_assistant | 56 | 0.964 | above |

Those first three sum to 97.6 against the 94 needed: every one of them has to
reach 0.90, with no slack.

LoCoMo 1540 @ 0.824, needs +118: cat4 47.9, cat2 30.8, cat1 25.7, cat3 13.4 —
sums to 117.8, also no slack anywhere.

## The four levers, in order of evidence

1. **Evidence budget 32 -> 48.**  Already measured at +1.68pp, CI
   [+0.31, +3.14], and never adopted.  Free: no rebuild, and 48 turns still fits
   the 10K answer budget.
2. **The state chain is broken.**  125 of 210 `state_head` nodes have no edge at
   all; `state_next` has 179 edges in a 10-memory graph; 13.5% of facts carry a
   `time_interval`.  `knowledge_update` asks "what is the value *now*", which is
   exactly what that chain is for.
3. **multi_session is a routing failure, not an aggregation failure.**  Session
   routing 0.760 and turn-all-hit-given-correct-routing 0.684 multiply to 0.52,
   against a judged 0.489 — so roughly a quarter of these questions never see
   the right session.  Capacity is not the problem: LME sessions average 10.3
   turns and a 32-turn budget holds three of them, while gold spans 2.42.  The
   routing card is a term bag (`pipeline.py::_compile_routing`), which is what
   to rebuild.
4. **temporal_reasoning is probably not retrieval.**  LoCoMo's temporal category
   has the *highest* all_hit of any stratum (0.744) while LME temporal judges at
   0.714.  Check how many temporal questions take the closed-form date path
   versus falling back to free generation.

## The blind spot that the 100-memory build closes

Every all_hit number so far is LoCoMo.  LME turn-level gold exists for exactly
100 questions (50 multi_session + 50 temporal_reasoning), each its own memory,
and only one landed in the 10-memory samples.  **There is no LME retrieval
measurement at all yet**, so 0.489 cannot currently be split into "the index
missed it" versus "the answer stage missed it".  Building those 100 memories is
the first thing that has to happen, and it is a subset of the 510 anyway.

## Order of work

1. Build the 100 gold-annotated LME memories.
2. Attribute: run `measure_v5_8_arm_recall.py` over them.  Split multi_session's
   failures into routing, packing, and answering.
3. Fix in evidence order: budget first (free), then whatever step 2 blames.
4. Iterate on the 100 until the paired CI stops excluding zero.
5. Full 510 rebuild with the winning config, then answer + judge.

## Settled, do not redo

- **B1 coordinate**: `semantic_scene_summary_chars=0`, `semantic_scene_entities=False`.
  Measured 0.5716 vs the shipped all-three arm's 0.5493, at 103,012 fewer tokens
  per memory.  Lives in `configs/v5/v5_8_final.json`.
- **No truncation**: output ceiling 2,048 -> 32,768, per-memory cap 300,000,
  ledger reserves an estimate (600) per call rather than the ceiling.  Token
  counts from this config are not comparable with earlier runs.
- **Entity merge is a measured no-op** (+4 postings of 80,483; recall +0.0026,
  7 wins / 5 losses / 749 ties).  Postings are stored per card and a session
  card holds only its own session's scenes, so two sessions naming the same
  thing land in posting lists that never meet.  Kept behind a default-off
  switch.  Fixing it means changing which children a card holds, not the folding.
- **Dedup fix changes no recall** (control 0.5716, digit for digit the B1 arm).
  It is a budget fix.
- Ten earlier falsifications: traversal contributes zero, containment relations
  0.00-1.00x, content edges 0.33x/0.00x, routing channel fix hurts, budget split
  nothing, per-operand quota nothing, LLM category field, deterministic head
  noun, geometric clustering, scene-coherent packing.
- **H10 must stay out of the proof-packing set** in `navigator.py` — putting it
  in cost 11pp of LoCoMo.

## Measurement discipline

- Judge self-consistency is 1.69% of questions flipping between two runs of the
  same answers.  **A difference under ~2pp with a CI crossing zero is not
  resolvable in one judged run.**
- Retrieval is deterministic, so one pass per arm is a complete measurement.
  Prefer paired retrieval metrics; spend the judge only on finalists.
- Never draw a global conclusion from a local sample without saying which sample
  it came from.  Four attribution errors this session all had that shape.
