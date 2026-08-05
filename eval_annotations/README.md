# GraphMem V5 evaluation annotations

`longmemeval_v5_dev100_gold_turns.jsonl` is the versioned turn-level evidence
asset for the 100-question LongMemEval development subset. It contains source
coordinates and review metadata only; dialogue text, answers, and question-type
labels are intentionally absent.

The annotation process had two gated stages:

1. deterministic candidates were generated inside official gold sessions, then
   a local Qwen3-30B reviewer selected user turns independently per session;
2. candidates were reduced at question level and semantically reviewed against
   every operand and temporal endpoint. Fifty-one question-level evidence sets
   were explicitly changed during adjudication, mainly to restore missing
   aggregation members/endpoints or remove duplicate mentions.

Full-turn offsets are conservative exact spans. All 217 references resolve to a
user turn inside an official LongMemEval gold session. The runtime/build packages
must not import this directory; evaluation code reads it through `graphmem.eval`.

Source and asset hashes, coverage, and methodology version are recorded in
`manifest.json`.
