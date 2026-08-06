# Why the graph has no cross-session key, and what a merge pass could buy

All numbers below are measured on the B1 arm graph (10 memories, 5 LoCoMo +
5 LongMemEval, 4,669 facts, 1,700 scenes, 21,117 edges) and 761 gold-annotated
questions, of which 760 are LoCoMo.  Nothing here is judged; retrieval is
deterministic, so one pass is a complete measurement.

## 1. The topology is a forest of per-session trees

Only relations whose two endpoints both carry a session can be scored for
crossing one; the rest are marked as such rather than reported as zero.

| relation | connects | cross-session |
|---|---|---:|
| scene_contains 10,207 | scene -> evidence/fact | 0 / 10,207 |
| has_fact 3,997 | entity -> fact | endpoint carries no session |
| participates_in 3,377 | scene -> entity | endpoint carries no session |
| refines_to 2,117 | routing_card -> scene | 0 / 1,700 |
| at_time 625 | fact -> time_anchor | endpoint carries no session |
| collection_co_member 607 | fact -> fact | 75 / 312 |
| state_next 179 | fact -> fact | 41 / 179 |
| temporal_before 8 | event -> event | 8 / 8 |

**124 of 21,117 edges (0.59%) leave their session**, about 12 per memory.  This
is the mechanical reason traversal was measured to contribute exactly zero:
there is nothing to traverse to.

## 2. The entity layer degenerates to the speaker list

Entities come from `CAPITAL_RE = \b[A-Z][a-z]{2,}\b` minus a 26-word stop list,
so sentence-initial words ("How", "Can", "Here") become entities.  Of 1,305
distinct names, **60 (4.6%) appear in two or more sessions**, and the eight
widest -- `user, assistant, john, maria, tim, joanna, nate, gina` -- are
**every one of them a speaker name**, checked against `source_turns.speaker`.
A speaker is constant within a memory, so it matches everything and
discriminates nothing.

Neither is any fact field a usable key: `predicate` is 3,864 distinct strings
over 4,669 facts with 92.1% singletons, `owner_id` spans two sessions for 2.4%
of its keys and `value_key` for 2.5%.  This is the same failure mode already
measured for an LLM-supplied category field (390 categories per memory, 6.3%
reuse) -- independent extraction calls cannot see each other's vocabulary.

## 3. What a cross-session key could actually serve

Per question, from its gold turns:

| stratum | n | gold turns | gold sessions | multi-session |
|---|---:|---:|---:|---:|
| locomo_cat4 | 418 | 1.04 | 1.00 | **0.00** |
| locomo_cat2 | 156 | 1.10 | 1.06 | 0.05 |
| locomo_cat3 | 44 | 1.93 | 1.66 | 0.36 |
| locomo_cat1 | 142 | 2.92 | 2.61 | **0.96** |

cat4 -- 55% of the question set -- never needs a second session.  Any
cross-session mechanism is scored on cat1 and part of cat3.

For cat1's 137 multi-session questions:

| | share |
|---|---:|
| the query's own discriminative terms reach every gold session | 0.292 |
| a discriminative bridging term exists | 0.555 |
| ... and the query contains it | 0.204 |
| ... **and it appears only in the gold text** | **0.350** |

The routing channel keys postings on *query* terms
(`facts.py` -> `view.route_children(keys)` with `keys` from the query), so for
that 0.350 a merged key on its own fires nothing.  Reaching the second session
requires two hops: query -> session A -> bridging term in A's text -> session B.
The merge pass's contribution is giving hop 2 an edge to walk.

## 4. Two hops: the reachability is there, the ranking is not

Hop 1 is the query's discriminative terms over a 32-turn budget; hop 2 admits
sessions voted for by bridging terms carried by the hop-1 turns, weighted by
1/(sessions the term spans).  cat1, 137 questions:

| hop-2 sessions admitted | reaches all gold sessions | sessions pulled |
|---|---:|---:|
| 0 (current) | 0.292 | 4.5 / 26.8 |
| 1 | 0.336 | 5.3 |
| 3 | 0.372 | 6.9 |
| 8 | 0.474 | 11.0 |
| unbounded | **0.693** | 19.9 |

**Every gold session is two-hop reachable for 69.3% of cat1**, up from 29.2%.
The reachability is real.  But recovering it needs 19.9 of 26.8 sessions, and
the candidate pool is already 93% of the memory -- widening it is not the
deliverable, since the pack has 32 seats and cat1 already spreads them over 10
sessions.  At a defensible k=3 the gain is 0.292 -> 0.372, about 11 questions of
761 (+1.4pp), against a pool that grew from 4.5 to 6.9 sessions and a cat4 block
of 418 single-session questions that can only lose from the widening.

**The gap between 0.372 and 0.693 is ranker quality, not missing structure.**
Only one hop-2 ranker was measured (inverse session-span voting).  Before
building the merge pass into `graphmem.build`, a better hop-2 ranker should be
falsified offline the same way -- the structure it would consume is already
present in the frozen graph.

## 5. Two build defects found on the way

- `_compiled_summary` still overwrites the LLM scene sentence with a
  concatenation of fact triples, which V5.8 was meant to fix.  Scene summary and
  routing card both read as bags: `'Jon lost job as a banker yesterday Jon
  started business dance studio Gina lost job at Door Dash this month ...'`,
  with the same fragment repeated inside one card.
- Only 13.5% of facts carry a `time_interval`, and `temporal_before` has 8 edges
  in the whole 10-memory graph.  cat2 is not being served by the temporal layer;
  it is being served lexically.
