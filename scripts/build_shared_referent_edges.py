#!/usr/bin/env python3
"""Cross-session `shared_referent` edges, derived from term rarity in the raw text.

Every symbolic join key the graph currently has is built from text the extractor
wrote -- predicate, value_key, entity name -- and all three degenerate the same
way: 92.1% of predicates occur once, so any key containing one is a fact id.  The
measured consequence is that 100% of the graph's edges have single-session
evidence, and the three worst-scoring question types are exactly the ones whose
answers span sessions.

This pass skips the extractor.  A term that appears in a handful of a memory's
sessions is a referent -- a specific thing that got discussed more than once --
and two scenes that share several of them are talking about the same thing.
Measured on 324 cross-session LongMemEval questions, gold session pairs share
15-23 such terms against 1.1 for a random pair, and the gold pair lands in the
top 5% of all pairs for 85-96% of questions.

No LLM call, at build time or query time; the input is `source_turns` and the
output is edges.  Idempotent: existing shared_referent edges for a memory are
deleted before its new ones are written.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import stable_id  # noqa: E402

RELATION = "shared_referent"
WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")
STOPWORDS = frozenset("""
the and for that with you your have has had was were are its this those these they them their
but not can could would should about from what when where which who how all any some more most
just like really think know get got make made time day week month year thing things want need
going been being also very much many other into out over than then there here our ours mine his
her hers she him they'll i'm it's don't didn't that's there's what's let's yeah okay sure thanks
""".split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--df-share", type=float, default=0.05,
                        help="a term is a referent when it appears in <= this share of sessions")
    parser.add_argument("--min-shared", type=int, default=2,
                        help="minimum shared referent terms for an edge")
    parser.add_argument("--max-degree", type=int, default=6,
                        help="cross-session partners kept per scene, best score first")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scene_terms(con: sqlite3.Connection, memory_id: str):
    """scene_id -> (session_id, rare-term set), plus the memory's referent vocabulary."""
    turn_text, turn_session = {}, {}
    for tid, sid, text in con.execute(
            "select turn_id,session_id,raw_text from source_turns where memory_id=?", (memory_id,)):
        turn_text[tid] = text or ""
        turn_session[tid] = sid

    # scene -> turns, through the evidence_group_ref nodes it contains.
    group_turns = defaultdict(set)
    for gid, tid in con.execute(
            """select m.evidence_group_id, m.turn_id from evidence_members m
               join source_turns t on t.turn_id=m.turn_id where t.memory_id=?""", (memory_id,)):
        group_turns[gid].add(tid)
    ref_group = {}
    for nid, gid in con.execute(
            """select node_id,evidence_group_id from graph_nodes
               where memory_id=? and node_type='evidence_group_ref'""", (memory_id,)):
        ref_group[nid] = gid

    scene_turns = defaultdict(set)
    for src, dst in con.execute(
            """select src_id,dst_id from graph_edges
               where memory_id=? and relation='scene_contains'""", (memory_id,)):
        gid = ref_group.get(dst)
        if gid:
            scene_turns[src] |= group_turns.get(gid, set())

    scene_session = {}
    for nid, attrs in con.execute(
            """select node_id,attributes_json from graph_nodes
               where memory_id=? and node_type='scene'""", (memory_id,)):
        d = json.loads(attrs) if attrs else {}
        if d.get("session_id"):
            scene_session[nid] = d["session_id"]

    # Document frequency is counted over sessions, not turns: a term repeated
    # inside one session is still one session's worth of evidence.
    session_terms = defaultdict(set)
    for tid, text in turn_text.items():
        session_terms[turn_session[tid]] |= {
            w.lower() for w in WORD.findall(text)} - STOPWORDS
    df = Counter()
    for terms in session_terms.values():
        df.update(terms)
    return scene_turns, scene_session, turn_text, df, len(session_terms)


def main() -> None:
    args = parse_args()
    con = sqlite3.connect(args.db)
    memories = [r[0] for r in con.execute("select memory_id from conversations")]
    written = 0
    for memory_id in memories:
        scene_turns, scene_session, turn_text, df, n_sessions = scene_terms(con, memory_id)
        threshold = max(2, n_sessions * args.df_share)
        rare = {w for w, c in df.items() if c <= threshold}
        idf = {w: math.log(n_sessions / max(1, df[w])) for w in rare}

        scenes = []
        for sid, session in scene_session.items():
            terms = set()
            for tid in scene_turns.get(sid, ()):
                terms |= {w.lower() for w in WORD.findall(turn_text.get(tid, ""))}
            terms &= rare
            if terms:
                scenes.append((sid, session, terms))

        # Inverted index over referent terms, so pairing is over co-occurring
        # scenes rather than the full O(n^2) scene product.
        postings = defaultdict(list)
        for i, (_, _, terms) in enumerate(scenes):
            for w in terms:
                postings[w].append(i)
        pair_score = defaultdict(float)
        pair_count = Counter()
        for w, members in postings.items():
            if len(members) > 64:      # a term in this many scenes is not a referent
                continue
            weight = idf[w]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    if scenes[i][1] == scenes[j][1]:
                        continue
                    pair_score[(i, j)] += weight
                    pair_count[(i, j)] += 1

        best = defaultdict(list)
        for (i, j), count in pair_count.items():
            if count < args.min_shared:
                continue
            best[i].append((pair_score[(i, j)], j))
            best[j].append((pair_score[(i, j)], i))
        keep = set()
        for i, partners in best.items():
            for _, j in sorted(partners, reverse=True)[:args.max_degree]:
                keep.add((min(i, j), max(i, j)))

        print(f"  {memory_id[:26]:28} sessions={n_sessions:3d} scenes={len(scenes):4d} "
              f"referent_terms={len(rare):6d} pairs={len(keep):5d}")
        if args.dry_run:
            written += len(keep) * 2
            continue

        con.execute("delete from graph_edges where memory_id=? and relation=?",
                    (memory_id, RELATION))
        rows = []
        for i, j in keep:
            for src, dst in ((scenes[i][0], scenes[j][0]), (scenes[j][0], scenes[i][0])):
                eid = stable_id("edge", memory_id, RELATION, src, dst)
                rows.append((eid, memory_id, src, RELATION, dst,
                             f"evidence:{RELATION}", json.dumps([]), 1,
                             min(1.0, pair_score[(i, j)] / 20.0), "shared_referent_v1",
                             "graphmem-v5"))
        con.executemany("insert or replace into graph_edges values (?,?,?,?,?,?,?,?,?,?,?)", rows)
        written += len(rows)
    if not args.dry_run:
        con.commit()
    print(f"\n{RELATION} edges {'(dry run) ' if args.dry_run else ''}= {written}")
    con.close()


if __name__ == "__main__":
    main()
