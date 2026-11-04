"""Original raw and graph-context embedding views, with deterministic ordering."""

from pathlib import Path
import sqlite3, json, collections


def jobs(root, benchmark):
    R = Path(root)
    c = sqlite3.connect(f"file:{R}/input/graphmem.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    allturns = json.load(open(R / "input/turns.json"))
    queries = list(map(json.loads, (R / "input/queries.jsonl").open()))
    rows = []
    for mid, ts in allturns.items():
        byid = {t["turn_id"]: t for t in ts}
        g = collections.defaultdict(set)
        for r in c.execute(
            "select em.evidence_group_id,em.turn_id from evidence_members em join source_turns t on t.turn_id=em.turn_id where t.memory_id=?",
            (mid,),
        ):
            g[r[0]].add(r[1])
        views = collections.defaultdict(list)
        for r in c.execute(
            "select * from graph_nodes where memory_id=? and node_type in ('scene','canonical_fact','event_skeleton')",
            (mid,),
        ):
            a = json.loads(r["attributes_json"])
            support = (
                set(a.get("turn_ids", []))
                | set(a.get("raw_fallback_turn_ids", []))
                | {v["turn_id"] for v in a.get("evidence_spans", [])}
            )
            for gid in [r["evidence_group_id"]] + json.loads(
                r["evidence_group_ids_json"]
            ):
                support.update(g[gid])
            desc = r["node_type"] + ": " + r["summary"]
            if r["event_time"]:
                desc += "; Event time: " + r["event_time"]
            entities = a.get("entities", a.get("entity_names", []))
            if entities:
                desc += "; Entities: " + ", ".join(entities)
            for tid in support:
                if tid in byid:
                    views[tid].append((len(support), desc))
        for t in ts:
            raw = f"Session time: {t['timestamp']}\n{t['speaker']} speaking to {t['listener']}: {t['raw_text']}"
            rows.append(
                dict(id="raw:" + t["turn_id"], memory_id=mid, text=raw, kind="raw")
            )
            nearby = sorted(
                [
                    s
                    for s in ts
                    if s["session_id"] == t["session_id"]
                    and abs(s["turn_index"] - t["turn_index"]) <= 2
                ],
                key=lambda s: s["turn_index"],
            )
            context = "\n".join(
                f"{s['speaker']}: {s['raw_text']}"
                for s in nearby
                if s["turn_id"] != t["turn_id"]
            )
            attrs = "\n".join(
                dict.fromkeys(v for _, v in sorted(views[t["turn_id"]])[:4])
            )[:1800]
            enriched = (
                raw
                + "\nRelated scene and event attributes:\n"
                + attrs
                + "\nSurrounding conversation:\n"
                + context
            )
            rows.append(
                dict(
                    id="graph:" + t["turn_id"],
                    memory_id=mid,
                    text=enriched,
                    kind="graph",
                )
            )
    for q in queries:
        text = (
            "Instruct: Retrieve conversation evidence that answers the question, matching the correct people, event, situation, and time. Include indirect supporting statements and necessary context.\nQuery: "
            + q["query"]
            + (
                "\nQuestion date: " + q.get("question_date", "")
                if benchmark == "longmemeval"
                else ""
            )
        )
        rows.append(
            dict(
                id="query:" + q["qid"],
                memory_id=q["memory_id"],
                text=text,
                kind="query",
            )
        )
    return rows
