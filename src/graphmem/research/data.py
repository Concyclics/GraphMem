"""Separate public inference inputs from benchmark annotations at ingestion."""

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import shutil
import sqlite3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def snapshot(source, target):
    if Path(source).resolve() == Path(target).resolve():
        return
    with sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)


def prepare(root, benchmark, data, graph_db, tokenizer):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark": benchmark,
        "data_sha256": sha(data),
        "graph_sha256": sha(graph_db),
        "tokenizer_sha256": sha(tokenizer),
    }
    marker = root / "inputs.json"
    if marker.exists():
        if json.loads(marker.read_text()) != manifest:
            raise ValueError("Inputs changed; use a new output directory")
        return
    target = root / "input"
    target.mkdir(exist_ok=True)
    snapshot(graph_db, target / "graphmem.sqlite")
    shutil.copyfile(tokenizer, target / "tokenizer.json")
    c = sqlite3.connect(f"file:{target.resolve()}/graphmem.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    source = json.loads(Path(data).read_text())
    queries, labels, metadata, turns = [], [], [], {}
    if benchmark == "locomo":
        for ci, conv in enumerate(source):
            mid = "locomo:" + conv["sample_id"]
            ts = [
                dict(r)
                for r in c.execute(
                    "select * from source_turns where memory_id=? order by session_id,turn_index",
                    (mid,),
                )
            ]
            if not ts:
                raise ValueError(f"No source turns for {mid}")
            mapping = {}
            for t in ts:
                original = conv["conversation"][t["session_id"]][t["turn_index"]]
                if not t["raw_text"].strip().startswith(original["text"].strip()):
                    raise ValueError("Source text differs from raw dataset")
                mapping[original["dia_id"]] = t["turn_id"]
            turns[mid] = ts
            for qi, qa in enumerate(conv["qa"]):
                if qa["category"] == 5:
                    continue
                qid = f"locomo{ci:02d}_{qi:04d}"
                queries.append(dict(qid=qid, query=qa["question"], memory_id=mid))
                text = "; ".join(qa.get("evidence", []))
                gold = {
                    f"D{int(a)}:{int(b)}"
                    for a, b in re.findall(r"D:?(\d+):(\d+)", text)
                }
                unknown = sorted(gold - mapping.keys())
                ambiguous = bool(re.search(r"\w", re.sub(r"D:?(\d+):(\d+)", "", text)))
                labels.append(
                    dict(
                        qid=qid,
                        gold_turn_ids=[
                            mapping[g] for g in sorted(gold) if g in mapping
                        ],
                        valid=bool(gold) and not unknown and not ambiguous,
                        abstention=False,
                        gold_session_ids=sorted(
                            {
                                "session_" + str(int(a))
                                for a in re.findall(r"D:?(\d+):", text)
                            },
                            key=lambda x: int(x.split("_")[-1]),
                        ),
                        issues=dict(
                            empty=not gold, unknown=unknown, ambiguous=ambiguous
                        ),
                    )
                )
                metadata.append(
                    dict(
                        qid=qid,
                        public_qid=f"conv{ci}_q{qi}",
                        question=qa["question"],
                        reference=str(qa.get("answer", "")),
                        category=qa["category"],
                        question_type=str(qa["category"]),
                    )
                )
    else:
        byid = {str(r["question_id"]): r for r in source}
        memories = {
            r["memory_id"]: r["source_id"]
            for r in c.execute(
                "select * from conversations where dataset='longmemeval'"
            )
        }
        if set(memories.values()) != set(byid):
            raise ValueError(
                "Graph does not cover exactly the requested LongMemEval histories"
            )
        for mid, qid in memories.items():
            row = byid[qid]
            raw = dict(zip(row["haystack_session_ids"], row["haystack_sessions"]))
            ordinals = {
                s["session_id"]: s["ordinal"] + 1
                for s in c.execute("select * from sessions where memory_id=?", (mid,))
            }
            ts, gold = [], []
            for rec in c.execute(
                "select * from source_turns where memory_id=? order by session_id,turn_index",
                (mid,),
            ):
                t = dict(rec)
                original = raw[t["session_id"]][t["turn_index"]]
                if not t["raw_text"].strip().startswith(original["content"].strip()):
                    raise ValueError("Source text differs from raw dataset")
                t["display_session"] = ordinals[t["session_id"]]
                ts.append(t)
                if original.get("has_answer"):
                    gold.append(t["turn_id"])
            turns[mid] = ts
            queries.append(
                dict(
                    qid=qid,
                    query=row["question"],
                    question_date=row["question_date"],
                    memory_id=mid,
                )
            )
            labels.append(
                dict(
                    qid=qid,
                    gold_turn_ids=gold,
                    gold_session_ids=row["answer_session_ids"],
                    valid=bool(gold),
                    abstention=qid.endswith("_abs"),
                )
            )
            metadata.append(
                dict(
                    qid=qid,
                    public_qid=qid,
                    question=row["question"],
                    reference=str(row["answer"]),
                    question_type=row["question_type"],
                    question_date=row["question_date"],
                    abstention=qid.endswith("_abs"),
                )
            )
    c.close()
    if len({q["qid"] for q in queries}) != len(queries):
        raise ValueError("Duplicate question IDs")
    write_rows(target / "queries.jsonl", queries)
    (target / "turns.json").write_text(json.dumps(turns, ensure_ascii=False))
    # The answer pipeline never opens this directory.
    evaluation = root / "evaluation"
    evaluation.mkdir(exist_ok=True)
    write_rows(evaluation / "labels.jsonl", labels)
    write_rows(evaluation / "metadata.jsonl", metadata)
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
