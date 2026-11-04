from pathlib import Path
import json, hashlib
import numpy as np
from .features import norm, rank_feature
from .retrieval import neighbors
from tokenizers import Tokenizer
from collections import defaultdict


class BaseCorpus:
    def __init__(self, root, benchmark):
        R = Path(root)
        self.root = R
        self.benchmark = benchmark
        self.queries = {
            q["qid"]: q for q in map(json.loads, (R / "input/queries.jsonl").open())
        }
        self.turns = json.load(open(R / "input/turns.json"))
        self.tokenizer = Tokenizer.from_file(str(R / "input/tokenizer.json"))
        self.names = json.load(open(R / "features/feature_names.json"))
        self.cache = {}
        for mid, turns in self.turns.items():
            paths = R / "features" / (mid.replace(":", "_") + "_turns.npz")
            npz = np.load(paths)
            costs = npz["costs"]
            assert list(npz["ids"]) == [t["turn_id"] for t in turns]
            headers = {
                t[
                    "session_id"
                ]: f"Session {t.get('display_session', t['session_id'].split('_')[-1])} — {t['timestamp']}\n"
                for t in turns
            }
            if benchmark == "locomo":
                costs = np.array(
                    [
                        self.count(
                            f"[{t['turn_index'] + 1}] {t['speaker']}: {t['raw_text']}\n"
                        )
                        for t in turns
                    ]
                )
            hc = {k: self.count(v) for k, v in headers.items()}
            self.cache[mid] = (costs, headers, hc)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def render(self, qid, selected):
        q = self.queries[qid]
        mid = q["memory_id"]
        ts = self.turns[mid]
        _, headers, _ = self.cache[mid]
        sessions = defaultdict(list)
        for i in selected:
            sessions[ts[i]["session_id"]].append(i)
        chunks = []
        for sid in sorted(
            sessions,
            key=lambda s: int(
                ts[sessions[s][0]].get("display_session", s.split("_")[-1])
            ),
        ):
            chunks.append(headers[sid])
            chunks.extend(
                f"[{ts[i]['turn_index'] + 1}] {ts[i]['speaker']}: {ts[i]['raw_text']}\n"
                for i in sorted(sessions[sid], key=lambda i: ts[i]["turn_index"])
            )
        return (
            "Question: "
            + q["query"]
            + (
                "\nQuestion date: " + q["question_date"]
                if self.benchmark == "longmemeval"
                else ""
            )
            + "\nEvidence:\n"
            + "".join(chunks)
        )

    def pack(self, qid, scores, budget=9900):
        q = self.queries[qid]
        mid = q["memory_id"]
        ts = self.turns[mid]
        costs, headers, hc = self.cache[mid]
        order = np.argsort(-np.asarray(scores), kind="stable")
        used = self.count("Question: " + q["query"] + "\nEvidence:\n") + 64
        selected = []
        seen = set()
        for i in order:
            sid = ts[i]["session_id"]
            cost = int(costs[i]) + (hc[sid] if sid not in seen else 0)
            if used + cost > budget:
                continue
            selected.append(int(i))
            used += cost
            seen.add(sid)
        actual = self.count(self.render(qid, selected))
        while actual > budget:
            selected.pop()
            actual = self.count(self.render(qid, selected))
        return selected, actual

    def features(self, qid):
        return np.load(self.root / "features" / (qid + ".npz"))["X"]

    def col(self, X, name):
        return X[:, self.names.index(name)]


def export(root, benchmark):
    R = Path(root)
    c = BaseCorpus(root, benchmark)
    root = R / "qwen_embeddings"
    manifest = json.load(open(root / "manifest.json"))
    vectors = {}
    for p in sorted(root.glob("batch_*.npz")):
        with np.load(p) as f:
            for key, vec in zip(f["ids"], f["vectors"]):
                v = vec.astype(np.float32)
                v /= max(1e-9, np.linalg.norm(v))
                vectors[str(key)] = v
    assert len(vectors) == manifest["jobs"]
    out = R / "features_qwen"
    out.mkdir(exist_ok=True)
    names = None
    for mid, ts in c.turns.items():
        raw = np.stack([vectors["raw:" + t["turn_id"]] for t in ts])
        graph = np.stack([vectors["graph:" + t["turn_id"]] for t in ts])
        nn = neighbors(c, mid, 2)
        for qid, q in c.queries.items():
            if q["memory_id"] != mid:
                continue
            vec = vectors["query:" + qid]
            r = raw @ vec
            g = graph @ vec
            owner = c.col(c.features(qid), "speaker_match")
            fs = {
                "qwen_raw": r,
                "qwen_graph_context": g,
                "qwen_neighbor_raw": np.array([max(r[js]) for js in nn]),
                "qwen_neighbor_graph": np.array([max(g[js]) for js in nn]),
            }
            for key in list(fs):
                fs[key + "_norm"] = norm(fs[key])
                fs[key + "_rank"] = rank_feature(fs[key])
                fs[key + "_owner"] = norm(fs[key]) * owner
            if names is None:
                names = c.names + list(fs)
                (out / "feature_names.json").write_text(json.dumps(names, indent=2))
            X = np.column_stack([c.features(qid)] + [v for v in fs.values()]).astype(
                np.float32
            )
            np.savez_compressed(out / (qid + ".npz"), X=X)
        print("ENHANCED", mid, flush=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "features": names,
                "labels_read": False,
                "embedding_manifest": manifest,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
