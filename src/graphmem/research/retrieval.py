"""Frozen CPU ranking and label-free batch token allocation (no fitting)."""

from pathlib import Path
from collections import defaultdict
import json
import statistics
import numpy as np
from tokenizers import Tokenizer
from .features import norm, rank_feature

ASSETS = Path(__file__).parent / "assets"


class Corpus:
    def __init__(self, root, benchmark):
        R = Path(root)
        self.root = R
        self.benchmark = benchmark
        self.queries = {
            q["qid"]: q for q in map(json.loads, (R / "input/queries.jsonl").open())
        }
        self.turns = json.load(open(R / "input/turns.json"))
        self.tokenizer = Tokenizer.from_file(str(R / "input/tokenizer.json"))
        self.names = json.load(open(R / "features_qwen/feature_names.json"))
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
        return np.load(self.root / "features_qwen" / (qid + ".npz"))["X"]

    def col(self, X, name):
        return X[:, self.names.index(name)]


class FastAllocator:
    def __init__(self, c, scores):
        self.c = c
        self.scores = scores
        self.prep = {}
        for q, s in scores.items():
            order = np.argsort(-s, kind="stable")
            ts = c.turns[c.queries[q]["memory_id"]]
            costs, _, hc = c.cache[c.queries[q]["memory_id"]]
            used = c.count("Question: " + c.queries[q]["query"] + "\nEvidence:\n") + 64
            seen = set()
            prefix = []
            kept = []
            for i in order:
                sid = ts[i]["session_id"]
                cost = int(costs[i]) + (hc[sid] if sid not in seen else 0)
                keep = used + cost <= 32000
                if keep:
                    used += cost
                    seen.add(sid)
                kept.append(keep)
                prefix.append(used)
            self.prep[q] = (order, -s[order], np.array(prefix), np.array(kept))

    def pick(self, q, threshold):
        order, neg, prefix, keep = self.prep[q]
        n = max(min(8, len(order)), int(np.searchsorted(neg, -threshold, side="right")))
        return order[:n][keep[:n]].tolist(), int(prefix[n - 1])

    def allocate(self, target, qids=None):
        qs = list(qids or self.scores)
        lo, hi = -20.0, 20.0
        for _ in range(18):
            mid = (lo + hi) / 2
            cost = statistics.mean(self.pick(q, mid)[1] for q in qs)
            if cost > target:
                lo = mid
            else:
                hi = mid
        return hi, statistics.mean(self.pick(q, hi)[1] for q in qs)


def neighbors(c, mid, radius):
    ts = c.turns[mid]
    return [
        [
            j
            for j, s in enumerate(ts)
            if s["session_id"] == t["session_id"]
            and abs(s["turn_index"] - t["turn_index"]) <= radius
        ]
        for t in ts
    ]


def adjust_scores(c, qid, base, config, neighbor_scores=None):
    q = c.queries[qid]
    costs = c.cache[q["memory_id"]][0]
    score = base.copy()
    radius = config["radius"]
    penalty = config["neighbor_penalty"]
    if radius:
        if neighbor_scores is None:
            neighbor_scores = np.array(
                [max(base[js]) for js in neighbors(c, q["memory_id"], radius)]
            )
        score = np.maximum(score, neighbor_scores - penalty)
    X = c.features(qid) if config["date_bonus"] else None
    if config["date_bonus"]:
        score += config["date_bonus"] * np.minimum(c.col(X, "date_match"), 1)
    score -= config["cost_penalty"] * np.log(np.maximum(costs, 1))
    return score


def rank(root, benchmark):
    import xgboost as xgb

    c = Corpus(root, benchmark)
    expected = json.loads((ASSETS / "transfer_to_lme_manifest.json").read_text())[
        "features"
    ]
    if c.names != expected:
        raise ValueError("Feature order differs from the frozen 117-feature ranker")
    cfg = json.loads((ASSETS / "closure_config.json").read_text())
    models, scores = {}, {}
    for qid, q in c.queries.items():
        # Fold ID is conversation membership only, never the answer/category.
        fold = int(qid[6:8]) if benchmark == "locomo" else None
        filename = (
            f"fold_{fold}_model.json" if fold is not None else "transfer_to_lme.json"
        )
        if filename not in models:
            model = xgb.XGBRanker(n_jobs=4)
            model.load_model(ASSETS / filename)
            models[filename] = model
        base = models[filename].predict(c.features(qid))
        scores[qid] = adjust_scores(c, qid, base, cfg)
    np.savez_compressed(Path(root) / "scores.npz", **scores)
    allocator = FastAllocator(c, scores)
    target = 8050 if benchmark == "locomo" else 8700
    threshold, estimate = allocator.allocate(target)
    rows = []
    for qid, q in c.queries.items():
        idx, cost = allocator.pick(qid, threshold)
        rows.append(
            dict(
                qid=qid,
                selected_ids=[c.turns[q["memory_id"]][i]["turn_id"] for i in idx],
                tokens=c.count(c.render(qid, idx)),
                conservative_tokens=cost,
            )
        )
    (Path(root) / "retrieval.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (Path(root) / "allocation.json").write_text(
        json.dumps(
            dict(
                threshold=threshold,
                conservative_mean=estimate,
                target=target,
                cap_per_query=32000,
                minimum_turns=8,
                labels_used=False,
                training=False,
            ),
            indent=2,
        )
        + "\n"
    )
    return rows
