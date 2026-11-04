"""Label-free graph feature exporter. Never reads benchmark labels/answers."""

from pathlib import Path
import json, sqlite3, hashlib, re, collections, time
import numpy as np
from tokenizers import Tokenizer

STOP = set(
    "a an the is are was were be been being do does did have has had would could should will can may might what which who whom whose when where why how and or of to in on at by for from with as that this these those it its i me my you your we our they their he she his her both about during after before".split()
)


def terms(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 4 and w.endswith("ed"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


class Lexical:
    def __init__(self, texts):
        self.n = len(texts)
        self.postings = collections.defaultdict(list)
        self.length = np.array([len(terms(t)) for t in texts], dtype=float)
        self.avg = max(1, self.length.mean())
        for i, t in enumerate(texts):
            for w, n in collections.Counter(terms(t)).items():
                self.postings[w].append((i, n))

    def score(self, q):
        score = np.zeros(self.n)
        for w in set(terms(q)):
            pairs = self.postings.get(w, [])
            if not pairs:
                continue
            idx = np.array([i for i, n in pairs])
            tf = np.array([n for i, n in pairs])
            idf = np.log(1 + (self.n - len(pairs) + 0.5) / (len(pairs) + 0.5))
            score[idx] += (
                idf
                * tf
                * 2.2
                / (tf + 1.2 * (0.25 + 0.75 * self.length[idx] / self.avg))
            )
        return score


def norm(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def rank_feature(x):
    x = np.asarray(x)
    rank = len(x) - np.searchsorted(np.sort(x), x, side="right")
    return np.where(x > 0, 1 / (1 + rank / 20), 0)


def project(values, ni, ti, n):
    out = np.zeros(n)
    np.maximum.at(out, ti, values[ni])
    return out


def export(root, base_model, *, longmemeval=False):
    R = Path(root)
    c = sqlite3.connect(f"file:{R}/input/graphmem.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    qc = sqlite3.connect(f"file:{R}/input/query_embeddings.sqlite?mode=ro", uri=True)
    tokenizer = Tokenizer.from_file(str(R / "input/tokenizer.json"))
    queries = [json.loads(l) for l in (R / "input/queries.jsonl").open()]
    allturns = json.load(open(R / "input/turns.json"))
    out = R / "features"
    out.mkdir(exist_ok=True)
    names = None
    for mid, turns in allturns.items():
        start = time.time()
        qs = [q for q in queries if q["memory_id"] == mid]
        n = len(turns)
        ids = {t["turn_id"]: i for i, t in enumerate(turns)}
        nodes = [
            dict(r)
            for r in c.execute("select * from graph_nodes where memory_id=?", (mid,))
        ]
        nodeids = {r["node_id"]: i for i, r in enumerate(nodes)}
        attrs = [json.loads(r["attributes_json"]) for r in nodes]
        groups = collections.defaultdict(set)
        for row in c.execute(
            "select em.evidence_group_id,em.turn_id from evidence_members em join source_turns t on em.turn_id=t.turn_id where t.memory_id=?",
            (mid,),
        ):
            groups[row[0]].add(ids[row[1]])
        supports = []
        for node, a in zip(nodes, attrs):
            support = set()
            for gid in [node["evidence_group_id"]] + json.loads(
                node["evidence_group_ids_json"]
            ):
                support.update(groups[gid])
            for tid in (
                a.get("turn_ids", [])
                + a.get("raw_fallback_turn_ids", [])
                + [s["turn_id"] for s in a.get("evidence_spans", [])]
            ):
                if tid in ids:
                    support.add(ids[tid])
            supports.append(support)
        ni = np.array([i for i, s in enumerate(supports) for t in s], dtype=int)
        ti = np.array([t for s in supports for t in s], dtype=int)
        edges = collections.defaultdict(list)
        for edge in c.execute(
            "select src_id,relation,dst_id from graph_edges where memory_id=?", (mid,)
        ):
            if edge[0] in nodeids and edge[2] in nodeids:
                edges[edge[1]].append((nodeids[edge[0]], nodeids[edge[2]]))
        vectors = {
            r["item_id"]: np.frombuffer(r["vector"], dtype=np.float32)
            for r in c.execute(
                "select item_id,vector from embeddings where memory_id=? and model_id=?",
                (mid, base_model),
            )
        }
        emb = np.stack([vectors[t["turn_id"]] for t in turns])
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        turnlex = Lexical([t["speaker"] + " " + t["raw_text"] for t in turns])
        nodelex = Lexical([r["summary"] for r in nodes])
        attrlex = Lexical(
            [
                " ".join(
                    str(a.get(k, ""))
                    for k in [
                        "entities",
                        "entity_names",
                        "predicate",
                        "scope",
                        "value",
                        "aliases",
                    ]
                )
                for a in attrs
            ]
        )
        bysession = collections.defaultdict(list)
        for i, t in enumerate(turns):
            bysession[t["session_id"]].append(i)
        neighbors = []
        for i, t in enumerate(turns):
            neighbors.append(
                [
                    j
                    for j in bysession[t["session_id"]]
                    if abs(turns[j]["turn_index"] - t["turn_index"]) <= 2
                ]
            )
        rendered = [
            f"[D{t.get('display_session', t['session_id'].split('_')[-1])}:{t['turn_index'] + 1}] {t['speaker']}: {t['raw_text']}\n"
            for t in turns
        ]
        # Compact original dialogue IDs; session date headers are charged separately by the packer.
        costs = np.array(
            [len(tokenizer.encode(s, add_special_tokens=False).ids) for s in rendered]
        )
        np.savez_compressed(
            out / (mid.replace(":", "_") + "_turns.npz"),
            costs=costs,
            ids=np.array([t["turn_id"] for t in turns]),
        )
        for qi, q in enumerate(qs):
            query = q["query"]
            text = (
                "Instruct: Retrieve conversation turns that provide all evidence needed to answer the memory question.\nQuery: "
                + query
            )
            key = hashlib.sha256(
                (base_model + "\ngraphmem-turn-evidence-v1\n" + text).encode()
            ).hexdigest()
            row = qc.execute(
                "select vector from query_embeddings where cache_key=?", (key,)
            ).fetchone()
            if row is None:
                raise RuntimeError("query cache miss: " + q["qid"])
            vec = np.frombuffer(row[0], dtype=np.float32)
            dense = emb @ vec
            lex = turnlex.score(query)
            nl = nodelex.score(query)
            al = attrlex.score(query)
            qterms = set(terms(query))
            named = [
                s for s in set(t["speaker"] for t in turns) if set(terms(s)) <= qterms
            ]
            named = (
                (
                    named
                    or (
                        ["assistant"]
                        if re.search(
                            r"\b(?:you|assistant).{0,25}\b(?:suggest|recommend|provide|tell|mention|advis)",
                            query.lower(),
                        )
                        else ["user"]
                        if re.search(r"\b(?:i|my|me|mine)\b", query.lower())
                        else []
                    )
                )
                if longmemeval
                else named
            )
            owner = np.array([float(t["speaker"] in named) for t in turns])
            date_terms = set(
                re.findall(
                    r"\b(?:20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b",
                    query.lower(),
                )
            )
            fs = {
                "dense": dense,
                "lex": lex,
                "speaker_match": owner,
                "query_has_speaker": np.full(n, bool(named)),
                "temporal_query": np.full(
                    n,
                    bool(
                        re.search(
                            r"\b(when|before|after|first|last|how long|date|year|month|week)\b",
                            query.lower(),
                        )
                    ),
                ),
                "collection_query": np.full(
                    n,
                    bool(
                        re.search(
                            r"\b(both|all|list|how many|which activities|what activities)\b",
                            query.lower(),
                        )
                    ),
                ),
                "date_match": np.array(
                    [
                        len(
                            date_terms
                            & set(re.findall(r"\w+", str(t["timestamp"]).lower()))
                        )
                        for t in turns
                    ]
                ),
                "neighbor_dense": np.array(
                    [max(dense[j] for j in v) for v in neighbors]
                ),
                "neighbor_lex": np.array([max(lex[j] for j in v) for v in neighbors]),
            }
            node_dense = np.array(
                [max((dense[j] for j in s), default=0) for s in supports]
            )
            node_seed = 0.6 * norm(nl) + 0.4 * norm(al)
            for typ in [
                "canonical_fact",
                "scene",
                "event_skeleton",
                "routing_card",
                "canonical_entity",
                "time_anchor",
            ]:
                mask = np.array([r["node_type"] == typ for r in nodes])
                fs[typ + "_lex"] = project(nl * mask, ni, ti, n)
                fs[typ + "_attr"] = project(al * mask, ni, ti, n)
                fs[typ + "_dense"] = project(node_dense * mask, ni, ti, n)
            for relation in [
                "coarse_related",
                "participates_in",
                "at_time",
                "state_next",
                "temporal_before",
                "scene_contains",
                "coreference",
                "shared_value",
            ]:
                pairs = edges.get(relation, [])
                v = np.zeros(len(nodes))
                if pairs:
                    src, dst = np.array(pairs).T
                    degree = np.bincount(
                        np.concatenate([src, dst]), minlength=len(nodes)
                    )
                    np.maximum.at(v, dst, node_seed[src] / np.sqrt(1 + degree[src]))
                    np.maximum.at(v, src, node_seed[dst] / np.sqrt(1 + degree[dst]))
                fs["edge_" + relation] = project(v, ni, ti, n)
            scalar_names = list(fs)
            for name in scalar_names:
                if name not in [
                    "query_has_speaker",
                    "temporal_query",
                    "collection_query",
                    "speaker_match",
                ]:
                    fs[name + "_rank"] = rank_feature(fs[name])
                    fs[name + "_norm"] = norm(fs[name])
            for name in [
                "dense_norm",
                "lex_norm",
                "canonical_fact_lex_norm",
                "scene_lex_norm",
            ]:
                fs[name + "_owner"] = fs[name] * owner
            if names is None:
                names = list(fs)
                (out / "feature_names.json").write_text(json.dumps(names, indent=2))
            assert names == list(fs)
            features = np.stack([fs[name] for name in names], axis=1).astype(np.float32)
            np.savez_compressed(out / (q["qid"] + ".npz"), X=features)
        print(
            "FEATURES",
            mid,
            len(qs),
            "turns",
            n,
            "nodes",
            len(nodes),
            "seconds",
            round(time.time() - start, 2),
            flush=True,
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "queries": len(queries),
                "features": names,
                "label_access": False,
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
