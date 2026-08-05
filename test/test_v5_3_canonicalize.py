from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from graphmem.build import PredicateCanonicalizer
from graphmem.config import GraphMemV5Config

from test_v5_gate_b_core import _store


class FakeEmbeddings:
    def create(self, *, model, input):
        vectors = {"live in": [1.0, 0.0], "lives in": [0.999, 0.001],
                   "work at": [0.0, 1.0]}
        data = [SimpleNamespace(index=index, embedding=vectors[text])
                for index, text in enumerate(input)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(input)))


class FlakyEmbeddings(FakeEmbeddings):
    def __init__(self):
        self.calls = 0

    def create(self, *, model, input):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("embedding service restarted")
        return super().create(model=model, input=input)


def test_predicate_clustering_requires_mutual_nearest_and_compatible_slots(tmp_path: Path) -> None:
    store = _store(tmp_path / "canonical.sqlite")
    base = GraphMemV5Config(); config = replace(base, edges=replace(
        base.edges, predicate_embedding_threshold=0.92))
    client = SimpleNamespace(embeddings=FakeEmbeddings())
    canonicalizer = PredicateCanonicalizer(store, config, client=client)
    keys = [
        ("alice", "live in", "home", "place", "positive"),
        ("alice", "lives in", "home", "place", "positive"),
        ("alice", "work at", "home", "place", "positive"),
        ("bob", "lives in", "home", "place", "positive"),
    ]
    rows = canonicalizer.canonicalize("travel", keys)
    assert rows[keys[0]] == rows[keys[1]] == "live in"
    assert rows[keys[2]] == "work at"
    assert rows[keys[3]] == "lives in"
    before = store._connection.execute(
        "SELECT count(*) FROM embedding_calls WHERE model_id LIKE '%predicate-v1'"
    ).fetchone()[0]
    canonicalizer.canonicalize("travel", keys)
    after = store._connection.execute(
        "SELECT count(*) FROM embedding_calls WHERE model_id LIKE '%predicate-v1'"
    ).fetchone()[0]
    assert before == after == 1


def test_predicate_embedding_retries_transient_service_restart(tmp_path: Path) -> None:
    store = _store(tmp_path / "canonical-retry.sqlite")
    embeddings = FlakyEmbeddings()
    canonicalizer = PredicateCanonicalizer(
        store, GraphMemV5Config(), client=SimpleNamespace(embeddings=embeddings))
    key = ("alice", "live in", "home", "place", "positive")
    assert canonicalizer.canonicalize("travel", [key])[key] == "live in"
    assert embeddings.calls == 2
