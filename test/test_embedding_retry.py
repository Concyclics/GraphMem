from types import SimpleNamespace

import numpy as np

from graphmem.embedding import QwenEmbeddingIndex


class APIConnectionError(Exception):
    pass


class _Embeddings:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise APIConnectionError("supervised engine is restarting")
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])],
            usage=SimpleNamespace(prompt_tokens=3),
        )


def test_embedding_retries_a_transient_supervisor_restart(monkeypatch):
    monkeypatch.setattr("graphmem.embedding.time.sleep", lambda _seconds: None)
    index = object.__new__(QwenEmbeddingIndex)
    index.model_id = "embedding-test"
    index.client = SimpleNamespace(embeddings=_Embeddings(failures=2))

    vectors, tokens, _latency = index._embed(["hello"])

    assert vectors == [[1.0, 2.0]]
    assert tokens == 3
    assert index.client.embeddings.calls == 3


def test_embedding_does_not_retry_nonrecoverable_errors(monkeypatch):
    monkeypatch.setattr("graphmem.embedding.time.sleep", lambda _seconds: None)
    index = object.__new__(QwenEmbeddingIndex)
    index.model_id = "embedding-test"
    index.client = SimpleNamespace(embeddings=_Embeddings(failures=20))
    index.client.embeddings.create = lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad input"))

    try:
        index._embed(["hello"])
    except ValueError as error:
        assert str(error) == "bad input"
    else:  # pragma: no cover
        raise AssertionError("nonrecoverable errors must be raised immediately")


def test_embedding_splits_a_batch_that_exceeds_endpoint_context():
    calls = []

    def create(*, model, input):
        calls.append((model, tuple(input)))
        if len(input) > 2:
            error = RuntimeError("maximum context length is 8192 tokens")
            error.status_code = 400
            raise error
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[float(index), 1.0])
                  for index, _text in enumerate(input)],
            usage=SimpleNamespace(prompt_tokens=len(input)),
        )

    index = object.__new__(QwenEmbeddingIndex)
    index.model_id = "embedding-storage-id"
    index.request_model_id = "embedding-served-alias"
    index.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create))

    vectors, tokens, _latency = index._embed(["a", "b", "c", "d"])

    assert len(vectors) == 4
    assert tokens == 4
    assert calls == [
        ("embedding-served-alias", ("a", "b", "c", "d")),
        ("embedding-served-alias", ("a", "b")),
        ("embedding-served-alias", ("c", "d")),
    ]


def test_embedding_chunks_one_oversized_turn_and_aggregates_one_vector():
    calls = []

    def create(*, model, input):
        calls.append((model, tuple(input)))
        if len(input[0]) > 4:
            error = RuntimeError("maximum context length is 8192 tokens")
            error.status_code = 400
            raise error
        vector = [1.0, 0.0] if input[0].startswith("a") else [0.0, 1.0]
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=vector)],
            usage=SimpleNamespace(prompt_tokens=len(input[0])),
        )

    index = object.__new__(QwenEmbeddingIndex)
    index.model_id = "embedding-storage-id"
    index.request_model_id = "embedding-served-alias"
    index.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create))

    vectors, tokens, _latency = index._embed(["aaaabbbb"])

    assert len(vectors) == 1
    assert np.allclose(vectors[0], [2 ** -0.5, 2 ** -0.5])
    assert tokens == 8
    assert calls == [
        ("embedding-served-alias", ("aaaabbbb",)),
        ("embedding-served-alias", ("aaaa",)),
        ("embedding-served-alias", ("bbbb",)),
    ]
