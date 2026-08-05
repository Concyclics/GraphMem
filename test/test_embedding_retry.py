from types import SimpleNamespace

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
