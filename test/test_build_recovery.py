from __future__ import annotations

from graphmem.build import reset_unpublished_llm_attempts
from graphmem.storage import SQLiteGraphStore


def _call(store: SQLiteGraphStore, memory_id: str, key: str,
          *, cached: bool = False) -> None:
    usage = {
        "cached_input_tokens": 17 if cached else 0,
        "uncached_input_tokens": 0 if cached else 100,
        "output_tokens": 0 if cached else 25,
        "total_tokens": 17 if cached else 125,
    }
    store.cache_put(key, "scene_semantic", {"key": key}, {"content": "{}"},
                    usage, "prompt")
    store.log_llm_call(
        call_id=f"call:{memory_id}:{key}", memory_id=memory_id,
        stage="scene_semantic", cache_key=key, cached=cached,
        request={"key": key}, response={"content": "{}"}, usage=usage,
        latency_ms=1.0, retry_count=0, batch_size=1,
        prompt_hash="prompt")


def test_recovery_resets_only_unpublished_llm_attempts(tmp_path) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite")
    try:
        _call(store, "published", "published-key")
        _call(store, "partial", "partial-key")
        with store.transaction() as db:
            db.execute(
                "INSERT INTO graph_versions(memory_id,graph_version,graph_checksum) "
                "VALUES('published',1,'checksum')")

        audit = reset_unpublished_llm_attempts(store)

        assert audit == ({
            "memory_id": "partial", "calls": 1, "api_calls": 1,
            "input_tokens": 100, "cached_input_tokens": 0,
            "output_tokens": 25, "total_api_tokens": 125,
            "retry_count": 0, "reason": "unpublished_attempt_reset",
        },)
        assert [row["memory_id"] for row in store._read(
            "SELECT memory_id FROM llm_calls")] == ["published"]
        assert store.cache_get("published-key") is not None
        assert store.cache_get("partial-key") is None
    finally:
        store.close()


def test_recovery_is_idempotent(tmp_path) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite")
    try:
        _call(store, "partial", "partial-key", cached=True)
        first = reset_unpublished_llm_attempts(store)
        second = reset_unpublished_llm_attempts(store)

        assert first[0]["cached_input_tokens"] == 17
        assert first[0]["total_api_tokens"] == 0
        assert second == ()
    finally:
        store.close()
