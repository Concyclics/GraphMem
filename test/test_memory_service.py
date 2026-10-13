from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import http.client
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from graphmem import MemoryService, ServiceConfig
from graphmem.answer.optimized import prepare_optimized_readout
from graphmem.build.pipeline import GraphBuildPipeline
from graphmem.domain import canonical_json
from graphmem.service.cli import main
from graphmem.service.http import make_server
from graphmem.service.memory import MemoryConflict, MemoryNotFound, MemoryNotReady, ModelUnavailable


MESSAGES = [{"role": "user", "speaker": "Alice", "timestamp": "2026-10-13",
             "content": "Alice adopted a dog named Luna."}]
QUESTION = "What is the name of Alice's dog?"


def add(service, *, tenant="team-a", request="event-1", messages=None, session="chat-1"):
    return service.add("alice", MESSAGES if messages is None else messages, tenant_id=tenant,
                       session_id=session, request_id=request)


@pytest.fixture
def memory(tmp_path):
    with MemoryService(ServiceConfig.offline(tmp_path / "memory")) as service:
        yield service


def test_persistent_restart_preserves_graph_and_idempotency(tmp_path):
    config = ServiceConfig.offline(tmp_path / "memory")
    with MemoryService(config) as service:
        first = add(service)
        before = service.prepare("alice", QUESTION, tenant_id="team-a")
    with MemoryService(config) as service:
        replay = add(service)
        after = service.prepare("alice", QUESTION, tenant_id="team-a")
        assert replay["replayed"] and replay["turn_count"] == 1
        assert replay["graph_version"] == first["graph_version"]
        assert after.evidence_turn_ids == before.evidence_turn_ids
        assert after.prompt_payload_hash == before.prompt_payload_hash
        assert (tmp_path / "memory" / "memory.sqlite").is_file()


def test_same_id_different_payload_never_overwrites_raw(memory):
    add(memory)
    with pytest.raises(MemoryConflict, match="different payload"):
        add(memory, messages=[{"content": "Mallory changed the memory."}])
    assert memory.status("alice", tenant_id="team-a")["turn_count"] == 1
    assert "Luna" in memory.retrieve("alice", QUESTION, tenant_id="team-a")["evidence"][0]["raw_text"]


def test_same_external_ids_are_isolated_by_tenant(memory):
    first = add(memory)
    second = add(memory, tenant="team-b", messages=[{"content": "Alice has a cat called Coco."}])
    assert set(first["turn_ids"]).isdisjoint(second["turn_ids"])
    prompt = str(memory.prepare("alice", QUESTION, tenant_id="team-b").messages)
    assert "Coco" in prompt and "Luna" not in prompt
    with pytest.raises(MemoryNotFound):
        memory.retrieve("alice", QUESTION, tenant_id="team-c")


def test_concurrent_duplicate_requests_publish_once(memory):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: add(memory), range(8)))
    assert sum(not result["replayed"] for result in results) == 1
    assert {row["graph_version"] for row in results} == {1}
    assert memory.status("alice", tenant_id="team-a")["turn_count"] == 1


def test_health_does_not_wait_for_model_build(memory, monkeypatch):
    started, finish = threading.Event(), threading.Event()
    real_build = GraphBuildPipeline.build
    def waiting_build(self, *args):
        started.set()
        assert finish.wait(timeout=5)
        return real_build(self, *args)
    monkeypatch.setattr(GraphBuildPipeline, "build", waiting_build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(add, memory)
        try:
            assert started.wait(timeout=5)
            snapshot = pool.submit(memory.health).result(timeout=2)
            assert snapshot["pending_memories"] == 1
        finally:
            finish.set()
        assert writing.result(timeout=5)["status"] == "ready"


def test_new_writes_refresh_live_retrieval_and_prompt_caches(memory):
    add(memory)
    old = memory.prepare("alice", QUESTION, tenant_id="team-a")
    written = add(memory, request="event-2", messages=[{
        "role": "user", "speaker": "Alice", "content": "Alice adopted a second dog named Milo."}])
    new = memory.prepare("alice", "What are the names of Alice's dogs?", tenant_id="team-a")
    assert written["turn_count"] == 2 and written["graph_version"] == 2
    assert "Milo" in str(new.messages) and "Luna" in str(new.messages)
    assert new.prompt_payload_hash != old.prompt_payload_hash
    turns = memory.store.turns(memory._key("team-a", "alice"))
    assert [turn.turn_index for turn in turns] == [0, 1]


def test_failed_index_is_durable_and_retryable_after_restart(tmp_path, monkeypatch):
    config = ServiceConfig.offline(tmp_path / "memory")
    with MemoryService(config) as service:
        with monkeypatch.context() as patch:
            def fail(*_args):
                raise RuntimeError("simulated model outage")
            patch.setattr(GraphBuildPipeline, "build", fail)
            with pytest.raises(ModelUnavailable, match="raw messages are durable"):
                add(service)
        assert service.health()["pending_memories"] == 1
        with pytest.raises(MemoryNotReady):
            service.prepare("alice", QUESTION, tenant_id="team-a")
    with MemoryService(config) as service:
        assert service.status("alice", tenant_id="team-a")["status"] == "pending"
        recovered = add(service)
        assert recovered["replayed"] and recovered["turn_count"] == 1
        assert recovered["status"] == "ready"
        assert service.prepare("alice", QUESTION, tenant_id="team-a").evidence_turn_ids


def test_failed_explicit_rebuild_does_not_claim_old_graph_is_ready(memory, monkeypatch):
    add(memory)
    with monkeypatch.context() as patch:
        def fail(*_args):
            raise RuntimeError("simulated publication failure")
        patch.setattr(GraphBuildPipeline, "build", fail)
        with pytest.raises(ModelUnavailable):
            memory.rebuild("alice", tenant_id="team-a")
    with pytest.raises(MemoryNotReady):
        memory.retrieve("alice", QUESTION, tenant_id="team-a")
    assert memory.rebuild("alice", tenant_id="team-a")["status"] == "ready"


def test_single_writer_lease_released_on_close(tmp_path):
    config = ServiceConfig.offline(tmp_path)
    with MemoryService(config):
        with pytest.raises(MemoryConflict, match="already has"):
            MemoryService(config)
    with MemoryService(config) as service:
        add(service)


def test_backup_restores_at_new_path_without_rebuild(memory, tmp_path):
    original = add(memory)
    destination = tmp_path / "restored"
    destination.mkdir()
    memory.backup(destination / "memory.sqlite")
    with pytest.raises(FileExistsError):
        memory.backup(destination / "memory.sqlite")
    with MemoryService(ServiceConfig.offline(destination)) as restored:
        assert restored.status("alice", tenant_id="team-a")["status"] == "ready"
        assert add(restored)["graph_version"] == original["graph_version"]
        assert "Luna" in str(restored.prepare("alice", QUESTION, tenant_id="team-a").messages)


def test_build_changes_require_rebuild_query_changes_do_not(tmp_path):
    config = ServiceConfig.offline(tmp_path)
    with MemoryService(config) as service:
        add(service)
    changed_runtime = replace(config, runtime_overrides={"retrieval": {"adaptive_recall": False}})
    with MemoryService(changed_runtime) as service:
        assert service.status("alice", tenant_id="team-a")["status"] == "ready"
    changed_build = replace(config, build_overrides={"scenes": {"max_turns": 6}})
    with MemoryService(changed_build) as service:
        with pytest.raises(MemoryNotReady):
            service.retrieve("alice", QUESTION, tenant_id="team-a")
        assert service.rebuild("alice", tenant_id="team-a")["status"] == "ready"


@pytest.mark.parametrize("messages", [[], "oops", [{"content": ""}], [{"content": 7}],
                                     [{"content": "x", "unexpected": True}],
                                     [{"content": "x", "timestamp": "yesterday"}]])
def test_invalid_writes_leave_no_partial_memory(memory, messages):
    with pytest.raises(ValueError):
        add(memory, messages=messages)
    with pytest.raises(MemoryNotFound):
        memory.status("alice", tenant_id="team-a")


def test_online_prompt_preserves_graph_and_hashes_final_payload(memory):
    add(memory)
    prepared = memory.prepare("alice", QUESTION, tenant_id="team-a")
    graph = memory.retrieve("alice", QUESTION, tenant_id="team-a")
    assert set(row["turn_id"] for row in graph["evidence"]) <= set(prepared.evidence_turn_ids)
    assert "Graph witnesses" in prepared.messages[1]["content"]
    assert "Source-focus witnesses" in prepared.messages[1]["content"]
    assert prepared.trace["optimized_readout"]["uses_gold_or_judge"] is False
    assert prepared.trace["optimized_readout"]["presentation_tags"] == []
    assert prepared.prompt_payload_hash == hashlib.sha256(canonical_json(prepared.messages).encode()).hexdigest()
    assert prepared.packing_prompt_tokens <= memory.runtime.query_budget.max_answer_tokens_hard
    assert prepared.trace["base_readout"]["typed_evidence_card"] is True


def test_readout_budget_fallback_keeps_graph_evidence(memory):
    add(memory)
    key = memory._key("team-a", "alice")
    result = memory._navigate(key, QUESTION)
    prepared = memory._stage().prepare("question", QUESTION, result, memory.runtime.query_budget)
    output = prepare_optimized_readout(
        prepared, question=QUESTION, question_date=None, navigation=result,
        turns={row.turn_id: row for row in memory.store.turns(key)}, counter=memory.counter,
        max_prompt_tokens=1)
    assert output.messages == prepared.messages
    assert output.evidence_turn_ids == prepared.evidence_turn_ids
    assert output.trace["optimized_readout"]["fallback"] == "v5_73_prompt_budget"


def test_answer_uses_prepared_payload_and_durable_cache(tmp_path):
    calls = []
    def complete(**request):
        calls.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Luna", reasoning_content=None), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=1, total_tokens=101))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    config = replace(ServiceConfig.offline(tmp_path), generate_answers=True)
    with MemoryService(config, answer_client=client) as service:
        add(service)
        prepared = service.prepare("alice", QUESTION, tenant_id="team-a")
        answered = service.answer("alice", QUESTION, tenant_id="team-a")
        assert answered.prediction == "Luna"
        assert calls[0]["messages"] == list(prepared.messages)
    with MemoryService(config, answer_client=client) as service:
        assert service.answer("alice", QUESTION, tenant_id="team-a").cached
    assert len(calls) == 1


def test_semantic_dense_product_path_and_persistent_query_cache(tmp_path):
    from test_v5_1_llm_graph import FakeCompletions
    from test_v5_18_dense_query import _Embeddings
    class Embeddings(_Embeddings):
        def create(self, **request):
            result = super().create(**request)
            for row in result.data:
                row.embedding = [1.0] + [0.0] * 31
            return result
    completions, embeddings = FakeCompletions(), Embeddings()
    config = ServiceConfig(data_dir=str(tmp_path), require_exact_tokenizer=False)
    text = [{"content": "Alice visited Paris.", "speaker": "Alice"}]
    with MemoryService(config,
                       llm_client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
                       embedding_client=SimpleNamespace(embeddings=embeddings)) as service:
        add(service, messages=text)
        assert completions.calls > 0 and embeddings.calls > 0
        first = service.prepare("alice", "Where did Alice visit?", tenant_id="team-a")
        assert first.trace["optimized_readout"]["dense_focus"]
        assert "Paris" in str(first.messages)
        assert service.store.nodes(service._key("team-a", "alice"))
    second_embeddings = Embeddings()
    with MemoryService(config, embedding_client=SimpleNamespace(embeddings=second_embeddings)) as service:
        second = service.prepare("alice", "Where did Alice visit?", tenant_id="team-a")
        assert second.prompt_payload_hash == first.prompt_payload_hash
        assert second_embeddings.calls == 0


def test_config_is_portable_and_rejects_typographical_errors(tmp_path, monkeypatch):
    path = tmp_path / "service.json"
    path.write_text(json.dumps({"data_dir": "state", "semantic_build": False, "dense_retrieval": False}))
    monkeypatch.chdir(tmp_path.parent)
    config = ServiceConfig.from_file(path)
    assert config.data_dir == str(tmp_path / "state")
    build, runtime = config.profiles()
    assert runtime.retrieval.adaptive_recall
    assert runtime.query_budget.max_evidence_turns == 80
    assert "artifacts" not in canonical_json(asdict_profile(runtime))
    with pytest.raises(ValueError, match="unknown profile"):
        replace(config, runtime_overrides={"retrieval": {"adaptve_recall": True}}).profiles()
    with pytest.raises(ValueError, match="boolean"):
        ServiceConfig(semantic_build="false")


def asdict_profile(value):
    from dataclasses import asdict
    return asdict(value)


def test_cli_check_is_read_only_and_backup_reads_live_wal(memory, tmp_path, capsys):
    target = tmp_path / "not-created"
    main(["check", "--offline", "--data-dir", str(target)])
    assert json.loads(capsys.readouterr().out)["valid"]
    assert not target.exists()
    add(memory)
    backup = tmp_path / "backup.sqlite"
    main(["backup", "--data-dir", str(memory.data_dir), "--output", str(backup)])
    assert backup.is_file()
    import sqlite3
    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT COUNT(*) FROM source_turns").fetchone()[0] == 1


@pytest.fixture
def http_server(memory):
    server = make_server(memory, port=0, api_key="test-service-key")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def request(server, path, payload=None, *, token="test-service-key", tenant="team-a"):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    try:
        headers = {"Authorization": "Bearer " + token, "X-Tenant-ID": tenant,
                   "Content-Type": "application/json"}
        connection.request("GET" if payload is None else "POST", path,
                           body=None if payload is None else json.dumps(payload), headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_http_authenticated_write_read_prepare_and_errors(http_server):
    base = "/v1/memories/alice"
    assert request(http_server, "/healthz", token="")[0] == 200
    assert request(http_server, base, token="wrong")[0] == 401
    assert request(http_server, base, tenant="")[0] == 400
    payload = {"messages": MESSAGES, "session_id": "chat", "request_id": "http-1"}
    status, written = request(http_server, base + "/messages", payload)
    assert status == 200 and written["status"] == "ready"
    assert request(http_server, base)[1]["turn_count"] == 1
    assert request(http_server, base, tenant="other")[0] == 404
    status, prepared = request(http_server, base + "/prepare", {"query": QUESTION})
    assert status == 200 and "Luna" in str(prepared["messages"])
    assert request(http_server, base + "/answer", {"query": QUESTION})[0] == 503
    assert request(http_server, base + "/rebuild", {})[0] == 200
    assert request(http_server, base + "/retrieve", {"query": QUESTION, "tenant_id": "other"})[0] == 400
    payload["messages"] = [{"content": "conflicting content"}]
    assert request(http_server, base + "/messages", payload)[0] == 409


def test_nonlocal_bind_requires_service_key(memory):
    with pytest.raises(ValueError, match="GRAPHMEM_API_KEY"):
        make_server(memory, host="0.0.0.0", port=0)
