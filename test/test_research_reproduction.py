"""Offline integration tests; synthetic sources only, no real API calls."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import sqlite3

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from graphmem.research import astra, contexts
from graphmem.research.data import prepare, read_rows, write_rows
from graphmem.research.transport import Meter, native_usage
from graphmem.research.cli import answer, grading, report, verify_assets
from graphmem.research.encoding import base_queries
from graphmem.research.features import export
from graphmem.research.enhanced_features import export as enhance
from graphmem.research.embedding_views import jobs
from graphmem.research.retrieval import rank, FastAllocator


def tokenizer(tmp_path):
    tok = Tokenizer(
        models.WordLevel({"[UNK]": 0, "Paris": 1, "Alice": 2}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def make_source(tmp_path):
    from graphmem.eval.devset import DevQuestion, ingest_questions
    from graphmem.research.locomo_format import convert_sample
    from graphmem.build import GraphBuildPipeline
    from graphmem.storage import SQLiteGraphStore
    from graphmem.service.config import ServiceConfig

    sample = {
        "sample_id": "synthetic",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "10:00 am on 1 November, 2026",
            "session_1": [
                {"speaker": "Alice", "text": "I visited Paris.", "dia_id": "D1:1"},
                {"speaker": "Bob", "text": "Did you like the visit?", "dia_id": "D1:2"},
            ],
        },
        "qa": [
            {
                "question": "Where did Alice visit?",
                "answer": "Paris",
                "category": 4,
                "evidence": ["D1:1"],
            }
        ],
    }
    raw = tmp_path / "locomo.json"
    raw.write_text(json.dumps([sample]))
    converted = convert_sample(sample, 0)[0]
    question = DevQuestion(
        "locomo00_0000",
        "locomo:synthetic",
        "locomo",
        "single_hop",
        sample["qa"][0]["question"],
        (),
        (),
        converted,
    )
    graph = tmp_path / "original.sqlite"
    store = SQLiteGraphStore(graph)
    ingest_questions(store, [question])
    cfg, _ = ServiceConfig.offline(tmp_path / "memory").profiles()
    GraphBuildPipeline(store, dataset_hash="synthetic").build(question.memory_id, cfg)
    store.upsert_embeddings(
        question.memory_id,
        "BAAI/bge-m3",
        [
            (t.turn_id, t.content_hash, [1.0, 0.0])
            for t in store.turns(question.memory_id)
        ],
    )
    store.close()
    return raw, graph


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    raw, graph = make_source(tmp_path)
    tok = tokenizer(tmp_path)
    root = tmp_path / "run"
    prepare(root, "locomo", raw, graph, tok)
    assert "reference" not in (root / "input/queries.jsonl").read_text()
    assert "has_answer" not in (root / "input/turns.json").read_text()
    cfg = {
        "base_embedding": {"base_url": "http://localhost/v1", "model": "BAAI/bge-m3"}
    }
    response = {
        "data": [{"embedding": [1.0, 0.0]}],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: io.BytesIO(json.dumps(response).encode()),
    )
    base_queries(root, "BAAI/bge-m3", Meter(root / "embedding_calls.jsonl", cfg))
    export(root, "BAAI/bge-m3")
    views = jobs(root, "locomo")
    encoded = root / "qwen_embeddings"
    encoded.mkdir()
    np.savez_compressed(
        encoded / "batch_00000.npz",
        ids=[v["id"] for v in views],
        vectors=np.tile(np.array([1, 0], dtype=np.float16), (len(views), 1)),
    )
    (encoded / "manifest.json").write_text(json.dumps({"jobs": len(views)}))
    enhance(root, "locomo")
    rows = rank(root, "locomo")
    assert rows and rows[0]["selected_ids"]
    write_rows(
        root / "enhanced_query_usage.jsonl",
        [{"qid": "locomo00_0000", "enhanced_query_tokens": 5}],
    )
    return root


@pytest.mark.parametrize("phase", ["qwen", "astra"])
def test_synthetic_pipeline_from_graph_to_graded_report(
    corpus, tmp_path, monkeypatch, phase
):
    configurations = {
        name: {"base_url": "http://localhost/v1", "model": model}
        for name, model in [
            ("qwen", "graphmem-eval-qwen30b"),
            ("judge", "graphmem-eval-qwen30b"),
            ("astra", "gpt-6-astra"),
            ("base_embedding_locomo", "BAAI/bge-m3"),
            ("base_embedding_longmemeval", "Qwen/Qwen3-Embedding-0.6B"),
        ]
    }
    config = tmp_path / "endpoints.json"
    config.write_text(json.dumps(configurations))
    calls = []

    def respond(req, **kwargs):
        body = json.loads(req.data)
        calls.append(body)
        if req.full_url.endswith("/responses"):
            value = {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"answer":"Paris"}'}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                    "output_tokens_details": {"reasoning_tokens": 8},
                },
            }
        else:
            raw = (
                json.dumps(
                    {
                        "evidence": [
                            {"source_id": "S1T1", "quote": "I visited Paris."}
                        ],
                        "reasoning": "The person names the city.",
                        "answer": "Paris",
                    }
                )
                if body["seed"] == 20261030
                else '{"reasoning":"Matches","label":"CORRECT"}'
            )
            value = {
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            }
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    args = SimpleNamespace(
        root=corpus,
        output=tmp_path / phase,
        config=config,
        phase=phase,
        oracle=False,
        limit=0,
        workers=1,
    )
    answer(args)
    answer(args)  # Resume must use exact cached requests without another HTTP call.
    assert len(calls) == 1
    grading(args)
    result = report(corpus, args.output)
    assert (
        result["checks_pass"]
        and result["accuracy"] == 1
        and not result["full_benchmark"]
    )
    assert result["mean_all_query_model_tokens"] == 42
    assert result["mean_source_id_recall"] == 1
    with (args.output / "query_calls.jsonl").open("a") as f:
        f.write(
            json.dumps({"key": "unknown", "status": "unknown", "usage": None}) + "\n"
        )
    result = report(corpus, args.output)
    assert not result["checks_pass"] and result["mean_all_query_model_tokens"] is None


def test_unknown_transport_is_not_retried_or_counted_as_zero(tmp_path, monkeypatch):
    count = 0

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        raise TimeoutError("provider not completed")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    config = {"qwen": dict(base_url="http://localhost/v1", model="reader")}
    path = tmp_path / "calls.jsonl"
    meter = Meter(path, config)
    with pytest.raises(TimeoutError):
        meter.call("q", "answer", {"messages": []})
    resumed = Meter(path, config)
    with pytest.raises(RuntimeError, match="Unresolved"):
        resumed.call("q", "answer", {"messages": []})
    assert count == 1 and resumed.usage(["q:answer"]) is None
    assert native_usage(None) is None
    with pytest.raises(ValueError):
        native_usage({"input_tokens": 3, "output_tokens": 2, "total_tokens": 4})


def test_changed_request_cannot_reuse_cached_call(tmp_path, monkeypatch):
    response = {
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: io.BytesIO(json.dumps(response).encode()),
    )
    meter = Meter(
        tmp_path / "calls.jsonl",
        {"qwen": dict(base_url="http://localhost/v1", model="reader")},
    )
    meter.call("q", "a", {"messages": ["old"]})
    with pytest.raises(ValueError, match="Request changed"):
        meter.call("q", "a", {"messages": ["new"]})


def test_astra_topic_closure_only_adds_user_sources(tmp_path):
    tok = Tokenizer.from_file(str(tokenizer(tmp_path)))
    turns = [
        dict(
            turn_id=str(i),
            session_id="s",
            display_session=1,
            turn_index=i,
            timestamp="2026/11/01 (Sun) 10:00",
            speaker=role,
            raw_text=text,
        )
        for i, (role, text) in enumerate(
            [
                ("user", "I attended a workshop."),
                ("assistant", "Try a lecture."),
                ("user", "It was helpful."),
            ]
        )
    ]
    row = astra.prepare(
        "longmemeval",
        "How many workshops did I attend?",
        "2026/11/04 (Wed) 10:00",
        turns,
        ["0"],
        np.zeros(3),
        tok,
    )
    assert row["source_ids"] == ["0", "2"] and row["branch"] == "topic_count_sources"
    assert "Try a lecture." not in row["context"]


def test_date_resolver_preserves_source_anchor_and_uncertainty():
    from datetime import date
    from graphmem.research.readers.calendar_source_resolver import resolve

    assert resolve("last month", date(2024, 3, 1)) == (
        date(2024, 2, 1),
        date(2024, 2, 29),
    )
    assert resolve("two months ago", date(2024, 3, 31)) == (date(2024, 1, 31),) * 2


def test_release_assets_are_complete():
    result = verify_assets()
    assert result["verified"] and result["files"] > 10


def test_orphan_started_request_blocks_reissue(tmp_path, monkeypatch):
    from graphmem.research.transport import digest

    path = tmp_path / "calls.jsonl"
    body = {"messages": [], "model": "reader"}
    row = {
        "key": "q:a",
        "event": "start",
        "status": "unknown",
        "usage": None,
        "request_sha256": digest(
            {"url": "http://localhost/v1/chat/completions", "body": body}
        ),
    }
    write_rows(path, [row])
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: pytest.fail("must not reissue an orphan request"),
    )
    meter = Meter(
        path, {"qwen": {"base_url": "http://localhost/v1", "model": "reader"}}
    )
    with pytest.raises(RuntimeError, match="Unresolved"):
        meter.call("q", "a", body)


def test_concurrent_identical_call_is_recorded_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from graphmem.research.transport import read_ledger

    calls = []

    def respond(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(
            json.dumps(
                {
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    }
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", respond)
    path = tmp_path / "calls.jsonl"
    meter = Meter(
        path, {"qwen": {"base_url": "http://localhost/v1", "model": "reader"}}
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: meter.call("q", "a", {"messages": []}), range(4))
        )
    assert (
        len(calls) == 1
        and len(read_ledger(path)) == 1
        and all(r == results[0] for r in results)
    )
