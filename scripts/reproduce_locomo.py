#!/usr/bin/env python3
"""Reproduce the shipped Memory pipeline on all 1,540 LoCoMo category 1–4 questions.

Cold build is the default. --source-db explicitly reuses a graph snapshot;
--prepared explicitly replays frozen prompts. Neither reuses prior answers.
All model calls, including retries and judging, are logged independently.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import fcntl
import importlib.metadata
import platform
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphmem.answer import AnswerConfig, AnswerStage, PreparedAnswer
from graphmem.answer.optimized import prepare_optimized_readout
from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller
from graphmem.build.canonicalize import PredicateCanonicalizer
from graphmem.domain import canonical_json
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval.devset import ingest_questions
from graphmem.eval.fullset import load_full_questions
from graphmem.retrieval import GraphNavigator
from graphmem.service.config import ServiceConfig
from graphmem.storage import SQLiteGraphStore
from graphmem.tokenization import resolve_token_counter

JUDGE_SHA = "8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def stats(values):
    values = sorted(values)
    return {"count": len(values), "sum": sum(values),
            "mean": sum(values) / len(values) if values else 0,
            **{f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)] if values else 0
               for p in [50, 95, 99]}, "max": max(values, default=0)}


def package_versions():
    versions = {}
    for name, module in [("openai", "openai"), ("numpy", "numpy"),
                         ("tokenizers", "tokenizers"), ("faiss-cpu", "faiss"), ("hnswlib", "hnswlib")]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # Conda can install FAISS without Python distribution metadata.
            versions[name] = str(getattr(importlib.import_module(module), "__version__", "unknown"))
    return versions


def copy_graph(source_path, destination):
    """Copy a coherent snapshot but never import another run's answer cache."""
    with sqlite3.connect(f"file:{source_path.resolve()}?mode=ro", uri=True) as source:
        with sqlite3.connect(destination) as target:
            source.backup(target)
            target.execute("DELETE FROM llm_cache WHERE stage='answer'")
            target.execute("DELETE FROM llm_calls WHERE stage='answer'")


class Meter:
    """Record actual response.usage at the transport boundary, before any caches."""
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.phase = "build"

    def client(self, url, key):
        from openai import OpenAI
        client = OpenAI(base_url=url, api_key=key, max_retries=0, timeout=180)

        def wrap(function, kind):
            def call(**kwargs):
                start = time.monotonic()
                row = {"phase": self.phase, "kind": kind, "model": kwargs.get("model"),
                       "started_at": time.time(), "request_sha256": hashlib.sha256(
                           canonical_json({k: v for k, v in kwargs.items() if k != "timeout"}).encode()).hexdigest()}
                try:
                    response = function(**kwargs)
                    row["usage"] = response.usage.model_dump() if response.usage else None
                    row["finish_reasons"] = [c.finish_reason for c in getattr(response, "choices", ())]
                    row["error"] = None
                    return response
                except Exception as error:
                    row["error"] = type(error).__name__ + ": " + str(error)
                    raise
                finally:
                    row["seconds"] = time.monotonic() - start
                    with self.lock:
                        append(self.path, row)
            return call
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=wrap(client.chat.completions.create, "chat"))),
            embeddings=SimpleNamespace(create=wrap(client.embeddings.create, "embedding")))


def checkpoint(items, key, path, operation, workers):
    previous = read_rows(path)
    done = {row[key]: row for row in previous}
    if len(done) != len(previous):
        raise ValueError(f"duplicate checkpoint IDs: {path}")
    pending = [item for item in items if item[0] not in done]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(operation, item): item[0] for item in pending}
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                row = future.result()
            except Exception as error:
                append(path.with_suffix(".errors.jsonl"), {key: identifier, "error": repr(error)})
                failures.append(identifier)
                print(f"FAILED {identifier}: {error}", flush=True)
                continue
            append(path, row)
            done[identifier] = row
            if len(done) % 25 == 0 or len(items) < 25 or len(done) == len(items):
                print(f"{path.stem}: {len(done)}/{len(items)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} failures in {path.stem}; rerun to resume")
    return [done[item[0]] for item in items]


def report(output, expected):
    answers, verdicts, calls = [read_rows(output / name) for name in
                               ["answers.jsonl", "judge.jsonl", "api_calls.jsonl"]]
    groups = defaultdict(list)
    for call in calls:
        groups[call["phase"] + "/" + call["kind"]].append(call)
    usage = {}
    for name, rows in sorted(groups.items()):
        usage[name] = {"calls": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
                       "missing_usage": sum(not r.get("usage") for r in rows if not r.get("error")),
                       **{field: sum(int((r.get("usage") or {}).get(field) or 0) for r in rows)
                          for field in ["prompt_tokens", "completion_tokens", "total_tokens"]}}
    counts = defaultdict(lambda: [0, 0])
    for row in verdicts:
        counts[str(row["category"])][0] += int(row["correct"])
        counts[str(row["category"])][1] += 1
    failed_calls = sum(bool(r.get("error")) for r in calls)
    truncated = sum(r.get("finish_reason") == "length" for r in answers)
    empty = sum(not r.get("prediction", "").strip() for r in answers)
    complete = ({r["question_id"] for r in answers} == expected ==
                {r["question_id"] for r in verdicts} and len(answers) == len(verdicts) == len(expected))
    predictions = {r["question_id"]: hashlib.sha256(r.get("prediction", "").encode()).hexdigest()
                   for r in answers}
    aligned = all(r.get("prediction_sha256") == predictions.get(r["question_id"])
                  for r in verdicts)
    prepared = read_rows(output / "prepared.jsonl")
    retrieval = [r.get("trace", {}).get("retrieval", {}) for r in prepared]
    accounted = failed_calls == 0 and all(r["missing_usage"] == 0 for r in usage.values())
    result = {"expected_questions": len(expected), "answers": len(answers), "judged": len(verdicts),
              "correct": sum(r["correct"] for r in verdicts),
              "accuracy": sum(r["correct"] for r in verdicts) / len(expected),
              "by_category": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]}
                              for k, v in sorted(counts.items())},
              "complete": complete, "judge_aligned_with_answers": aligned,
              "full_locomo": len(expected) == 1540,
              "truncated_answers": truncated, "empty_answers": empty,
              "failed_api_attempts": failed_calls, "actual_api_usage": usage,
              "api_token_accounting_complete": accounted,
              "truncated_calls_by_phase": dict(Counter(r["phase"] for r in calls
                   if "length" in r.get("finish_reasons", ()))),
              "answer_tokens_per_question": {field: stats([r.get(field, 0) for r in answers])
                    for field in ["api_prompt_tokens", "completion_tokens", "api_total_tokens"]},
              "cached_answers": sum(r.get("cached", False) for r in answers),
              "deterministic_answers": sum(r.get("finish_reason") == "deterministic" for r in answers),
              "budget_note": "Actual API ledger counts retries and cold build; cached answer metadata is not charged again. Judge is separate. No n-best oracle.",
              "adaptive_turn_tiers": dict(Counter(str(r["adaptive_recall_target_turns"])
                   for r in retrieval if "adaptive_recall_target_turns" in r)),
              "adaptive_token_tiers": dict(Counter(str(r["adaptive_recall_target_tokens"])
                   for r in retrieval if "adaptive_recall_target_tokens" in r)),
              "checks_pass": complete and aligned and truncated == 0 and empty == 0 and accounted}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Converted locomo10_graphmem.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/memory-service.json")
    parser.add_argument("--judge-repo", type=Path, required=True, help="Pinned mem0ai/memory-benchmarks checkout")
    parser.add_argument("--source-db", type=Path, help="Optional frozen graph; copied by SQLite backup")
    parser.add_argument("--prepared", type=Path, help="Optional frozen prompts, for an explicit replay control")
    parser.add_argument("--stage", choices=["all", "build", "prepare", "answer", "judge", "report"], default="all")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--build-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; full run must have 1,540 questions")
    args = parser.parse_args()
    if args.workers < 1 or args.build_workers < 1 or args.limit < 0:
        parser.error("workers must be positive; limit must be nonnegative")
    if args.prepared and not args.source_db:
        parser.error("--prepared requires its --source-db")
    # One benchmark owns an endpoint throughout build/answer/judge. Mixing
    # long extraction outputs with continuous short-prefill traffic can starve
    # decoding on one GPU and cause retries with unknowable failed-call usage.
    lease_path = Path(os.environ.get("GRAPHMEM_BENCHMARK_LOCK", "/tmp/graphmem-locomo-gpu.lock"))
    lease = lease_path.open("a+")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another LoCoMo reproduction owns the local endpoint; run sequentially") from None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = replace(ServiceConfig.from_file(args.config), data_dir=str(output / "memory"))
    build, runtime = config.profiles()
    questions = load_full_questions(locomo_path=args.data)
    if args.limit:
        questions = questions[:args.limit]
    expected = {q.question_id for q in questions}
    judge_path = args.judge_repo / "benchmarks/locomo/prompts.py"
    if sha(judge_path) != JUDGE_SHA:
        raise ValueError("judge prompt checksum differs from the pinned protocol")
    spec = importlib.util.spec_from_file_location("locomo_judge_prompts", judge_path)
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    identity = {"dataset_sha256": sha(args.data), "config": config.to_dict(),
                "question_ids": sorted(expected), "judge_prompt_sha256": JUDGE_SHA,
                "source_db_sha256": sha(args.source_db) if args.source_db else None,
                "prepared_sha256": sha(args.prepared) if args.prepared else None,
                "answer_sampling": {"temperature": 0, "seed": 0, "n": 1},
                "judge_model": config.llm_model, "judge_with_evidence": False}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("run identity changed; use a new output directory")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    if args.stage == "report":
        if not report(output, expected)["checks_pass"]:
            raise SystemExit(1)
        return
    environment = {"python": platform.python_version(), "packages": package_versions()}
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    meter = Meter(output / "api_calls.jsonl")
    chat = meter.client(config.llm_base_url, os.environ.get(config.llm_api_key_env, "local"))
    embed = meter.client(config.embedding_base_url, os.environ.get(config.embedding_api_key_env, "local"))
    db = output / "memory.sqlite"
    if not db.exists() and args.source_db:
        copy_graph(args.source_db, db)
    store = SQLiteGraphStore(db)
    if args.source_db and not args.prepared and not store._read_one(
            "SELECT 1 FROM embeddings WHERE model_id=? LIMIT 1", (config.embedding_model,)):
        raise ValueError("source graph embedding differs from the product config; cold rebuild required")
    counter = resolve_token_counter(config.llm_model, require_exact=True)
    embedding = QwenEmbeddingIndex(store, build, client=embed, **runtime.retrieval.embedding_options())
    stage = AnswerStage(store, build, "locomo-product-reproduction", client=chat,
                        answer_config=AnswerConfig.v5_73(max_output_tokens=config.max_output_tokens),
                        answer_model=config.llm_model, answer_request_profile=config.answer_request_profile,
                        require_exact_tokenizer=True)
    if args.stage in {"all", "build"} and not args.prepared:
        meter.phase = "build"
        memories = sorted({q.memory_id for q in questions})
        if not args.source_db:
            ingest_questions(store, questions)

        def build_memory(item):
            memory_id, _ = item
            embedding.index_memory(memory_id)
            if not args.source_db:
                pipeline = GraphBuildPipeline(
                    store, dataset_hash=identity["dataset_sha256"],
                    distiller=QwenSemanticDistiller(store, build, identity["dataset_sha256"],
                                client=chat, worker_limit=4, request_profile=config.request_profile),
                    predicate_canonicalizer=PredicateCanonicalizer(store, build, client=embed),
                    coarsen_vector_provider=embedding.embed_graph_nodes,
                    relation_vector_provider=embedding.embed_graph_nodes)
                pipeline.build(memory_id, build)
            embedding.embed_graph_nodes(memory_id, store.nodes(memory_id))
            return {"memory_id": memory_id, "graph_version": store.graph_version(memory_id),
                    "turns": len(store.turns(memory_id)), "reused_graph": bool(args.source_db)}
        checkpoint([(m, None) for m in memories], "memory_id", output / "built.jsonl",
                   build_memory, args.build_workers)
    if args.stage in {"all", "prepare"}:
        meter.phase = "prepare"
        if args.prepared:
            frozen = [r for r in read_rows(args.prepared) if r["question_id"] in expected]
            if {r["question_id"] for r in frozen} != expected or len(frozen) != len(expected):
                raise ValueError("frozen prompts do not cover the selected questions exactly")
            (output / "prepared.jsonl").write_text("".join(json.dumps(r) + "\n" for r in frozen))
        else:
            embedding.prewarm_queries([q.query for q in questions])
            navigator = GraphNavigator(
                store, token_counter=counter, dense_search=embedding.search,
                dense_search_many=embedding.search_many, fact_dense_search=embedding.search_items,
                predicate_dense_search=lambda memory_id, query, ids, limit: embedding.search_items(
                    memory_id, query, ids, limit, index_model_id=embedding.model_id + ":predicate-v1"),
                **runtime.retrieval.navigator_options())
            turns = {m: {t.turn_id: t for t in store.turns(m)} for m in {q.memory_id for q in questions}}

            def prepare(item):
                _, q = item
                nav = navigator.navigate(q.memory_id, q.query, runtime.query_budget)
                date = q.raw.get("question_date")
                prepared = stage.prepare(q.question_id, q.query, nav, runtime.query_budget, question_date=date)
                if config.readout == "v5_81":
                    prepared = prepare_optimized_readout(
                        prepared, question=q.query, question_date=date, turns=turns[q.memory_id],
                        navigation=nav, counter=counter, embedding=embedding,
                        max_prompt_tokens=runtime.query_budget.max_answer_tokens_hard)
                row = prepared.to_record()
                row["trace"]["retrieval"] = dict(nav.trace)
                return row
            checkpoint([(q.question_id, q) for q in questions], "question_id", output / "prepared.jsonl",
                       prepare, args.workers)
    if args.stage in {"all", "answer"}:
        meter.phase = "answer"
        prepared = read_rows(output / "prepared.jsonl")
        if {r["question_id"] for r in prepared} != expected:
            raise ValueError("prepare stage incomplete")
        checkpoint([(r["question_id"], r) for r in prepared], "question_id", output / "answers.jsonl",
                   lambda item: asdict(stage.complete(PreparedAnswer.from_record(item[1]))), args.workers)
    if args.stage in {"all", "judge"}:
        meter.phase = "judge"
        answers = {r["question_id"]: r for r in read_rows(output / "answers.jsonl")}
        if set(answers) != expected:
            raise ValueError("answer stage incomplete")

        def judge(item):
            identifier, q = item
            category = int(q.raw["locomo_category"])
            prediction = answers[identifier]["prediction"]
            messages = [{"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompts.get_judge_prompt(
                            category, q.query, prompts.preprocess_answer(category, str(q.raw["answer"])), prediction)}]
            for attempt in range(2):
                response = chat.chat.completions.create(
                    model=config.llm_model, temperature=0, seed=0, max_tokens=512,
                    response_format={"type": "json_object"}, messages=messages)
                raw = response.choices[0].message.content or ""
                try:
                    verdict = json.loads(raw)
                    label = str(verdict.get("label", "")).strip().upper()
                except (ValueError, AttributeError):
                    label = ""
                if label in {"CORRECT", "WRONG"} and response.choices[0].finish_reason == "stop":
                    break
                if attempt:
                    raise ValueError(f"invalid or truncated judge response: {raw[:1000]!r}")
                # Format repair only, independent of the desired verdict.
                messages[-1] = {**messages[-1], "content": messages[-1]["content"] +
                    '\nReturn exactly {"reasoning": "one sentence", "label": "CORRECT or WRONG"}. '
                    'Both fields are required. Use one of the two labels.'}
            return {"question_id": identifier, "category": category,
                    "correct": label == "CORRECT", "reasoning": verdict.get("reasoning"),
                    "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest()}
        checkpoint([(q.question_id, q) for q in questions], "question_id", output / "judge.jsonl", judge, args.workers)
    store.close()
    if args.stage in {"all", "judge"} and not report(output, expected)["checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
