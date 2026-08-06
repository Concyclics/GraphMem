#!/usr/bin/env python3
"""Navigate, then answer, then emit judge-ready rows.

This is the first V5 stage that produces an answer at all, so it is also the
first chance to check whether ``turn_all_hit`` -- the proxy every h0..h9
conclusion rests on -- predicts judged correctness.  ``answers.jsonl`` is
written in the field shape ``evaluate_mem0_judge.py`` expects; ``retrieval.jsonl``
carries the per-question retrieval metrics so the two can be joined afterwards
by ``analyze_v5_6_proxy_validity.py``.

Gold labels are read here, in the runner, and are never passed to
``graphmem.answer``.  The answer stage sees only the question and the graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer import AnswerConfig, AnswerStage, PROMPT_HASH  # noqa: E402
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.domain import dataclass_dict  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.eval.metrics import navigation_metrics  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="h9")
    parser.add_argument("--max-evidence-turns", type=int, default=32)
    parser.add_argument("--max-answer-tokens", type=int)
    parser.add_argument("--span-window", type=int, default=-1,
                        help="-1 renders whole turns; >=0 renders cited spans widened by N chars")
    parser.add_argument("--no-closed-form", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--answer-workers", type=int, default=32)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or args.profile
    root = args.output_root / f"v5_6_answer_{label}_{stamp}"
    root.mkdir(parents=True)

    # The authority graph is opened read-only; answers are cached in a separate
    # sidecar so a scoring run can never mutate the frozen build.
    store = SQLiteGraphStore(args.source_db, read_only=True)
    cache_db = root / "answer_cache.sqlite"
    cache_store = SQLiteGraphStore(cache_db)

    config = load_config(args.config)
    questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    if args.max_questions:
        questions = questions[:args.max_questions]

    budget = replace(config.query_budget, max_evidence_turns=args.max_evidence_turns,
                     **({"max_answer_tokens": args.max_answer_tokens}
                        if args.max_answer_tokens else {}))
    answer_config = AnswerConfig(
        span_window=None if args.span_window < 0 else args.span_window,
        closed_form_enabled=not args.no_closed_form)

    embedding = QwenEmbeddingIndex(store, config, record_usage=False) if args.embedding else None
    navigator = GraphNavigator(store, dense_search=embedding.search if embedding else None,
                               harness_profile=HarnessProfile(args.profile))

    # Navigation is single-threaded and in-process; answering is IO-bound on the
    # backbone, so only that stage fans out.
    print(f"navigating {len(questions)} questions with {args.profile}", flush=True)
    navigations = []
    for index, question in enumerate(questions, 1):
        result = navigator.navigate(question.memory_id, question.query, budget)
        navigations.append((question, result, navigation_metrics(question, result, store)))
        if index % 25 == 0:
            print(f"  navigated {index}/{len(questions)}", flush=True)

    stage = AnswerStage(store, config, "v5.6-answer", answer_config=answer_config,
                        require_exact_tokenizer=True, cache_store=cache_store)

    def run(row):
        question, result, _metric = row
        question_date = str(question.raw.get("question_date") or "") or None
        return stage.answer(question.question_id, question.query, result, budget,
                            question_date=question_date)

    print(f"answering with {args.answer_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.answer_workers)) as pool:
        answers = list(pool.map(run, navigations))

    answer_rows, retrieval_rows = [], []
    for (question, result, metric), answer in zip(navigations, answers):
        raw = question.raw
        answer_rows.append({
            "question_id": question.question_id,
            "question": question.query,
            "question_type": str(raw.get("question_type")
                                 or f"locomo_category_{raw.get('locomo_category')}"),
            "question_date": raw.get("question_date") or "",
            "answer": raw.get("answer", ""),
            "gold_answer": raw.get("answer", ""),
            "prediction": answer.prediction,
            "benchmark": question.benchmark,
            "stratum": question.stratum,
        })
        retrieval_rows.append({
            "dev_question_id": question.question_id, "stratum": question.stratum,
            "benchmark": question.benchmark, "configuration": label,
            "prompt_tokens": answer.prompt_tokens, "evidence_tokens": answer.evidence_tokens,
            "completion_tokens": answer.completion_tokens,
            "packed_turns": len(answer.evidence_turn_ids),
            "closed_form": answer.closed_form, "budget_relaxed": answer.budget_relaxed,
            "answer_warnings": list(answer.warnings),
            **{key: value for key, value in metric.items() if key != "question_id"},
        })

    (root / "answers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in answer_rows), encoding="utf-8")
    (root / "retrieval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in retrieval_rows),
        encoding="utf-8")

    tokens = sorted(row["prompt_tokens"] for row in retrieval_rows)
    manifest = {
        "profile": args.profile, "label": label, "questions": len(answer_rows),
        "source_db": str(args.source_db), "config_hash": config_hash(config),
        "answer_prompt_hash": PROMPT_HASH,
        "span_window": answer_config.span_window,
        "closed_form_enabled": answer_config.closed_form_enabled,
        "budget": dataclass_dict(budget),
        "token_counter": stage.counter.describe(),
        "prompt_tokens": {
            "mean": sum(tokens) / max(1, len(tokens)), "p50": tokens[len(tokens) // 2],
            "p95": tokens[max(0, int(0.95 * len(tokens)) - 1)], "max": max(tokens),
            "over_soft_budget": sum(1 for row in tokens if row > budget.max_answer_tokens),
        },
        "closed_form_rate": sum(row["closed_form"] for row in retrieval_rows) / max(1, len(retrieval_rows)),
        "generated_at": stamp,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {root}")
    store.close()
    cache_store.close()


if __name__ == "__main__":
    main()
