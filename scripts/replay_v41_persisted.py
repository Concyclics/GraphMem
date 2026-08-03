#!/usr/bin/env python3
"""Replay current V4.1 retrieval/planner/answer from immutable persisted indexes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_v36_answers import load_indexes, rows
from graphmem_demo.clients import EmbeddingClient, OpenAICompatibleClient
from graphmem_demo.data import load_longmemeval_cases
from graphmem_demo.hierarchical_v2 import provider_token_estimate
from graphmem_demo.v4 import build_capability_view, build_query_ir
from graphmem_demo.v41 import (
    QueryPolicyV41,
    answer_messages,
    build_query_plan,
    build_sidecar,
    parse_planner_result,
    planner_messages,
    query_views,
    retrieve,
    trim_latest_addition,
)

VARIANT = "hierarchical_hybrid_graph_v4_1_query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index-run-root", type=Path)
    parser.add_argument(
        "--index-dir", type=Path,
        help=(
            "Shared persisted variant directory containing several question "
            "indexes; mutually exclusive with --index-run-root."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--context-token-budget", type=int, default=9200)
    parser.add_argument("--query-target-tokens", type=int, default=10000)
    parser.add_argument("--query-hard-limit-tokens", type=int, default=13000)
    parser.add_argument("--planner-output-tokens", type=int, default=192)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--llm-model", default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--llm-base-url", default=os.environ.get("SGAO_BASE_URL", "https://sub2api.sgao.me/v1/"))
    parser.add_argument("--llm-api-key-env", default="SGAO_API_KEY")
    parser.add_argument("--llm-request-profile", default="openai")
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    cases = load_longmemeval_cases(args.data, question_type="all")
    if (args.index_run_root is None) == (args.index_dir is None):
        raise ValueError(
            "provide exactly one of --index-run-root or --index-dir"
        )
    if args.index_dir is not None:
        if not (args.index_dir / "nodes.jsonl").exists():
            raise RuntimeError(
                f"missing shared persisted index: {args.index_dir}"
            )
        index_dirs = [args.index_dir]
    else:
        index_dirs = []
        assert args.index_run_root is not None
        for case in cases:
            directory = args.index_run_root / case.question_id / VARIANT
            if not (directory / "nodes.jsonl").exists():
                raise RuntimeError(
                    f"missing persisted index for {case.question_id}: {directory}"
                )
            index_dirs.append(directory)
    indexes = load_indexes(index_dirs, {case.question_id for case in cases})
    missing_indexes = [
        case.question_id for case in cases
        if case.question_id not in indexes
    ]
    # LoCoMo and other conversation-level memories intentionally persist one
    # graph for many questions.  Alias only when the shared directory contains
    # exactly one loaded memory; never guess among multiple question graphs.
    if missing_indexes and args.index_dir is not None and len(indexes) == 1:
        shared_index = next(iter(indexes.values()))
        for question_id in missing_indexes:
            indexes[question_id] = shared_index
        missing_indexes = []
    if missing_indexes:
        raise RuntimeError(
            f"persisted index missing {len(missing_indexes)} requested questions: "
            f"{missing_indexes[:8]}"
        )
    capability_by_object: dict[int, object] = {}
    sidecar_by_object: dict[int, object] = {}
    capabilities = {}
    sidecars = {}
    for question_id, index in indexes.items():
        identity = id(index)
        if identity not in capability_by_object:
            capability_by_object[identity] = build_capability_view(index)
            sidecar_by_object[identity] = build_sidecar(index)
        capabilities[question_id] = capability_by_object[identity]
        sidecars[question_id] = sidecar_by_object[identity]
    llm = OpenAICompatibleClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        request_profile=args.llm_request_profile,
    )
    embedder = EmbeddingClient(args.embedding_base_url, args.embedding_model)
    policy = QueryPolicyV41(
        normal_context_target=8400,
        complex_context_target=args.context_token_budget,
        planner_output_max=args.planner_output_tokens,
        query_target=args.query_target_tokens,
        query_hard_limit=args.query_hard_limit_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        name: args.output_dir / name
        for name in ("answers.jsonl", "retrieval_results.jsonl", "llm_calls.jsonl")
    }
    completed: set[str] = set()
    if args.resume:
        completed = {
            row["question_id"] for row in rows(output_files["answers.jsonl"])
        }
    else:
        for path in output_files.values():
            path.write_text("", encoding="utf-8")

    def run(case):
        started = time.perf_counter()
        index = indexes[case.question_id]
        query_ir = build_query_ir(case.question)
        plan = build_query_plan(query_ir)
        vectors = embedder.embed(
            query_views(query_ir, plan),
            question_id=case.question_id,
            variant=VARIANT,
        )
        retrieval = retrieve(
            case=case,
            variant=VARIANT,
            index=index,
            capability_view=capabilities[case.question_id],
            sidecar=sidecars[case.question_id],
            query_ir=query_ir,
            query_vectors=vectors,
            token_budget=args.context_token_budget,
            policy=policy,
        )
        call_records = []
        if retrieval.retrieval_trace.get("planner_required") is True:
            planner_response = llm.chat(
                question_id=case.question_id,
                variant=VARIANT,
                stage="answer_query_planner",
                messages=planner_messages(
                    case,
                    query_ir,
                    plan,
                    retrieval.retrieval_trace.get("v41_evidence_certificate") or {},
                    retrieval.retrieval_trace.get("v41_planner_evidence") or [],
                ),
                thinking_mode="none",
                max_tokens=args.planner_output_tokens,
                json_mode=True,
            )
            call_records.append(planner_response.record)
            planner = parse_planner_result(planner_response.text)
            offered = {
                str(row.get("source_turn_id") or "")
                for row in retrieval.retrieval_trace.get("v41_planner_evidence") or []
            }
            planner.selected_source_ids = [
                source_id for source_id in planner.selected_source_ids
                if source_id in offered
            ]
            planner.member_candidates = [
                row for row in planner.member_candidates
                if row.get("source_turn_id") in offered
            ]
            planner.slot_candidates = [
                row for row in planner.slot_candidates
                if row.get("source_turn_id") in offered
            ]
            planner.selected_source_ids = list(dict.fromkeys([
                *planner.selected_source_ids,
                *[
                    str(row.get("source_turn_id") or "")
                    for row in planner.slot_candidates
                    if row.get("source_turn_id")
                ],
            ]))[:8]
            remaining = max(
                policy.normal_context_target,
                policy.complex_context_target - planner_response.record.total_tokens,
            )
            retrieval = retrieve(
                case=case,
                variant=VARIANT,
                index=index,
                capability_view=capabilities[case.question_id],
                sidecar=sidecars[case.question_id],
                query_ir=query_ir,
                query_vectors=vectors,
                token_budget=remaining,
                policy=policy,
                planner=planner,
            )
            retrieval.retrieval_trace["planner_token_usage"] = {
                "cache_miss_input_tokens": planner_response.record.prompt_cache_miss_tokens,
                "cache_hit_input_tokens": planner_response.record.prompt_cache_hit_tokens,
                "output_tokens": planner_response.record.completion_tokens,
                "total_tokens": planner_response.record.total_tokens,
            }
        answer_messages_value = answer_messages(case, retrieval)
        planner_tokens = sum(
            record.total_tokens
            for record in call_records
            if record.stage == "answer_query_planner"
        )
        algebra = str(
            (retrieval.retrieval_trace.get("v41_query_augmentation") or {}).get(
                "answer_algebra"
            ) or ""
        )
        complex_query = bool(planner_tokens) or algebra in {
            "collection", "temporal_comparison", "state_update",
            "multi_hop_explanation",
        }
        preflight_limit = min(
            args.query_hard_limit_tokens,
            12000 if complex_query else args.query_target_tokens,
        )
        max_prompt = max(
            1000,
            preflight_limit - planner_tokens - min(512, args.max_answer_tokens),
        )
        estimate = provider_token_estimate(
            "\n".join(message.get("content", "") for message in answer_messages_value)
        )
        while estimate > max_prompt:
            if trim_latest_addition(retrieval) is None:
                break
            answer_messages_value = answer_messages(case, retrieval)
            estimate = provider_token_estimate(
                "\n".join(message.get("content", "") for message in answer_messages_value)
            )
        retrieval.retrieval_trace["v41_preflight_budget"] = {
            "provider_prompt_estimate": estimate,
            "max_prompt_tokens": max_prompt,
            "preflight_total_limit": preflight_limit,
            "complex_query": complex_query,
            "planner_tokens": planner_tokens,
            "trimmed_source_ids": retrieval.retrieval_trace.get(
                "v41_budget_trimmed_source_ids", []
            ),
        }
        response = llm.chat(
            question_id=case.question_id,
            variant=VARIANT,
            stage="answer_qa",
            messages=answer_messages_value,
            thinking_mode="none",
            max_tokens=min(512, args.max_answer_tokens),
        )
        call_records.append(response.record)
        query_records = [
            record for record in call_records
            if record.stage in {"answer_query_planner", "answer_qa"}
            and not record.excluded_from_budget
        ]
        usage = {
            "cache_miss_input_tokens": sum(r.prompt_cache_miss_tokens for r in query_records),
            "cache_hit_input_tokens": sum(r.prompt_cache_hit_tokens for r in query_records),
            "output_tokens": sum(r.completion_tokens for r in query_records),
            "reasoning_tokens": sum(r.reasoning_tokens for r in query_records),
            "total_tokens": sum(r.total_tokens for r in query_records),
        }
        usage.update({
            "over_10k": usage["total_tokens"] > 10000,
            "over_12k": usage["total_tokens"] > 12000,
            "over_13k": usage["total_tokens"] > 13000,
        })
        retrieval.retrieval_trace["v41_query_token_usage"] = usage
        retrieval.retrieval_trace["answer_mode"] = "llm_from_role_complete_evidence"
        retrieval.retrieval_trace["answer_target_budget_pass"] = (
            usage["total_tokens"] <= args.query_target_tokens
        )
        retrieval.retrieval_trace["answer_target_budget_tokens"] = args.query_target_tokens
        retrieval.retrieval_trace["answer_hard_budget_tokens"] = args.query_hard_limit_tokens
        retrieval.latency_sec = time.perf_counter() - started
        answer = {
            "question_id": case.question_id,
            "variant": VARIANT,
            "question": case.question,
            "question_type": case.question_type,
            "question_date": case.question_date,
            "gold_answer": case.answer,
            "prediction": response.text.strip(),
            "answer_session_ids": case.answer_session_ids,
        }
        return answer, asdict(retrieval), [asdict(record) for record in call_records]

    pending = [case for case in cases if case.question_id not in completed]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, case): case for case in pending}
        done = len(completed)
        for future in as_completed(futures):
            answer, retrieval, records = future.result()
            append_row(output_files["answers.jsonl"], answer)
            append_row(output_files["retrieval_results.jsonl"], retrieval)
            for record in records:
                append_row(output_files["llm_calls.jsonl"], record)
            done += 1
            print(
                f"[{done}/{len(cases)}] {answer['question_id']} "
                f"{answer['prediction'][:100]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
