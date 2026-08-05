#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import EmbeddingClient, OpenAICompatibleClient  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases  # noqa: E402
from graphmem_demo.v3.direct_answer import (  # noqa: E402
    direct_lossless_answer_messages,
    scalar_delta_proposal,
)
from graphmem_demo.v3.graph_recovery import (  # noqa: E402
    PersistedGraphStore,
    RecoveryResult,
)
from graphmem_demo.v3.llm_navigation import (  # noqa: E402
    NavigationPlan,
    compact_proposal_messages,
    deterministic_navigation_plan,
    ir_guided_recovery_seeds,
    is_aggregate_navigation_operation,
    navigated_answer_messages,
    navigation_messages,
    parse_navigation_plan,
    recovered_evidence_text,
    session_diverse_recovery_seeds,
    verification_messages,
)
from graphmem_demo.v3.retrieval import authoritative_catalog_answer  # noqa: E402
from graphmem_demo.v3.session_navigation import (  # noqa: E402
    dense_reranked_lossless_session_rows,
    routed_lossless_session_rows,
    scope_rows_to_sessions,
)
from graphmem_demo.v3.session_llm_navigation import (  # noqa: E402
    parse_session_navigation_result,
    session_navigation_messages,
)
from graphmem_demo.v3.structured_navigation import (  # noqa: E402
    authoritative_trace_answer,
    build_query_ir,
    canonical_operation,
    certified_trace_hint,
    materialize_multiview_rows,
    merged_relations,
    structured_navigation_summary,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    """Durably persist one completed question before the batch finishes."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _close_client(client: OpenAICompatibleClient) -> None:
    """Release per-question HTTP pools so long runs do not leak sockets."""
    close = getattr(getattr(client, "client", None), "close", None)
    if callable(close):
        close()


def _merge_evidence_rows(
    base_rows: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge by node identity while allowing a stronger late view to replace one."""

    merged: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in [*base_rows, *additions]:
        node_id = str(row.get("node_id") or "")
        if node_id:
            merged[node_id] = row
        else:
            anonymous.append(row)
    return [*merged.values(), *anonymous]


def _nested_provenance_ids(value: Any) -> tuple[str, ...]:
    """Collect only explicit node-ID arrays from a trace certificate."""
    found: list[str] = []

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, (list, tuple)):
            if key in {
                "source_turn_ids", "operand_ids", "operator_operand_node_ids"
            }:
                found.extend(str(child) for child in item if str(child))
            else:
                for child in item:
                    visit(child, key)

    visit(value)
    return tuple(dict.fromkeys(found))


def _bound_provenance_ids(
    rows: list[dict[str, Any]], query_ir: Any
) -> tuple[str, ...]:
    """Pin certificate sources only when they bind the current semantic query."""
    terms = {
        re.sub(r"[^a-z0-9]+", "", str(value).casefold())
        for value in [*query_ir.subjects, *query_ir.content_terms]
        if len(re.sub(r"[^a-z0-9]+", "", str(value).casefold())) > 2
    }
    pinned: list[str] = []
    for row in rows:
        compact = re.sub(
            r"[^a-z0-9]+", "", str(row.get("text") or "").casefold()
        )
        hits = sum(term in compact for term in terms)
        if hits >= 2:
            pinned.append(str(row.get("node_id") or ""))
    return tuple(value for value in dict.fromkeys(pinned) if value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-answer saved V3 retrieval with LLM relation selection and "
            "validated local graph recovery."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SGAO_BASE_URL", "https://sub2api.sgao.me/v1/"),
    )
    parser.add_argument("--api-key-env", default="SGAO_API_KEY")
    parser.add_argument("--variant", default="v3_5_structured_graph_navigator")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--session-dense-rerank", action="store_true")
    parser.add_argument("--session-navigation-candidate", action="store_true")
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
    )
    parser.add_argument("--embedding-api-key-env", default="EMBEDDING_API_KEY")
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument(
        "--navigation-mode",
        choices=("deterministic", "llm", "session-llm"),
        default="deterministic",
        help=(
            "Use benchmark-neutral Query IR and local ranking by default. "
            "The legacy LLM selector remains available for controlled A/B."
        ),
    )
    parser.add_argument(
        "--session-navigation",
        action="store_true",
        help="Experimentally expand query-focused L0 turns from coarse routed sessions.",
    )
    parser.add_argument(
        "--answer-prompt-profile",
        choices=("audit", "direct"),
        default="audit",
        help="Use the full audit prompt or a minimal lossless-only answer contract.",
    )
    parser.add_argument("--navigation-max-tokens", type=int, default=384)
    parser.add_argument(
        "--self-proposal",
        action="store_true",
        help=(
            "Generate a compact fallible proposal from this V3 graph before "
            "lossless evidence verification."
        ),
    )
    parser.add_argument("--proposal-max-tokens", type=int, default=192)
    parser.add_argument("--proposal-context-rough-tokens", type=int, default=2200)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument(
        "--answer-budget-tokens", type=int, default=10000,
        help="Reported target budget; small overruns are listed explicitly.",
    )
    parser.add_argument(
        "--answer-hard-budget-tokens", type=int, default=10500,
        help="Hard per-question stop above the normal target budget.",
    )
    parser.add_argument("--answer-context-rough-tokens", type=int, default=7600)
    parser.add_argument(
        "--deterministic-max-selected",
        type=int,
        default=32,
        help="Maximum deterministic evidence seeds before provenance expansion.",
    )
    parser.add_argument(
        "--fallback-rows",
        type=int,
        default=12,
        help="Session-diverse base-ledger rows retained after graph recovery.",
    )
    parser.add_argument(
        "--recovery-seed-extra",
        type=int,
        default=0,
        help=(
            "Optional number of additional routed-session graph seeds for "
            "aggregate operations. Zero preserves the selective frontier."
        ),
    )
    parser.add_argument(
        "--aggregate-expanded-recovery",
        action="store_true",
        help="Use larger graph/lexical row caps for aggregate operations.",
    )
    parser.add_argument(
        "--verify-max-tokens",
        type=int,
        default=0,
        help="When positive, run one bounded evidence verifier call after the draft answer.",
    )
    parser.add_argument("--verify-context-rough-tokens", type=int, default=1500)
    parser.add_argument(
        "--evidence-profile",
        choices=("mixed", "lossless-first"),
        default="mixed",
    )
    parser.add_argument(
        "--recovery-policy",
        choices=("always", "missing-only", "never"),
        default="always",
        help="Apply local graph recovery always, only for reported missing slots, or never.",
    )
    return parser.parse_args()


def _run_one(
    case: Any,
    retrieval: dict[str, Any],
    graph_store: PersistedGraphStore,
    embedder: EmbeddingClient | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    client = OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        request_profile="openai",
    )
    query_ir = build_query_ir(case.question)
    trace_hint = certified_trace_hint(
        retrieval.get("retrieval_trace") or {}, query_ir
    )
    ledger = list(retrieval.get("evidence_ledger") or [])
    multiview_rows = materialize_multiview_rows(
        graph_store, case.question_id, retrieval.get("retrieval_trace") or {}, query_ir
    )
    session_rows = (
        routed_lossless_session_rows(
            graph_store=graph_store, question_id=case.question_id,
            retrieved_session_ids=retrieval.get("retrieved_session_ids") or (),
            query_ir=query_ir,
        )
        if args.session_navigation or args.navigation_mode == "session-llm"
        else []
    )
    existing_ids = {str(row.get("node_id") or "") for row in ledger}
    ledger.extend(
        row
        for row in [*session_rows, *multiview_rows]
        if str(row.get("node_id") or "") not in existing_ids
    )
    trace_rows = graph_store.evidence_rows_for_ids(
        case.question_id, _nested_provenance_ids(trace_hint)
    )
    ledger = _merge_evidence_rows(ledger, trace_rows)
    certified_provenance_ids = _bound_provenance_ids(trace_rows, query_ir)
    nav_result = None
    navigation_response = ""
    scoped_local_navigation = False
    if (
        args.navigation_mode == "deterministic"
        and args.session_dense_rerank
        and embedder is not None
    ):
        chosen_sessions = tuple(
            str(value)
            for value in (retrieval.get("retrieved_session_ids") or ())
            if str(value)
        )[:8]
        scoped_local_navigation = query_ir.intent in {
            "earliest",
            "location",
        }
        expanded_session_rows = dense_reranked_lossless_session_rows(
            graph_store=graph_store,
            embedder=embedder,
            question_id=case.question_id,
            question=case.question,
            retrieved_session_ids=chosen_sessions,
            query_ir=query_ir,
            max_sessions=8,
            semantic_seeds_per_session=3,
            max_turns_per_session=6,
        )
        if scoped_local_navigation:
            ledger = scope_rows_to_sessions(ledger, chosen_sessions)
        ledger = _merge_evidence_rows(ledger, expanded_session_rows)
    if args.navigation_mode == "session-llm":
        session_messages, valid_sessions = session_navigation_messages(
            question=case.question, question_date=case.question_date,
            session_rows=session_rows, query_ir=query_ir,
            include_candidate=args.session_navigation_candidate,
        )
        nav_result = client.chat(
            question_id=case.question_id, variant=args.variant,
            stage="answer_session_navigation", thinking_mode="none",
            messages=session_messages, max_tokens=args.navigation_max_tokens,
            json_mode=True, temperature=0.0, seed=0,
        )
        navigation_response = nav_result.text
        session_selection = parse_session_navigation_result(
            nav_result.text, valid_sessions
        )
        selected_sessions = session_selection.selected_session_ids
        session_missing = session_selection.missing_slots
        session_parse_error = session_selection.parse_error
        selected_sessions = selected_sessions or tuple(valid_sessions[:4])
        fallback_limit = 8 if (
            query_ir.set_wide or query_ir.state_sensitive
        ) else 4
        coarse_fallback = tuple(
            str(value)
            for value in (retrieval.get("retrieved_session_ids") or ())
            if str(value)
        )[:fallback_limit]
        chosen_sessions = tuple(dict.fromkeys(
            [*selected_sessions, *coarse_fallback]
        ))[:8]
        scoped_local_navigation = (
            query_ir.intent in {"earliest", "location"}
            or (
                query_ir.intent == "recommendation"
                and len(selected_sessions) <= 2
            )
        )
        expanded_session_rows = (
            dense_reranked_lossless_session_rows(
                graph_store=graph_store, embedder=embedder,
                question_id=case.question_id, question=case.question,
                retrieved_session_ids=chosen_sessions, query_ir=query_ir,
                max_sessions=8, semantic_seeds_per_session=3,
                max_turns_per_session=6,
            )
            if args.session_dense_rerank and embedder is not None
            else routed_lossless_session_rows(
                graph_store=graph_store, question_id=case.question_id,
                retrieved_session_ids=chosen_sessions, query_ir=query_ir,
                max_sessions=8, max_turns_per_session=8,
                seed_turns_per_session=4,
            )
        )
        if scoped_local_navigation:
            ledger = scope_rows_to_sessions(ledger, chosen_sessions)
        ledger = _merge_evidence_rows(ledger, expanded_session_rows)
        deterministic_plan = deterministic_navigation_plan(
            question=case.question, evidence_ledger=ledger, query_ir=query_ir,
            max_selected=32 if scoped_local_navigation else 64,
            include_adjacent_context=scoped_local_navigation,
        )
        raw_plan = NavigationPlan(
            selected_ids=deterministic_plan.selected_ids,
            operation=deterministic_plan.operation, missing_slots=session_missing,
            needed_relations=deterministic_plan.needed_relations,
            parse_error=session_parse_error,
            resolved_slots=session_selection.resolved_slots,
            candidate_answer=session_selection.candidate_answer,
            confidence=session_selection.confidence,
        )
        valid_ids = list(raw_plan.selected_ids)
    elif args.navigation_mode == "llm":
        nav_messages, valid_ids = navigation_messages(
            question=case.question,
            question_date=case.question_date,
            evidence_ledger=ledger,
            query_ir=query_ir,
        )
        nav_result = client.chat(
            question_id=case.question_id,
            variant=args.variant,
            stage="answer_navigation",
            thinking_mode="none",
            messages=nav_messages,
            max_tokens=args.navigation_max_tokens,
            json_mode=True,
            temperature=0.0,
            seed=0,
        )
        navigation_response = nav_result.text
        raw_plan = parse_navigation_plan(nav_result.text, valid_ids)
    else:
        raw_plan = deterministic_navigation_plan(
            question=case.question,
            evidence_ledger=ledger,
            query_ir=query_ir,
            max_selected=args.deterministic_max_selected,
            include_adjacent_context=scoped_local_navigation,
        )
        valid_ids = list(raw_plan.selected_ids)
        if args.self_proposal:
            proposal_messages, proposal_valid_ids = compact_proposal_messages(
                question=case.question,
                question_date=case.question_date,
                evidence_ledger=ledger,
                query_ir=query_ir,
                operator_hint=trace_hint,
                max_candidates=args.deterministic_max_selected,
                max_prompt_rough_tokens=args.proposal_context_rough_tokens,
            )
            nav_result = client.chat(
                question_id=case.question_id,
                variant=args.variant,
                stage="answer_proposal",
                thinking_mode="none",
                messages=proposal_messages,
                max_tokens=args.proposal_max_tokens,
                json_mode=True,
                temperature=0.0,
                seed=0,
            )
            navigation_response = nav_result.text
            proposal = parse_navigation_plan(
                nav_result.text, proposal_valid_ids
            )
            combined_ids = tuple(dict.fromkeys(
                [*proposal.selected_ids, *raw_plan.selected_ids]
            ))[:args.deterministic_max_selected]
            raw_plan = NavigationPlan(
                selected_ids=combined_ids,
                operation=(
                    proposal.operation
                    if proposal.operation != "unknown"
                    else raw_plan.operation
                ),
                missing_slots=proposal.missing_slots,
                needed_relations=tuple(dict.fromkeys(
                    [*proposal.needed_relations, *raw_plan.needed_relations]
                )),
                resolved_slots=proposal.resolved_slots,
                candidate_answer=proposal.candidate_answer,
                confidence=proposal.confidence,
                parse_error=proposal.parse_error,
            )
            valid_ids = list(combined_ids)
    plan = NavigationPlan(
        selected_ids=raw_plan.selected_ids,
        operation=canonical_operation(query_ir, raw_plan.operation),
        missing_slots=raw_plan.missing_slots,
        needed_relations=merged_relations(query_ir, raw_plan.needed_relations),
        parse_error=raw_plan.parse_error,
        resolved_slots=raw_plan.resolved_slots,
        candidate_answer=raw_plan.candidate_answer,
        confidence=raw_plan.confidence,
    )
    seed_budget = 12 if query_ir.set_wide else 0
    recovery_seed_ids = ir_guided_recovery_seeds(
        case.question,
        ledger,
        plan,
        query_ir,
        max_extra=max(seed_budget, args.recovery_seed_extra),
    )
    evidence_plan = NavigationPlan(
        selected_ids=tuple(dict.fromkeys(
            [*certified_provenance_ids, *recovery_seed_ids]
        )),
        operation=plan.operation,
        missing_slots=plan.missing_slots,
        needed_relations=plan.needed_relations,
        parse_error=plan.parse_error,
        resolved_slots=plan.resolved_slots,
        candidate_answer=plan.candidate_answer,
        confidence=plan.confidence,
    )
    expanded_recovery = (
        args.navigation_mode == "deterministic"
        or query_ir.set_wide
        or query_ir.state_sensitive
    )
    recovery_requested = args.recovery_policy == "always" or (
        args.recovery_policy == "missing-only" and bool(plan.missing_slots)
    )
    # Dense session reranking changes the seed ledger, not whether typed graph
    # traversal is allowed. The former conditional accidentally disabled all
    # relation expansion for most intents whenever dense reranking was enabled.
    recovery_applied = recovery_requested
    recovery = (
        graph_store.recover(
            question_id=case.question_id,
            question=case.question,
            selected_ids=recovery_seed_ids,
            missing_slots=plan.missing_slots,
            needed_relations=plan.needed_relations,
            operation=plan.operation,
            max_graph_rows=20 if expanded_recovery else 6,
            max_lexical_rows=24 if expanded_recovery else 8,
            max_source_rows=16 if expanded_recovery else 6,
            max_adjacency_rows=16 if expanded_recovery else 6,
        )
        if recovery_applied
        else RecoveryResult(
            rows=(),
            graph_rows=0,
            lexical_rows=0,
            source_rows=0,
            searched_nodes=len(graph_store.nodes_for(case.question_id)),
        )
    )
    navigation_spend = (
        int(nav_result.record.total_tokens) if nav_result is not None else 0
    )
    adaptive_evidence_budget = max(
        1800,
        min(
            max(1000, args.answer_context_rough_tokens - 500),
            args.answer_hard_budget_tokens
            - navigation_spend
            - args.answer_max_tokens
            - 1400,
        ),
    )
    evidence, closure_ids = recovered_evidence_text(
        question=case.question,
        evidence_ledger=ledger,
        plan=evidence_plan,
        recovery_rows=recovery.rows,
        max_rough_tokens=adaptive_evidence_budget,
        fallback_rows=args.fallback_rows,
        evidence_profile=args.evidence_profile,
    )
    if trace_hint is not None:
        rendered_hint = json.dumps(trace_hint, ensure_ascii=False, separators=(",", ":"))
        evidence = (
            f"[OPERATOR_PROPOSALS_VERIFY_AGAINST_LOSSLESS]\n{rendered_hint[:2400]}\n\n{evidence}"
        )
    delta_proposal = scalar_delta_proposal(case.question, evidence)
    if delta_proposal is not None:
        rendered_delta = json.dumps(
            delta_proposal, ensure_ascii=False, separators=(",", ":")
        )
        evidence = (
            "[DETERMINISTIC_QUERY_ALGEBRA_VERIFY]\n"
            f"{rendered_delta}\n\n{evidence}"
        )
    answer_result = client.chat(
        question_id=case.question_id,
        variant=args.variant,
        stage="answer_qa",
        thinking_mode="none",
        messages=(
            direct_lossless_answer_messages(
                question=case.question, question_date=case.question_date,
                evidence_text=evidence,
            )
            if args.answer_prompt_profile == "direct"
            else navigated_answer_messages(
                question=case.question, question_date=case.question_date,
                evidence_text=evidence, plan=plan,
            )
        ),
        max_tokens=args.answer_max_tokens,
        temperature=0.0,
        seed=0,
    )
    records = (
        [asdict(nav_result.record), asdict(answer_result.record)]
        if nav_result is not None
        else [asdict(answer_result.record)]
    )
    final_prediction = answer_result.text
    verification_response = ""
    if args.verify_max_tokens > 0:
        verify_result = client.chat(
            question_id=case.question_id,
            variant=args.variant,
            stage="answer_verify",
            thinking_mode="none",
            messages=verification_messages(
                question=case.question,
                question_date=case.question_date,
                draft_answer=answer_result.text,
                evidence_text=evidence,
                max_rough_tokens=args.verify_context_rough_tokens,
            ),
            max_tokens=args.verify_max_tokens,
            temperature=0.0,
            seed=0,
        )
        records.append(asdict(verify_result.record))
        verification_response = verify_result.text
        if verify_result.text.strip():
            final_prediction = verify_result.text.strip()
    authoritative_answer = authoritative_trace_answer(
        retrieval.get("retrieval_trace") or {}
    )
    if authoritative_answer is None:
        authoritative_answer = authoritative_catalog_answer(retrieval.get("retrieval_trace") or {})
    if authoritative_answer is not None:
        final_prediction = authoritative_answer
    answer_total = sum(int(row.get("total_tokens") or 0) for row in records)
    if answer_total > args.answer_hard_budget_tokens:
        _close_client(client)
        raise RuntimeError(
            f"answer hard budget exceeded for {case.question_id}: "
            f"{answer_total}>{args.answer_hard_budget_tokens}"
        )
    if any(int(row.get("reasoning_tokens") or 0) != 0 for row in records):
        _close_client(client)
        raise RuntimeError(f"reasoning tokens were returned for {case.question_id}")
    answer = {
        "question_id": case.question_id,
        "variant": args.variant,
        "question": case.question,
        "gold_answer": case.answer,
        "prediction": final_prediction,
        "draft_prediction": answer_result.text,
        "verification_applied": args.verify_max_tokens > 0,
        "authoritative_trace_override": authoritative_answer is not None,
        "answer_session_ids": case.answer_session_ids,
        "retrieved_answer_session_hit": retrieval.get("answer_session_hit", False),
        "retrieved_answer_session_all_hit": retrieval.get("answer_session_all_hit", False),
        "retrieved_answer_session_recall": retrieval.get("answer_session_recall", 0.0),
        "query_ir": structured_navigation_summary(query_ir),
        "navigation_mode": args.navigation_mode,
        "answer_prompt_profile": args.answer_prompt_profile,
        "multiview_frontier_count": len(multiview_rows),
        "routed_lossless_session_count": len(session_rows),
        "navigation_operation": plan.operation,
        "navigation_parse_error": plan.parse_error,
        "navigation_selected_ids": list(plan.selected_ids),
        "navigation_recovery_seed_ids": list(recovery_seed_ids),
        "navigation_missing_slots": list(plan.missing_slots),
        "navigation_closure_ids": closure_ids,
        "graph_recovery_count": recovery.graph_rows,
        "adjacency_recovery_count": recovery.adjacency_rows,
        "lexical_recovery_count": recovery.lexical_rows,
        "source_recovery_count": recovery.source_rows,
        "relation_filtered_edge_count": recovery.relation_filtered_edges,
        "constraint_filtered_candidate_count": recovery.constraint_filtered_candidates,
        "graph_recovery_applied": recovery_applied,
        "graph_recovery_policy": args.recovery_policy,
        "evidence_profile": args.evidence_profile,
        "answer_total_tokens": answer_total,
        "answer_target_budget_pass": answer_total <= args.answer_budget_tokens,
        "answer_hard_budget_pass": answer_total <= args.answer_hard_budget_tokens,
    }
    navigation = {
        "question_id": case.question_id,
        "graph_recovery_applied": recovery_applied,
        "graph_recovery_policy": args.recovery_policy,
        "evidence_profile": args.evidence_profile,
        "query_ir": structured_navigation_summary(query_ir),
        "navigation_mode": args.navigation_mode,
        "answer_prompt_profile": args.answer_prompt_profile,
        "multiview_frontier_count": len(multiview_rows),
        "routed_lossless_session_count": len(session_rows),
        "plan": asdict(plan),
        "recovery_seed_ids": list(recovery_seed_ids),
        "valid_candidate_ids": valid_ids,
        "navigation_response": navigation_response,
        "verification_response": verification_response,
        "closure_ids": closure_ids,
        "recovery": {
            "graph_rows": recovery.graph_rows,
            "adjacency_rows": recovery.adjacency_rows,
            "lexical_rows": recovery.lexical_rows,
            "source_rows": recovery.source_rows,
            "searched_nodes": recovery.searched_nodes,
            "relation_filtered_edges": recovery.relation_filtered_edges,
            "constraint_filtered_candidates": recovery.constraint_filtered_candidates,
            "rows": list(recovery.rows),
        },
        "selected_evidence": evidence,
    }
    _close_client(client)
    return answer, records, navigation


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = args.retrieval_results or (args.run_dir / "retrieval_results.jsonl")
    cases = load_longmemeval_cases(args.data, "all", args.max_questions)
    retrieval_by_id = {
        str(row["question_id"]): row for row in _read_jsonl(retrieval_path)
    }
    missing = [case.question_id for case in cases if case.question_id not in retrieval_by_id]
    if missing:
        raise RuntimeError(f"missing retrieval rows: {missing[:8]}")
    graph_store = PersistedGraphStore(args.run_dir)
    if graph_store.node_count == 0 or graph_store.edge_count == 0:
        raise RuntimeError(
            "run-dir does not contain a persisted V3 graph: "
            f"nodes={graph_store.node_count} edges={graph_store.edge_count} "
            f"path={args.run_dir}"
        )
    missing_graph_scopes = [
        case.question_id for case in cases if not graph_store.has_scope(case.question_id)
    ]
    if missing_graph_scopes:
        raise RuntimeError(
            "persisted V3 graph is missing question scopes: "
            f"{missing_graph_scopes[:8]}"
        )
    embedder = (
        EmbeddingClient(
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            api_key=os.environ.get(args.embedding_api_key_env),
        )
        if args.session_dense_rerank
        else None
    )
    order = {case.question_id: index for index, case in enumerate(cases)}
    checkpoint_path = args.output_dir / "question_checkpoints.jsonl"
    checkpoint_rows = _read_jsonl(checkpoint_path)
    completed_by_id: dict[str, tuple[int, dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}
    for row in checkpoint_rows:
        question_id = str(row.get("question_id") or "")
        if question_id not in order:
            continue
        answer = row.get("answer")
        records = row.get("records")
        navigation = row.get("navigation")
        if isinstance(answer, dict) and isinstance(records, list) and isinstance(navigation, dict):
            completed_by_id[question_id] = (
                order[question_id], answer, records, navigation
            )
    pending_cases = [case for case in cases if case.question_id not in completed_by_id]
    if completed_by_id:
        print(
            f"resuming graph navigator: completed={len(completed_by_id)} "
            f"pending={len(pending_cases)}",
            flush=True,
        )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _run_one, case, retrieval_by_id[case.question_id], graph_store, embedder, args
            ): case
            for case in pending_cases
        }
        for future in as_completed(futures):
            case = futures[future]
            answer, records, navigation = future.result()
            completed_by_id[case.question_id] = (
                order[case.question_id], answer, records, navigation
            )
            _append_checkpoint(
                checkpoint_path,
                {
                    "question_id": case.question_id,
                    "answer": answer,
                    "records": records,
                    "navigation": navigation,
                },
            )
            print(
                f"graph-navigator question={case.question_id} "
                f"tokens={answer['answer_total_tokens']} "
                f"closure={len(answer['navigation_closure_ids'])}",
                flush=True,
            )
    completed = sorted(completed_by_id.values(), key=lambda row: row[0])
    answers = [row[1] for row in completed]
    records = [record for row in completed for record in row[2]]
    navigations = [row[3] for row in completed]
    _write_jsonl(args.output_dir / "answers.jsonl", answers)
    _write_jsonl(args.output_dir / "llm_calls.jsonl", records)
    _write_jsonl(args.output_dir / "navigation_results.jsonl", navigations)
    if embedder is not None:
        embedding_records = [asdict(record) for record in embedder.records]
        _write_jsonl(args.output_dir / "embedding_calls.jsonl", embedding_records)
        embedding_stats = {
            "excluded_from_answer_budget": True,
            "call_count": len(embedding_records),
            "prompt_tokens": sum(
                int(row.get("prompt_tokens") or 0) for row in embedding_records
            ),
            "total_tokens": sum(
                int(row.get("total_tokens") or 0) for row in embedding_records
            ),
        }
        (args.output_dir / "embedding_token_stats.json").write_text(
            json.dumps(embedding_stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    per_question = [int(row["answer_total_tokens"]) for row in answers]
    stats = {
        "question_count": len(answers),
        "call_count": len(records),
        "prompt_cache_miss_tokens": sum(
            int(row.get("prompt_cache_miss_tokens") or 0) for row in records
        ),
        "prompt_cache_hit_tokens": sum(
            int(row.get("prompt_cache_hit_tokens") or 0) for row in records
        ),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in records
        ),
        "reasoning_tokens": sum(
            int(row.get("reasoning_tokens") or 0) for row in records
        ),
        "answer_total_tokens": sum(per_question),
        "answer_max_tokens": max(per_question, default=0),
        "answer_target_tokens": args.answer_budget_tokens,
        "answer_hard_budget_tokens": args.answer_hard_budget_tokens,
        "answer_over_target_count": sum(
            value > args.answer_budget_tokens for value in per_question
        ),
        "answer_over_hard_budget_count": sum(
            value > args.answer_hard_budget_tokens for value in per_question
        ),
        "answer_over_budget_count": sum(
            value > args.answer_budget_tokens for value in per_question
        ),
        "navigation_parse_error_count": sum(
            bool(row["navigation_parse_error"]) for row in answers
        ),
    }
    (args.output_dir / "token_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
