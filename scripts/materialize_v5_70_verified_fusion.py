#!/usr/bin/env python3
"""Materialize V5.70 graph+flat verifier prompts over frozen Qwen memories.

Only question text, a previous answer, immutable source turns and retrieval
scores are consumed.  Gold answers, category labels and judge verdicts are not
read by selection, ranking, packing or prompt construction.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.verified_readout import build_verified_prompt  # noqa: E402
from graphmem.domain import CandidateScore, canonical_json  # noqa: E402
from graphmem.retrieval.flat_plan import (  # noqa: E402
    FLAT_PLAN_VERSION,
    build_flat_fusion_plan,
    preserve_graph_with_flat_packet,
    verification_gate,
)
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


VERSION = "graphmem-v5.70-verified-fusion-materializer-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-answers", type=Path, required=True)
    parser.add_argument("--candidate-retrieval", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--packing-model", default=(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"))
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--graph-head", type=int, default=24)
    parser.add_argument("--flat-head", type=int, default=24)
    parser.add_argument("--max-token-increase", type=int, default=500)
    parser.add_argument("--min-turns", type=int, default=32)
    parser.add_argument("--memory-cache", type=int, default=16)
    parser.add_argument("--preserve-graph-evidence", action="store_true")
    parser.add_argument("--max-flat-packet-turns", type=int, default=8)
    return parser.parse_args()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    result = {str(row["question_id"]): row for row in rows(path)}
    if not result:
        raise ValueError(f"no rows in {path}")
    return result


def score_from_record(row: dict[str, Any], turn) -> CandidateScore:
    return CandidateScore(
        turn_id=turn.turn_id,
        session_id=turn.session_id,
        exact_score=float(row.get("exact_score", 0.0)),
        bm25_score=float(row.get("bm25_score", 0.0)),
        dense_score=float(row.get("dense_score", 0.0)),
        graph_score=float(row.get("graph_score", 0.0)),
        role_gain=float(row.get("role_gain", 0.0)),
        slot_gain=float(row.get("slot_gain", 0.0)),
        token_cost=max(1, int(row.get("token_cost") or len(turn.raw_text.split()))),
        fused_score=float(row.get("fused_score", 0.0)),
        source_channels=tuple(map(str, row.get("source_channels", ()))),
        session_score=float(row.get("session_score", 0.0)),
        adjacency_score=float(row.get("adjacency_score", 0.0)),
        graph_path_ids=tuple(map(str, row.get("graph_path_ids", ()))),
        relation_contributions=tuple(map(
            str, row.get("relation_contributions", ()))),
        operand_ids=tuple(map(str, row.get("operand_ids", ()))),
        binding_score=float(row.get("binding_score", 0.0)),
        relation_path_score=float(row.get("relation_path_score", 0.0)),
        obligation_gain=float(row.get("obligation_gain", 0.0)),
        provenance_novelty=float(row.get("provenance_novelty", 0.0)),
        relational_consensus_score=float(
            row.get("relational_consensus_score", 0.0)),
        mandatory=bool(row.get("mandatory", False)),
        proof_unit_ids=tuple(map(str, row.get("proof_unit_ids", ()))),
    )


def nearest(values: Iterable[int | float], fraction: float) -> int | float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0


def stats(values: Iterable[int | float]) -> dict[str, int | float]:
    data = list(values)
    if not data:
        return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "count": len(data), "mean": sum(data) / len(data),
        "p50": nearest(data, 0.50), "p95": nearest(data, 0.95),
        "p99": nearest(data, 0.99), "max": max(data),
    }


def main() -> None:
    args = parse_args()
    if not (0 < args.min_turns <= args.max_turns):
        raise ValueError("require 0 < min_turns <= max_turns")
    if min(args.graph_head, args.flat_head, args.max_token_increase,
           args.memory_cache) <= 0:
        raise ValueError("heads, token increase and cache must be positive")
    if args.max_flat_packet_turns < 0:
        raise ValueError("max-flat-packet-turns must be non-negative")
    baseline = keyed(args.baseline_answers)
    store = SQLiteGraphStore(args.source_db, read_only=True)
    counter = resolve_token_counter(args.packing_model)
    memory_cache: OrderedDict[str, tuple[dict, tuple]] = OrderedDict()

    def memory(memory_id: str):
        cached = memory_cache.get(memory_id)
        if cached is None:
            source_turns = tuple(store.turns(memory_id))
            cached = ({turn.turn_id: turn for turn in source_turns}, source_turns)
            memory_cache[memory_id] = cached
        memory_cache.move_to_end(memory_id)
        while len(memory_cache) > args.memory_cache:
            memory_cache.popitem(last=False)
        return cached

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    route_counts: Counter[str] = Counter()
    gate_reasons: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    added_counts: list[int] = []
    removed_counts: list[int] = []
    selected_counts: list[int] = []
    prompt_deltas: list[int] = []
    supports: list[float] = []
    question_count = 0
    deterministic_count = 0
    retry_shrinks = 0

    prepared_stream = rows(args.prepared)
    retrieval_stream = rows(args.candidate_retrieval)
    with args.output.open("w", encoding="utf-8") as output:
        for prepared, retrieval in zip(prepared_stream, retrieval_stream):
            question_id = str(prepared["question_id"])
            retrieval_id = str(retrieval.get("dev_question_id") or "")
            if question_id != retrieval_id:
                raise ValueError(
                    f"prepared/retrieval order mismatch: {question_id} != {retrieval_id}")
            metadata = baseline.get(question_id)
            if metadata is None:
                raise ValueError(f"baseline answer missing {question_id}")
            memory_id = str(prepared["memory_id"])
            by_id, source_turns = memory(memory_id)
            question = " ".join(str(metadata.get("question") or "").split())
            previous = " ".join(str(metadata.get("prediction") or "").split())
            # Do not read metadata gold/category/stratum fields below this line.
            candidates = tuple(
                score_from_record(row, by_id[str(row["turn_id"])])
                for row in retrieval.get("candidate_scores", ())
                if str(row.get("turn_id")) in by_id)
            if not candidates:
                raise ValueError(f"candidate scores absent for {question_id}")
            original_messages = tuple(dict(row) for row in prepared.get("messages", ()))
            original_tokens = sum(
                counter.count(str(row.get("content") or ""))
                for row in original_messages)
            graph_ids = tuple(map(str, prepared.get("evidence_turn_ids", ())))
            plan = build_flat_fusion_plan(
                question=question, turns=by_id, candidate_scores=candidates,
                graph_turn_ids=graph_ids, max_turns=args.max_turns,
                graph_head=args.graph_head, flat_head=args.flat_head)
            if args.preserve_graph_evidence:
                plan = preserve_graph_with_flat_packet(
                    plan, graph_ids,
                    max_extra_turns=args.max_flat_packet_turns)
            graph_turns = tuple(by_id[turn_id] for turn_id in graph_ids if turn_id in by_id)
            gate = verification_gate(
                question=question, previous_answer=previous, plan=plan,
                graph_turns=graph_turns)
            source_deterministic = bool(
                prepared.get("deterministic_prediction")
                or not original_messages)
            gate_eligible = gate.eligible and not source_deterministic
            supports.append(gate.answer_support)
            route_counts[plan.route] += 1
            for reason in gate.reasons:
                gate_reasons[reason] += 1
            if source_deterministic:
                gate_reasons["source_deterministic"] += 1
            eligible_counts[
                f"{plan.route}:{'verify' if gate_eligible else 'freeze'}"] += 1

            trace = dict(prepared.get("trace", {}))
            current_version = str(trace.get("prompt_version") or "")
            trace["prompt_version"] = "+".join(filter(None, (
                current_version, VERSION)))
            trace["v5_70_flat_plan"] = dict(plan.trace)
            trace["v5_70_verification_gate"] = {
                "eligible": gate_eligible,
                "reasons": [
                    *gate.reasons,
                    *(("source_deterministic",) if source_deterministic else ()),
                ],
                "answer_support": gate.answer_support,
                "flat_novelty": gate.flat_novelty,
                "selection_uses_gold_or_judge": False,
            }

            if not gate_eligible:
                # Preserve the already-measured Luna-max answer byte-for-byte;
                # AnswerStage recognizes this as a deterministic pass-through
                # and makes no extra API request.
                record = dict(prepared)
                record.update({
                    "messages": [],
                    "closed_form": True,
                    "draft_text": previous,
                    "draft_certified": True,
                    "deterministic_prediction": previous,
                    "packing_prompt_tokens": 0,
                    "prompt_hash": hashlib.sha256(
                        f"{VERSION}:freeze:{question_id}".encode()).hexdigest(),
                    "prompt_payload_hash": hashlib.sha256(
                        f"{VERSION}:freeze:{question_id}:{previous}".encode()).hexdigest(),
                    "trace": trace,
                })
                prompt_tokens = 0
                deterministic_count += 1
            else:
                rendered = build_verified_prompt(
                    question=question,
                    question_date=str(metadata.get("question_date") or ""),
                    previous_answer=previous, plan=plan, turns=by_id,
                    max_spans=2, max_block_chars=640)
                prompt_tokens = sum(counter.count(str(row["content"]))
                                    for row in rendered.messages)
                turn_limit = args.max_turns
                flat_packet_limit = args.max_flat_packet_turns
                max_spans = 2
                max_block_chars = 640
                # Respect the public +500-token cap.  First shorten spans, then
                # trim only the weakest tail while rebuilding all provenance.
                while prompt_tokens > original_tokens + args.max_token_increase:
                    retry_shrinks += 1
                    if max_spans == 2:
                        max_spans = 1
                    elif max_block_chars > 320:
                        max_block_chars -= 80
                    elif args.preserve_graph_evidence and flat_packet_limit > 0:
                        flat_packet_limit -= 1
                    elif args.preserve_graph_evidence and max_block_chars > 160:
                        max_block_chars -= 80
                    elif args.preserve_graph_evidence:
                        break
                    elif turn_limit > args.min_turns:
                        turn_limit = max(args.min_turns, turn_limit - 4)
                    else:
                        break
                    plan = build_flat_fusion_plan(
                        question=question, turns=by_id,
                        candidate_scores=candidates, graph_turn_ids=graph_ids,
                        max_turns=turn_limit,
                        graph_head=min(args.graph_head, max(8, turn_limit // 2)),
                        flat_head=min(args.flat_head, max(8, turn_limit // 2)))
                    if args.preserve_graph_evidence:
                        plan = preserve_graph_with_flat_packet(
                            plan, graph_ids,
                            max_extra_turns=flat_packet_limit)
                    rendered = build_verified_prompt(
                        question=question,
                        question_date=str(metadata.get("question_date") or ""),
                        previous_answer=previous, plan=plan, turns=by_id,
                        max_spans=max_spans,
                        max_block_chars=max_block_chars)
                    prompt_tokens = sum(counter.count(str(row["content"]))
                                        for row in rendered.messages)
                if prompt_tokens > original_tokens + args.max_token_increase:
                    raise AssertionError(
                        f"{question_id}: verifier prompt exceeds token cap: "
                        f"{prompt_tokens} > {original_tokens + args.max_token_increase}")
                messages = tuple(dict(row) for row in rendered.messages)
                payload_hash = hashlib.sha256(canonical_json(messages).encode()).hexdigest()
                trace["v5_70_flat_plan"] = dict(plan.trace)
                trace["v5_70_verified_readout"] = dict(rendered.trace)
                trace["v5_70_original_prompt_tokens"] = original_tokens
                trace["v5_70_prompt_token_delta"] = prompt_tokens - original_tokens
                record = dict(prepared)
                record.update({
                    "messages": list(messages),
                    "evidence_turn_ids": list(rendered.evidence_turn_ids),
                    "dropped_turn_ids": [
                        turn.turn_id for turn in source_turns
                        if turn.turn_id not in frozenset(rendered.evidence_turn_ids)],
                    "evidence_tokens": sum(
                        counter.count(block) for block in rendered.evidence_blocks),
                    "packing_prompt_tokens": prompt_tokens,
                    "closed_form": False,
                    "draft_text": previous,
                    "draft_certified": False,
                    "deterministic_prediction": "",
                    "prompt_hash": hashlib.sha256(
                        (trace["prompt_version"] + messages[0]["content"]).encode()
                    ).hexdigest(),
                    "prompt_payload_hash": payload_hash,
                    "trace": trace,
                })
                added_counts.append(len(plan.added_flat_turn_ids))
                removed_counts.append(len(plan.removed_graph_turn_ids))
                selected_counts.append(len(plan.selected_turn_ids))
                prompt_deltas.append(prompt_tokens - original_tokens)
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
            question_count += 1
            if question_count % 100 == 0:
                print(f"materialized {question_count}", flush=True)

    # zip() must not silently truncate either contract.
    try:
        next(prepared_stream)
        raise ValueError("prepared has more rows than candidate retrieval")
    except StopIteration:
        pass
    try:
        next(retrieval_stream)
        raise ValueError("candidate retrieval has more rows than prepared")
    except StopIteration:
        pass
    if question_count != len(baseline):
        raise ValueError(
            f"question count {question_count} != baseline {len(baseline)}")

    manifest = {
        "schema_version": VERSION,
        "flat_plan_version": FLAT_PLAN_VERSION,
        "questions": question_count,
        "eligible_verifier_calls": question_count - deterministic_count,
        "frozen_previous_answers": deterministic_count,
        "routes": dict(route_counts),
        "gate_routes": dict(eligible_counts),
        "gate_reasons": dict(gate_reasons),
        "selection_inputs": [
            "question", "previous answer", "immutable source turns",
            "source-facing exact/BM25/dense scores"],
        "selection_uses_gold_answers_categories_or_judges": False,
        "source_prepared": str(args.prepared),
        "source_candidate_retrieval": str(args.candidate_retrieval),
        "source_db": str(args.source_db),
        "output": str(args.output),
        "max_turns": args.max_turns,
        "preserve_graph_evidence": args.preserve_graph_evidence,
        "max_flat_packet_turns": args.max_flat_packet_turns,
        "min_turns_after_token_fit": args.min_turns,
        "max_token_increase": args.max_token_increase,
        "prompt_token_delta": stats(prompt_deltas),
        "selected_turns": stats(selected_counts),
        "added_flat_turns": stats(added_counts),
        "removed_graph_turns": stats(removed_counts),
        "previous_answer_support": stats(supports),
        "token_fit_retries": retry_shrinks,
        "packing_token_counter": counter.describe(),
        "sha256": {
            "prepared": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
            "candidate_retrieval": hashlib.sha256(
                args.candidate_retrieval.read_bytes()).hexdigest(),
            "output": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
