#!/usr/bin/env python3
"""Append a <=N-token query-aware rescue packet to frozen 64-turn prompts.

The selector consumes only the question, immutable source turns, and retrieval
candidate scores.  Gold annotations are used after selection for an audit in
the manifest and never influence prompt membership or ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer import AnswerConfig  # noqa: E402
from graphmem.answer.rendering import render_turn  # noqa: E402
from graphmem.domain import CandidateScore, canonical_json  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.retrieval.packer import (  # noqa: E402
    rank_query_aware_candidates,
    salient_spans,
)
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.text import content_terms  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


VERSION = "graphmem-v5.68-query-aware-rescue-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--candidate-retrieval", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--packing-model", default=(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"))
    parser.add_argument("--max-rescue-turns", type=int, default=12)
    parser.add_argument("--max-token-increase", type=int, default=500)
    parser.add_argument("--max-rescue-turn-tokens", type=int, default=120)
    parser.add_argument("--witness-rare-df", type=int, default=4)
    parser.add_argument("--memory-cache", type=int, default=16)
    return parser.parse_args()


def rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def nearest(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0


def score_from_record(row: dict, turn) -> CandidateScore:
    return CandidateScore(
        turn_id=turn.turn_id,
        session_id=turn.session_id,
        exact_score=float(row.get("exact_score", 0.0)),
        bm25_score=float(row.get("bm25_score", 0.0)),
        dense_score=float(row.get("dense_score", 0.0)),
        graph_score=float(row.get("graph_score", 0.0)),
        role_gain=0.0,
        slot_gain=0.0,
        token_cost=max(1, len(turn.raw_text.split())),
        fused_score=float(row.get("fused_score", 0.0)),
        source_channels=tuple(map(str, row.get("source_channels", ()))),
        session_score=float(row.get("session_score", 0.0)),
        adjacency_score=float(row.get("adjacency_score", 0.0)),
        operand_ids=tuple(map(str, row.get("operand_ids", ()))),
        binding_score=float(row.get("binding_score", 0.0)),
        relational_consensus_score=float(
            row.get("relational_consensus_score", 0.0)),
        mandatory=bool(row.get("mandatory", False)),
    )


def main() -> None:
    args = parse_args()
    if min(args.max_rescue_turns, args.max_token_increase,
           args.max_rescue_turn_tokens, args.witness_rare_df,
           args.memory_cache) <= 0:
        raise ValueError("all rescue budgets and cache sizes must be positive")
    questions = {
        row.question.question_id: row.question
        for row in load_full_questions(
            args.lme, args.locomo, load_gold_turns(args.gold))}
    store = SQLiteGraphStore(args.source_db, read_only=True)
    counter = resolve_token_counter(args.packing_model)
    render_config = AnswerConfig.v5_63()
    memory_cache: OrderedDict[str, tuple[dict, dict, dict, Counter]] = (
        OrderedDict())

    def memory(memory_id: str):
        cached = memory_cache.get(memory_id)
        if cached is None:
            source_turns = tuple(store.turns(memory_id))
            by_id = {turn.turn_id: turn for turn in source_turns}
            by_position = {
                (turn.session_id, turn.turn_index): turn.turn_id
                for turn in source_turns}
            terms_by_turn = {
                turn.turn_id: content_terms(turn.raw_text)
                for turn in source_turns}
            frequency = Counter(
                term for values in terms_by_turn.values() for term in values)
            cached = (by_id, by_position, terms_by_turn, frequency)
            memory_cache[memory_id] = cached
        memory_cache.move_to_end(memory_id)
        while len(memory_cache) > args.memory_cache:
            memory_cache.popitem(last=False)
        return cached

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    prompt_deltas: list[int] = []
    rescue_counts: list[int] = []
    audit = Counter()
    question_count = 0
    prepared_stream = rows(args.prepared)
    candidate_stream = rows(args.candidate_retrieval)
    with args.output.open("w", encoding="utf-8") as output:
        for prepared, retrieval in zip(prepared_stream, candidate_stream):
            question_id = str(prepared["question_id"])
            if question_id != str(retrieval["dev_question_id"]):
                raise ValueError(
                    "prepared/candidate order mismatch: "
                    f"{question_id} != {retrieval['dev_question_id']}")
            question = questions[question_id]
            by_id, by_position, terms_by_turn, frequency = memory(
                question.memory_id)
            candidates = tuple(
                score_from_record(row, by_id[str(row["turn_id"])])
                for row in retrieval.get("candidate_scores", ())
                if str(row.get("turn_id")) in by_id)
            answer_kind = str(
                prepared.get("trace", {}).get("typed_readout_kind") or "lookup")
            ranked, _head, rank_trace = rank_query_aware_candidates(
                candidates, by_id, query=question.query,
                answer_kind=answer_kind, max_turns=64,
                terms_by_turn=terms_by_turn,
                document_frequency=frequency,
                witness_rare_df=args.witness_rare_df)
            core_ids = tuple(map(str, prepared.get("evidence_turn_ids", ())))
            core_set = set(core_ids)
            messages = [dict(message) for message in prepared["messages"]]
            original_tokens = sum(counter.count(message["content"])
                                  for message in messages)
            user_content = messages[-1]["content"] if messages else ""
            blocks: list[str] = []
            rescue_ids: list[str] = []
            rendered_token_costs: list[int] = []

            def rescue_section(values: list[str]) -> str:
                if not values:
                    return ""
                return (
                    "\n\nAdditional query-linked source evidence "
                    "(verify against the same rules):\n"
                    + "\n".join(values)
                    + "\n\nRe-check the original Question using all source "
                    f"evidence above: {question.query}\n"
                    "Return only the concise final answer once.")

            for candidate in (ranked if messages else ()):
                if (candidate.turn_id in core_set
                        or candidate.turn_id in rescue_ids):
                    continue
                turn = by_id[candidate.turn_id]
                spans = salient_spans(
                    turn, question.query, answer_kind=answer_kind, max_spans=2)
                rendered = render_turn(turn, render_config, spans)
                if counter.count(rendered) > args.max_rescue_turn_tokens:
                    # Keep the best span without spending the entire rescue
                    # packet on one verbose source turn.
                    rendered = render_turn(turn, render_config, spans[:1])
                rendered_tokens = counter.count(rendered)
                if rendered_tokens > args.max_rescue_turn_tokens:
                    continue
                block = f"[RESCUE {len(blocks) + 1}] {rendered}"
                candidate_blocks = [*blocks, block]
                # Tokenize only the bounded packet while selecting.  Retokenize
                # the complete prompt once below for the exact contract check;
                # doing that for every candidate made cold LME materialization
                # quadratic in the 10K-token base prompt.
                if counter.count(rescue_section(candidate_blocks)) > (
                        args.max_token_increase - 2):
                    continue
                blocks = candidate_blocks
                rescue_ids.append(candidate.turn_id)
                rendered_token_costs.append(rendered_tokens)
                if len(rescue_ids) >= args.max_rescue_turns:
                    break

            if messages and blocks:
                messages = [*messages[:-1], {
                    "role": messages[-1]["role"],
                    "content": user_content + rescue_section(blocks),
                }]
            prompt_tokens = sum(counter.count(message["content"])
                                for message in messages)
            delta = prompt_tokens - original_tokens
            # BPE boundary effects can differ by a token or two from the packet
            # estimate.  Drop only the final rescue until the exact full-prompt
            # delta satisfies the public budget.
            while messages and blocks and delta > args.max_token_increase:
                blocks.pop()
                rescue_ids.pop()
                rendered_token_costs.pop()
                messages = [*messages[:-1], {
                    "role": messages[-1]["role"],
                    "content": user_content + rescue_section(blocks),
                }]
                prompt_tokens = sum(counter.count(message["content"])
                                    for message in messages)
                delta = prompt_tokens - original_tokens
            if delta > args.max_token_increase:
                raise AssertionError(f"{question_id}: rescue token cap exceeded")
            rescue_render_tokens = sum(rendered_token_costs)
            trace = dict(prepared.get("trace", {}))
            version = "+".join(filter(None, (
                str(trace.get("prompt_version") or ""), VERSION)))
            trace.update({
                "prompt_version": version,
                "query_aware_rescue": True,
                "query_aware_rescue_turn_ids": rescue_ids,
                "query_aware_rescue_turns": len(rescue_ids),
                "query_aware_rescue_token_delta": delta,
                "query_aware_rescue_render_tokens": rescue_render_tokens,
                "query_aware_rank": dict(rank_trace),
                "query_aware_core_turns_preserved": len(core_ids),
            })
            prepared["messages"] = messages
            prepared["evidence_turn_ids"] = [*core_ids, *rescue_ids]
            prepared["dropped_turn_ids"] = [
                turn_id for turn_id in prepared.get("dropped_turn_ids", ())
                if turn_id not in set(rescue_ids)]
            prepared["evidence_tokens"] = int(
                prepared.get("evidence_tokens", 0)) + rescue_render_tokens
            prepared["packing_prompt_tokens"] = prompt_tokens
            prepared["prompt_hash"] = (
                hashlib.sha256(
                    (version + messages[0]["content"]).encode()).hexdigest()
                if messages else str(prepared.get("prompt_hash") or ""))
            prepared["prompt_payload_hash"] = hashlib.sha256(
                canonical_json(messages).encode()).hexdigest()
            prepared["trace"] = trace
            output.write(json.dumps(prepared, ensure_ascii=True) + "\n")

            gold = {
                by_position[(ref.session_id, ref.turn_index)]
                for ref in question.gold_turns
                if (ref.session_id, ref.turn_index) in by_position}
            if gold:
                before = gold <= core_set
                after = gold <= (core_set | set(rescue_ids))
                audit["questions_with_gold"] += 1
                audit["gold_turns"] += len(gold)
                audit["core_gold_hits"] += len(gold & core_set)
                audit["rescue_gold_hits"] += len(
                    gold & (core_set | set(rescue_ids)))
                audit["core_all_hit"] += before
                audit["rescue_all_hit"] += after
                audit["all_hit_recoveries"] += after and not before
            prompt_deltas.append(delta)
            rescue_counts.append(len(rescue_ids))
            question_count += 1
            if question_count % 200 == 0:
                print(f"materialized {question_count}", flush=True)

        try:
            next(prepared_stream)
            raise ValueError("prepared stream has extra rows")
        except StopIteration:
            pass
        try:
            next(candidate_stream)
            raise ValueError("candidate stream has extra rows")
        except StopIteration:
            pass

    manifest = {
        "schema_version": VERSION,
        "prepared_source": str(args.prepared),
        "candidate_source": str(args.candidate_retrieval),
        "source_db": str(args.source_db),
        "output": str(args.output),
        "questions": question_count,
        "selection_inputs": [
            "question", "source turns", "candidate scores", "QueryIR answer kind"],
        "selection_uses_gold_answers_or_judges": False,
        "core_evidence_turns_preserved": 64,
        "max_rescue_turns": args.max_rescue_turns,
        "max_token_increase": args.max_token_increase,
        "packing_token_counter": counter.describe(),
        "rescue_turns": {
            "mean": sum(rescue_counts) / max(1, len(rescue_counts)),
            "p95": nearest(rescue_counts, 0.95),
            "max": max(rescue_counts, default=0),
        },
        "prompt_token_delta": {
            "mean": sum(prompt_deltas) / max(1, len(prompt_deltas)),
            "p95": nearest(prompt_deltas, 0.95),
            "p99": nearest(prompt_deltas, 0.99),
            "max": max(prompt_deltas, default=0),
            "percentile_method": "nearest-rank",
        },
        "gold_audit_after_selection_only": {
            **dict(audit),
            "core_recall": (
                audit["core_gold_hits"] / max(1, audit["gold_turns"])),
            "rescue_recall": (
                audit["rescue_gold_hits"] / max(1, audit["gold_turns"])),
            "core_all_hit_rate": (
                audit["core_all_hit"] / max(1, audit["questions_with_gold"])),
            "rescue_all_hit_rate": (
                audit["rescue_all_hit"] / max(1, audit["questions_with_gold"])),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    store.close()


if __name__ == "__main__":
    main()
