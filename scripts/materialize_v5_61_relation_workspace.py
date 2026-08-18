#!/usr/bin/env python3
"""Add a bounded source-backed relation workspace to frozen answer prompts.

This is a full-corpus, label-free experiment.  It reads only the question,
already-packed turn IDs, immutable graph facts/provenance and frozen embedding
artifacts.  Gold answers, benchmark labels and judge verdicts are never used
for routing or ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.stage import PreparedAnswer, _focus_lexical_terms  # noqa: E402
from graphmem.domain import canonical_json  # noqa: E402
from graphmem.embedding import (  # noqa: E402
    QUERY_INSTRUCTION,
    QUERY_INSTRUCTION_REVISION,
)
from graphmem.tokenization import resolve_token_counter  # noqa: E402


VERSION = "graphmem-v5.61-source-backed-relation-workspace-v1"
_QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(?P<question>[^\n]+)")
_NAMED_SPEAKER_RE = re.compile(
    r"^\[(?:CHAIN \d+ (?:step=\d+|support)|GRAPH \d+ step=\d+|"
    r"AUX \d+ rank=\d+)\]\s+\[[^\]]+\]\s+([^:\n]{1,48}):",
    re.M,
)
_GENERIC_SPEAKERS = frozenset({
    "", "assistant", "system", "tool", "user", "human",
})
_TEMPORAL_ORDER_RE = re.compile(
    r"\b(?:order|first|second|third|earliest|latest|before|after)\b", re.I)
_NUMERIC_RE = re.compile(
    r"(?:[$\u00a3\u20ac\u00a5]\s*)?\b\d+(?:[.,]\d+)?\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|half)\b",
    re.I,
)
_PLAN_RE = re.compile(
    r"\b(?:plan|planning|intend|hope|want|might|may|could|will|consider)\b",
    re.I,
)
_QUESTION_PLAN_RE = re.compile(
    r"\b(?:plan|planning|intend|want|need|recommend|suggest|could|should)\b",
    re.I,
)
_ASSISTANT_TARGET_RE = re.compile(
    r"\b(?:you (?:said|suggested|recommended|provided|gave)|your (?:answer|"
    r"suggestion|recommendation)|assistant)\b",
    re.I,
)
_GENERIC_SCOPE_TERMS = frozenset(_focus_lexical_terms(
    "how many how much total count number amount all combined something item "
    "time times ago past current currently recent recently first second third "
    "last latest earliest today yesterday day days week weeks month months year "
    "years before after between since when long take took passed elapsed"))

_ALIASES = (
    frozenset({"buy", "bought", "purchase", "purchased", "acquire", "acquired", "got", "receive", "received"}),
    frozenset({"attend", "attended", "participate", "participated", "visit", "visited", "went"}),
    frozenset({"bake", "baked", "make", "made", "cook", "cooked"}),
    frozenset({"watch", "watched", "view", "viewed", "saw", "see"}),
    frozenset({"spend", "spent", "pay", "paid", "cost", "expense", "price"}),
    frozenset({"raise", "raised", "earn", "earned", "make", "made"}),
    frozenset({"class", "classes", "lesson", "lessons", "course", "courses"}),
    frozenset({"trip", "trips", "travel", "traveled", "journey"}),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def question_from(prepared: PreparedAnswer) -> str:
    match = _QUESTION_RE.search(prepared.messages[-1]["content"])
    if match is None:
        raise ValueError(f"{prepared.question_id}: question header is absent")
    return " ".join(match.group("question").split())


def expanded_terms(text: str) -> set[str]:
    terms = set(_focus_lexical_terms(text))
    for family in _ALIASES:
        normalized = set(_focus_lexical_terms(" ".join(family)))
        if terms & normalized:
            terms.update(normalized)
    return terms


def named_transcript(prepared: PreparedAnswer) -> bool:
    evidence = prepared.messages[-1]["content"]
    return any(match.group(1).casefold().strip() not in _GENERIC_SPEAKERS
               for match in _NAMED_SPEAKER_RE.finditer(evidence))


def query_embedding_key(model_id: str, question: str) -> str:
    text = QUERY_INSTRUCTION + question
    return hashlib.sha256((
        model_id + "\n" + QUERY_INSTRUCTION_REVISION + "\n" + text
    ).encode()).hexdigest()


def bounded_source_excerpt(raw: str, start: int, end: int, limit: int = 210) -> str:
    raw = " ".join(raw.split())
    if len(raw) <= limit:
        return raw
    center = max(0, min(len(raw), (max(0, start) + max(start, end)) // 2))
    left = max(0, center - limit // 2)
    right = min(len(raw), left + limit)
    left = max(0, right - limit)
    if left:
        boundary = raw.find(" ", left, min(right, left + 32))
        if boundary >= 0:
            left = boundary + 1
    if right < len(raw):
        boundary = raw.rfind(" ", max(left, right - 32), right)
        if boundary > left:
            right = boundary
    return ("..." if left else "") + raw[left:right].strip() + (
        "..." if right < len(raw) else "")


def event_clauses(question: str) -> tuple[str, ...]:
    quoted = tuple(dict.fromkeys(
        value.strip() for value in re.findall(r"['\"]([^'\"]{4,})['\"]", question)
        if len(expanded_terms(value)) >= 2))
    if len(quoted) >= 2:
        return quoted[:5]
    among = re.search(r"\bamong\s+(.+?)(?:\?|$)", question, re.I)
    if among:
        parts = re.split(r"\s*,\s*|\s+and\s+|\s+or\s+", among.group(1))
        values = tuple(part.strip(" .?") for part in parts
                       if len(expanded_terms(part)) >= 1)
        if len(values) >= 2:
            return values[:5]
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)", question, re.I)
    if between:
        return between.group(1).strip(), between.group(2).strip()
    alternatives = re.search(r",\s*(.+?)\s+or\s+(.+?)(?:\?|$)", question, re.I)
    if alternatives:
        return alternatives.group(1).strip(), alternatives.group(2).strip()
    return ()


class WorkspaceMaterializer:
    def __init__(self, graph_db: Path, relation_db: Path, query_cache: Path,
                 model_id: str) -> None:
        self.graph = sqlite3.connect(graph_db)
        self.graph.row_factory = sqlite3.Row
        self.relations = sqlite3.connect(relation_db)
        self.relations.row_factory = sqlite3.Row
        self.queries = sqlite3.connect(query_cache)
        self.queries.row_factory = sqlite3.Row
        self.model_id = model_id
        self.turn_cache: dict[str, dict[str, sqlite3.Row]] = {}
        self.fact_cache: dict[str, list[dict[str, Any]]] = {}

    def close(self) -> None:
        self.graph.close(); self.relations.close(); self.queries.close()

    def turns(self, memory_id: str) -> dict[str, sqlite3.Row]:
        if memory_id not in self.turn_cache:
            self.turn_cache[memory_id] = {
                str(row["turn_id"]): row for row in self.graph.execute(
                    "SELECT turn_id,session_id,turn_index,speaker,role,timestamp,raw_text "
                    "FROM source_turns WHERE memory_id=?", (memory_id,))}
        return self.turn_cache[memory_id]

    def facts(self, memory_id: str) -> list[dict[str, Any]]:
        if memory_id in self.fact_cache:
            return self.fact_cache[memory_id]
        vectors = {str(row["item_id"]): np.frombuffer(
            row["vector"], dtype=np.float32, count=int(row["dimension"])).copy()
            for row in self.relations.execute(
                "SELECT item_id,dimension,vector FROM embeddings "
                "WHERE memory_id=? AND model_id=?",
                (memory_id, self.model_id))}
        result: list[dict[str, Any]] = []
        for row in self.graph.execute(
                "SELECT node_id,summary,event_time,state,confidence,attributes_json "
                "FROM graph_nodes WHERE memory_id=? AND node_type='canonical_fact'",
                (memory_id,)):
            node_id = str(row["node_id"])
            vector = vectors.get(node_id)
            if vector is None:
                continue
            attributes = json.loads(str(row["attributes_json"]))
            result.append({
                "node_id": node_id,
                "summary": str(row["summary"]),
                "event_time": str(row["event_time"] or ""),
                "state": str(row["state"] or ""),
                "confidence": float(row["confidence"]),
                "attributes": attributes,
                "vector": vector,
            })
        self.fact_cache[memory_id] = result
        return result

    def query_vector(self, question: str) -> np.ndarray | None:
        key = query_embedding_key(self.model_id, question)
        row = self.queries.execute(
            "SELECT dimension,vector FROM query_embeddings WHERE cache_key=?",
            (key,)).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(
            row["vector"], dtype=np.float32, count=int(row["dimension"])).copy()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else None

    def candidates(self, prepared: PreparedAnswer, question: str,
                   operation: str) -> list[dict[str, Any]]:
        query_vector = self.query_vector(question)
        if query_vector is None:
            return []
        packed = set(prepared.evidence_turn_ids)
        pack_rank = {turn_id: rank for rank, turn_id in enumerate(
            prepared.evidence_turn_ids)}
        turns = self.turns(prepared.memory_id)
        qterms = expanded_terms(question)
        wants_plan = bool(_QUESTION_PLAN_RE.search(question))
        wants_assistant = bool(_ASSISTANT_TARGET_RE.search(question))
        numeric_operation = operation in {
            "sum", "difference", "mean", "minimum", "maximum", "unit_rate",
            "date_difference",
        }
        result: list[dict[str, Any]] = []
        for fact in self.facts(prepared.memory_id):
            attributes = fact["attributes"]
            spans = [span for span in attributes.get("evidence_spans", ())
                     if str(span.get("turn_id") or "") in packed]
            if not spans or fact["confidence"] < 0.75:
                continue
            modality = str(attributes.get("modality") or "asserted").casefold()
            if modality in {"planned", "hypothetical"} and not wants_plan:
                continue
            span = min(spans, key=lambda value: pack_rank[str(value["turn_id"])])
            turn_id = str(span["turn_id"])
            turn = turns.get(turn_id)
            if turn is None:
                continue
            role = str(turn["role"] or turn["speaker"] or "").casefold().strip()
            # Anonymous LongMemEval source facts come from the user.  An
            # assistant recommendation that happens to share a topic is not a
            # witness unless the question explicitly asks what the assistant
            # said or recommended.
            if role == "assistant" and not wants_assistant:
                continue
            raw = str(turn["raw_text"])
            searchable = " ".join((fact["summary"], fact["state"], raw))
            fterms = expanded_terms(searchable)
            relation_terms = expanded_terms(" ".join((
                fact["summary"], fact["state"])))
            overlap = qterms & fterms
            anchor_overlap = (qterms - _GENERIC_SCOPE_TERMS) & fterms
            dense = float(np.dot(query_vector, fact["vector"]) /
                          max(1e-12, np.linalg.norm(fact["vector"])))
            if not anchor_overlap:
                continue
            if dense < 0.42 and len(anchor_overlap) < 2:
                continue
            if numeric_operation and not (
                    _NUMERIC_RE.search(searchable) or fact["event_time"]):
                continue
            score = 7.0 * dense + 0.8 * len(overlap)
            score += 0.9 if fact["state"] else 0.0
            score += 0.8 if fact["event_time"] else 0.0
            score += 0.5 / (1 + pack_rank[turn_id])
            if _PLAN_RE.search(raw) and not wants_plan:
                score -= 1.0
            result.append({
                **fact,
                "span": span,
                "turn_id": turn_id,
                "turn": turn,
                "raw": raw,
                "terms": fterms,
                "relation_terms": relation_terms,
                "overlap": overlap,
                "anchor_overlap": anchor_overlap,
                "dense": dense,
                "score": score,
            })
        return result


def select_candidates(rows: list[dict[str, Any]], question: str,
                      operation: str, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    temporal_order = bool(_TEMPORAL_ORDER_RE.search(question) and not operation)
    clauses = event_clauses(question) if temporal_order else ()
    # A generic order request without explicit operands cannot be safely
    # reconstructed from topically similar graph facts.  Keep the validated
    # source prompt byte-identical in that case.
    if temporal_order and len(clauses) < 2:
        return []
    used_nodes: set[str] = set()
    used_signatures: set[tuple[str, str]] = set()
    for clause in clauses:
        terms = expanded_terms(clause)
        content_terms = terms - _GENERIC_SCOPE_TERMS
        ranked = sorted(
            (row for row in rows
             if len(row["relation_terms"] & content_terms) >= min(2, len(content_terms))
             and row["event_time"]),
            key=lambda row: (
                -(2.0 * len(row["terms"] & terms) / max(1, len(terms))
                  + row["score"]),
                row["turn_id"], row["node_id"]))
        if not ranked:
            return []
        row = ranked[0]
        if row["node_id"] not in used_nodes:
            selected.append(row); used_nodes.add(row["node_id"])
    # A one-event duration/count route still needs a strong relation match;
    # otherwise relative-time words alone select unrelated dated facts.
    eligible_ids = {row["node_id"] for row in rows if (
        (row["dense"] >= 0.55 and len(row["anchor_overlap"]) >= 1)
        or len(row["anchor_overlap"]) >= 2)}
    if (operation == "date_difference" and not event_clauses(question)
            and not any(row["dense"] >= 0.52
                        and len(row["anchor_overlap"]) >= 2 for row in rows)):
        return []
    for row in sorted(rows, key=lambda value: (
            -value["score"], value["turn_id"], value["node_id"])):
        if row["node_id"] not in eligible_ids:
            continue
        signature = (
            row["turn_id"], re.sub(r"\W+", " ", row["summary"].casefold()).strip())
        if row["node_id"] in used_nodes or signature in used_signatures:
            continue
        # Keep at most two complementary facts from one source turn.
        if sum(item["turn_id"] == row["turn_id"] for item in selected) >= 2:
            continue
        selected.append(row); used_nodes.add(row["node_id"])
        used_signatures.add(signature)
        if len(selected) >= limit:
            break
    return selected[:limit]


def render_line(index: int, row: dict[str, Any], temporal: bool) -> str:
    turn = row["turn"]
    span = row["span"]
    excerpt = bounded_source_excerpt(
        row["raw"], int(span.get("start") or 0), int(span.get("end") or 0))
    date = row["event_time"][:10] if temporal and row["event_time"] else str(
        turn["timestamp"] or "unknown")[:10]
    return (f"R{index} [time={date}] relation={row['summary']} | "
            f"source=\"{excerpt}\"")


def apply_workspace(prepared: PreparedAnswer, materializer: WorkspaceMaterializer,
                    counter: Any, max_added_tokens: int, max_rows: int) -> PreparedAnswer:
    if not prepared.messages or named_transcript(prepared):
        return prepared
    question = question_from(prepared)
    ledger = prepared.trace.get("aggregation_ledger") or {}
    operation = str(ledger.get("operation") or "")
    temporal_order = bool(_TEMPORAL_ORDER_RE.search(question))
    if not operation and not temporal_order:
        return prepared
    rows = materializer.candidates(prepared, question, operation)
    selected = select_candidates(rows, question, operation, max_rows)
    minimum_rows = 2 if temporal_order or operation != "date_difference" else 1
    if len(selected) < minimum_rows:
        return prepared
    temporal = temporal_order or operation == "date_difference"
    header = (
        "\n\nSource-backed relation workspace (query-ranked view over already "
        "packed turns; rows are witnesses, not a complete operand set):\n")
    footer = (
        "\nBind only the exact queried entity/event and its own source time. "
        "For totals or counts, include every distinct completed qualifying "
        "occurrence, deduplicate repeated mentions, and exclude plans or "
        "assistant suggestions. Verify against the full memories, then answer "
        "the original Question concisely.")
    source_tokens = prepared.packing_prompt_tokens
    chosen: list[str] = []
    messages = [dict(row) for row in prepared.messages]
    for index, row in enumerate(selected, 1):
        candidate = render_line(index, row, temporal)
        trial = header + "\n".join((*chosen, candidate)) + footer
        trial_messages = [dict(value) for value in messages]
        trial_messages[-1]["content"] = trial_messages[-1]["content"].rstrip() + trial
        tokens = sum(counter.count(value["content"]) for value in trial_messages)
        if tokens - source_tokens > max_added_tokens:
            break
        chosen.append(candidate)
    if len(chosen) < minimum_rows:
        return prepared
    appendix = header + "\n".join(chosen) + footer
    messages[-1]["content"] = messages[-1]["content"].rstrip() + appendix
    prompt_tokens = sum(counter.count(value["content"]) for value in messages)
    trace = dict(prepared.trace)
    version = str(trace.get("prompt_version") or "")
    trace.update({
        "prompt_version": "+".join(filter(None, (version, VERSION))),
        "relation_workspace": True,
        "relation_workspace_rows": len(chosen),
        "relation_workspace_operation": operation or "temporal_order",
        "relation_workspace_token_delta": prompt_tokens - source_tokens,
        "relation_workspace_node_ids": [row["node_id"] for row in selected[:len(chosen)]],
        "relation_workspace_source_payload_hash": prepared.prompt_payload_hash,
    })
    frozen = tuple(messages)
    prompt_hash = hashlib.sha256(
        (trace["prompt_version"] + frozen[0]["content"]).encode()).hexdigest()
    payload_hash = hashlib.sha256(canonical_json(frozen).encode()).hexdigest()
    return replace(
        prepared, messages=frozen, packing_prompt_tokens=prompt_tokens,
        prompt_hash=prompt_hash, prompt_payload_hash=payload_hash, trace=trace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--relation-db", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--packing-model", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-added-tokens", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.prepared)
    counter = resolve_token_counter(args.packing_model, require_exact=True)
    materializer = WorkspaceMaterializer(
        args.graph_db, args.relation_db, args.query_cache, args.embedding_model)
    output: list[dict[str, Any]] = []
    try:
        for source in rows:
            prepared = PreparedAnswer.from_record(source)
            output.append(apply_workspace(
                prepared, materializer, counter, args.max_added_tokens,
                args.max_rows).to_record())
    finally:
        materializer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in output),
        encoding="utf-8")
    changed = [row for row, source in zip(output, rows)
               if row["prompt_payload_hash"] != source["prompt_payload_hash"]]
    deltas = [int(row["packing_prompt_tokens"])
              - int(source["packing_prompt_tokens"])
              for row, source in zip(output, rows)
              if row["prompt_payload_hash"] != source["prompt_payload_hash"]]
    manifest = {
        "schema_version": VERSION,
        "source": str(args.prepared),
        "source_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "questions": len(output),
        "changed": len(changed),
        "unchanged": len(output) - len(changed),
        "max_added_tokens": args.max_added_tokens,
        "token_delta_mean_changed": sum(deltas) / max(1, len(deltas)),
        "token_delta_max": max(deltas, default=0),
        "routes": dict(sorted({
            route: sum(1 for row in changed if row["trace"].get(
                "relation_workspace_operation") == route)
            for route in {row["trace"].get("relation_workspace_operation")
                          for row in changed}
        }.items())),
        "uses_gold_answers_or_judges": False,
        "evidence_turn_ids_and_order_frozen": all(
            row["evidence_turn_ids"] == source["evidence_turn_ids"]
            for row, source in zip(output, rows)),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
